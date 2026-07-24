import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { truncateHead, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, formatSize } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFileSync, existsSync, statSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve, basename, join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { execSync } from "node:child_process";
import { Text } from "@earendil-works/pi-tui";

// ─── Backend Configuration ────────────────────────────────────
// DEFAULT: Gemma 4 (Ollama Cloud) with mini agent loop (crop_image tool)
// FALLBACK: GLM-4.1V-Thinking-Flash (ZAI) — single-turn, when no Ollama key

// ─── ZAI (fallback) ───────────────────────────────────────────
const ZAI_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions";
const ZAI_VISION_MODEL = "glm-4.1v-thinking-flash";

// ─── Gemma 4 (default) ────────────────────────────────────────
// Gemma 4 has a fixed 280-token image budget (Ollama hardcodes max_soft_tokens).
// Large images get aggressively compressed, losing fine details/small text.
// Solution: mini agent loop — gemma4 reverse-calls crop_image to zoom into
// specific regions, getting higher effective resolution on areas of interest.
const OLLAMA_API_URL = "https://ollama.com/api/chat";
const GEMMA4_MODEL = "gemma4:31b";
const MAX_AGENT_ROUNDS = 15;

// Appended to every prompt when using gemma4 agent loop.
const CROP_HINT =
  "\n\nIf any part of the image is too small or unclear to analyze accurately, " +
  "use the crop_image tool to zoom into that region for a clearer view. " +
  "You may crop multiple regions until you have enough detail. " +
  "When you have sufficient information, provide your final answer WITHOUT calling any tool.";

const CROP_TOOL = {
  type: "function" as const,
  function: {
    name: "crop_image",
    description:
      "Crop a rectangular region of the image to view at higher resolution. " +
      "Use when text or fine details are too small/blurry to read. " +
      "Coordinates normalized 0-1000 (per-mille of image dimensions; x=horizontal, y=vertical; origin=top-left).",
    parameters: {
      type: "object",
      properties: {
        x1: { type: "number", description: "Left boundary (0-1000)" },
        y1: { type: "number", description: "Top boundary (0-1000)" },
        x2: { type: "number", description: "Right boundary (0-1000)" },
        y2: { type: "number", description: "Bottom boundary (0-1000)" },
      },
      required: ["x1", "y1", "x2", "y2"],
    },
  },
};

// Default prompt used when no specific question is given.
const DEFAULT_SCAN_PROMPT =
  "You are a visual analysis assistant. Analyze the provided image and output a factual report.\n\n" +
  "Format your response as follows:\n" +
  "- Type: What kind of image is this (photo, screenshot, diagram, artwork, etc.)\n" +
  "- Subject: The main subject(s) or focal point(s)\n" +
  "- Content: Objects, people, and elements present (be specific)\n" +
  "- Text: Transcribe ALL visible text verbatim. If none, say \"None\"\n" +
  "- Layout: Spatial arrangement of elements (top/bottom/left/right)\n" +
  "- Details: Colors, notable features, anomalies, or anything unusual\n\n" +
  "Rules:\n" +
  "- State only what you can observe. Do not guess or infer meaning beyond the image.\n" +
  "- If you are unsure of exact text, mark it as [unclear].\n" +
  "- Be concise. No introductory or concluding remarks.";

// ─── Output truncation ────────────────────────────────────────
function truncateOutput(raw: string): string {
  const t = truncateHead(raw, { maxLines: DEFAULT_MAX_LINES, maxBytes: DEFAULT_MAX_BYTES });
  if (!t.truncated) return raw;
  const tmpFile = join(tmpdir(), `pi-pdf-${randomUUID()}.txt`);
  try { writeFileSync(tmpFile, raw, "utf8"); } catch {}
  return t.content + `\n\n[Output truncated: ${t.outputLines} of ${t.totalLines} lines (${formatSize(t.outputBytes)} of ${formatSize(t.totalBytes)}). Full content saved to: ${tmpFile}]`;
}

// ─── File helpers ─────────────────────────────────────────────
function fileToBase64(filePath: string): string {
  return readFileSync(filePath).toString("base64");
}

function mimeFromPath(filePath: string): string {
  const ext = filePath.toLowerCase().split(".").pop() || "";
  const map: Record<string, string> = {
    png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
    gif: "image/gif", webp: "image/webp", bmp: "image/bmp",
  };
  return map[ext] || "image/png";
}

function cleanResponse(text: string): string {
  return text.replace(/<\|begin_of_box\|>|<\|end_of_box\|>/g, "").trim();
}

// ─── Image passthrough block (for vision-capable models) ──────
// When the active model accepts image input, return images directly as
// content blocks instead of routing through the vision API. This lets the
// conversation model itself "see" the image — no external OCR/vision round-trip.
function imageBlock(filePath: string): { type: "image"; data: string; mimeType: string } | null {
  try {
    const buf = readFileSync(filePath);
    return { type: "image" as const, data: buf.toString("base64"), mimeType: mimeFromPath(filePath) };
  } catch {
    return null;
  }
}

// ─── API key helpers ──────────────────────────────────────────
function getZaiKey(): string {
  if (process.env.Z_AI_API_KEY) return process.env.Z_AI_API_KEY;
  try {
    const auth = JSON.parse(readFileSync(join(homedir(), ".pi", "agent", "auth.json"), "utf-8"));
    if (auth.zai?.key) return auth.zai.key;
  } catch {}
  return "";
}

function getOllamaCloudKey(): string {
  try {
    const auth = JSON.parse(readFileSync(join(homedir(), ".pi", "agent", "auth.json"), "utf-8"));
    if (auth["ollama-cloud"]?.key) return auth["ollama-cloud"].key;
  } catch {}
  return "";
}

// ─── Image cropping (PowerShell, no npm dependency) ───────────
// Coordinates normalized 0-1000. Returns path to cropped PNG.
function cropImagePS(srcPath: string, x1: number, y1: number, x2: number, y2: number): string {
  const cx1 = Math.max(0, Math.min(1000, Math.round(x1)));
  const cy1 = Math.max(0, Math.min(1000, Math.round(y1)));
  const cx2 = Math.max(0, Math.min(1000, Math.round(x2)));
  const cy2 = Math.max(0, Math.min(1000, Math.round(y2)));
  if (cx2 <= cx1 || cy2 <= cy1) throw new Error(`Invalid crop region [${x1},${y1},${x2},${y2}]`);

  const outPath = join(tmpdir(), `pi-crop-${randomUUID()}.png`);
  const psFile = join(tmpdir(), `pi-crop-${randomUUID()}.ps1`);
  const safeSrc = srcPath.replace(/'/g, "''");
  const safeOut = outPath.replace(/'/g, "''");
  const ps = `Add-Type -AssemblyName System.Drawing
$src = [System.Drawing.Image]::FromFile('${safeSrc}')
$w = $src.Width; $h = $src.Height
$px1 = [int](${cx1} * $w / 1000)
$py1 = [int](${cy1} * $h / 1000)
$px2 = [int](${cx2} * $w / 1000)
$py2 = [int](${cy2} * $h / 1000)
$cw = [Math]::Max(1, $px2 - $px1)
$ch = [Math]::Max(1, $py2 - $py1)
$crop = New-Object System.Drawing.Bitmap($cw, $ch)
$g = [System.Drawing.Graphics]::FromImage($crop)
$g.DrawImage($src, (New-Object System.Drawing.Rectangle(0,0,$cw,$ch)), $px1, $py1, $cw, $ch, ([System.Drawing.GraphicsUnit]::Pixel))
$crop.Save('${safeOut}', [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $crop.Dispose(); $src.Dispose()
`;
  writeFileSync(psFile, ps);
  try {
    execSync(`powershell -NoProfile -ExecutionPolicy Bypass -File "${psFile}"`, { stdio: "pipe", timeout: 15_000 });
    if (!existsSync(outPath)) throw new Error("Crop produced no output");
    return outPath;
  } finally {
    try { rmSync(psFile, { force: true }); } catch {}
  }
}

// ─── Vision result type ───────────────────────────────────────
type VisionResult = { ok: true; text: string; backend: string; rounds?: number } | { ok: false; error: string };

type VisionOpts = {
  images?: string[];
  video?: string;
  onUpdate?: (msg: string) => void;
  /** Skip agent loop (single gemma4 call). For batch operations like PDF OCR. */
  direct?: boolean;
};

// ─── ZAI backend (fallback) ───────────────────────────────────
async function callZai(prompt: string, opts: VisionOpts): Promise<VisionResult> {
  const apiKey = getZaiKey();
  if (!apiKey) return { ok: false, error: "ZAI API key not found" };

  const effectivePrompt = prompt.trim() || DEFAULT_SCAN_PROMPT;
  const content: any[] = [];

  for (const imgPath of opts.images || []) {
    try {
      const mime = mimeFromPath(imgPath);
      const b64 = fileToBase64(imgPath);
      content.push({ type: "image_url", image_url: { url: `data:${mime};base64,${b64}` } });
    } catch {
      return { ok: false, error: `Failed to read image: ${imgPath}` };
    }
  }

  if (opts.video) {
    try {
      const b64 = fileToBase64(opts.video);
      content.push({ type: "video_url", video_url: { url: `data:video/mp4;base64,${b64}` } });
    } catch {
      return { ok: false, error: `Failed to read video: ${opts.video}` };
    }
  }

  content.push({ type: "text", text: effectivePrompt });

  try {
    const resp = await fetch(ZAI_API_URL, {
      method: "POST",
      headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: ZAI_VISION_MODEL,
        messages: [{ role: "user", content }],
        thinking: { type: "enabled" },
      }),
    });
    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      return { ok: false, error: `ZAI API ${resp.status}: ${errText.slice(0, 200)}` };
    }
    const data = await resp.json() as any;
    const rawText = data.choices?.[0]?.message?.content;
    if (!rawText) return { ok: false, error: "Empty response from ZAI vision model" };
    return { ok: true, text: cleanResponse(rawText), backend: "zai" };
  } catch (e: any) {
    return { ok: false, error: e?.message || "ZAI vision API call failed" };
  }
}

// ─── Gemma 4 single API call ──────────────────────────────────
async function gemma4Call(
  apiKey: string,
  messages: any[],
  useTools: boolean
): Promise<any> {
  const body: any = { model: GEMMA4_MODEL, messages, stream: false };
  if (useTools) body.tools = [CROP_TOOL];
  const resp = await fetch(OLLAMA_API_URL, {
    method: "POST",
    headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const errText = await resp.text().catch(() => "");
    throw new Error(`Ollama API ${resp.status}: ${errText.slice(0, 200)}`);
  }
  return await resp.json() as any;
}

// ─── Gemma 4 direct (single call, no agent loop) ──────────────
async function callGemma4Direct(prompt: string, imagePaths: string[]): Promise<VisionResult> {
  const apiKey = getOllamaCloudKey();
  if (!apiKey) return { ok: false, error: "Ollama Cloud API key not found" };

  const effectivePrompt = prompt.trim() || DEFAULT_SCAN_PROMPT;
  const images: string[] = [];
  for (const p of imagePaths) {
    try { images.push(fileToBase64(p)); } catch { return { ok: false, error: `Failed to read image: ${p}` }; }
  }
  try {
    const data = await gemma4Call(apiKey, [{ role: "user", content: effectivePrompt, images }], false);
    const content = cleanResponse(data.message?.content || "");
    if (!content) return { ok: false, error: "Empty response from gemma4" };
    return { ok: true, text: content, backend: "gemma4", rounds: 1 };
  } catch (e: any) {
    return { ok: false, error: e?.message || "Gemma4 API call failed" };
  }
}

// ─── Gemma 4 agent loop (with crop_image tool) ────────────────
// gemma4 can reverse-call crop_image to zoom into regions.
// Loops until gemma4 returns a final answer (no tool call) or max rounds.
async function callGemma4AgentLoop(
  prompt: string,
  imagePaths: string[],
  onUpdate?: (msg: string) => void
): Promise<VisionResult> {
  const apiKey = getOllamaCloudKey();
  if (!apiKey) return { ok: false, error: "Ollama Cloud API key not found" };
  if (imagePaths.length === 0) return { ok: false, error: "No image provided" };

  const effectivePrompt = (prompt.trim() || DEFAULT_SCAN_PROMPT) + CROP_HINT;
  const mainImage = imagePaths[0];

  // Load all images as base64 (Ollama format: raw base64 array, not data URL)
  const initialImages: string[] = [];
  for (const p of imagePaths) {
    try { initialImages.push(fileToBase64(p)); }
    catch { return { ok: false, error: `Failed to read image: ${p}` }; }
  }

  const messages: any[] = [
    { role: "user", content: effectivePrompt, images: initialImages },
  ];

  const cleanupFiles: string[] = [];
  let round = 0;

  try {
    for (; round < MAX_AGENT_ROUNDS; round++) {
      onUpdate?.(`Analyzing${round > 0 ? ` (round ${round + 1})` : ""}...`);

      let data: any;
      try {
        data = await gemma4Call(apiKey, messages, true);
      } catch (e: any) {
        return { ok: false, error: e?.message || "Gemma4 API call failed" };
      }

      const msg = data.message;
      if (!msg) return { ok: false, error: "Empty response from gemma4" };

      const toolCalls = msg.tool_calls;
      const content = cleanResponse(msg.content || "");

      // No tool call → final answer
      if (!toolCalls || toolCalls.length === 0) {
        if (!content) return { ok: false, error: "Empty final response from gemma4" };
        return { ok: true, text: content, backend: "gemma4", rounds: round + 1 };
      }

      // Process tool calls — add assistant turn to history
      messages.push({ role: "assistant", content: msg.content || "", tool_calls: toolCalls });

      for (const tc of toolCalls) {
        if (tc.function?.name !== "crop_image") {
          messages.push({ role: "user", content: `Unknown tool "${tc.function?.name}". Provide your answer directly.` });
          continue;
        }
        let args: any;
        try {
          args = typeof tc.function.arguments === "string"
            ? JSON.parse(tc.function.arguments)
            : tc.function.arguments;
        } catch {
          messages.push({ role: "user", content: "Invalid crop arguments. Try again or provide your answer." });
          continue;
        }
        const { x1, y1, x2, y2 } = args;
        onUpdate?.(`Cropping [${x1},${y1},${x2},${y2}]...`);
        try {
          const croppedPath = cropImagePS(mainImage, x1, y1, x2, y2);
          cleanupFiles.push(croppedPath);
          const croppedB64 = fileToBase64(croppedPath);
          messages.push({
            role: "user",
            content: `Here is the cropped region [${x1},${y1},${x2},${y2}] at higher resolution. Continue your analysis. If you need another region, call crop_image again. When done, provide your final answer WITHOUT any tool call.`,
            images: [croppedB64],
          });
        } catch (e: any) {
          messages.push({ role: "user", content: `Crop failed for [${x1},${y1},${x2},${y2}]: ${e.message}. Try different coordinates or give your answer.` });
        }
      }
    }

    // Max rounds reached — final call WITHOUT tools to force an answer
    onUpdate?.("Finalizing analysis...");
    try {
      const data = await gemma4Call(apiKey, messages, false);
      const content = cleanResponse(data.message?.content || "");
      if (content) return { ok: true, text: content, backend: "gemma4", rounds: round };
    } catch {}
    return { ok: false, error: `gemma4 did not produce a final answer within ${MAX_AGENT_ROUNDS} rounds` };
  } finally {
    for (const f of cleanupFiles) try { rmSync(f, { force: true }); } catch {}
  }
}

// ─── Vision dispatcher ────────────────────────────────────────
// Routes to gemma4 (default) or zai (fallback) based on available keys.
async function callVision(prompt: string, opts: VisionOpts = {}): Promise<VisionResult> {
  const ollamaKey = getOllamaCloudKey();

  // Video: gemma4 doesn't support video → always zai
  if (opts.video) return callZai(prompt, opts);

  // Images: prefer gemma4
  if (ollamaKey && opts.images && opts.images.length > 0) {
    if (opts.direct) return callGemma4Direct(prompt, opts.images);
    return callGemma4AgentLoop(prompt, opts.images, opts.onUpdate);
  }

  // Fallback: zai
  return callZai(prompt, opts);
}

// ─── Error helper ─────────────────────────────────────────────
function err(msg: string) {
  return { content: [{ type: "text" as const, text: msg }], isError: true as const };
}

// ─── PDF helpers ──────────────────────────────────────────────
function hasPdftoppm(): boolean {
  try { execSync("pdftoppm -v", { stdio: "ignore" }); return true; } catch { return false; }
}

function hasPdftotext(): boolean {
  try { const r = execSync("pdftotext -v", { encoding: "utf8", stdio: ["pipe","pipe","pipe"] }); return true; } catch (e: any) { return (e.stdout || "").includes("pdftotext") || (e.stderr || "").includes("pdftotext"); }
}

function tryPdfTextExtract(pdfPath: string, pageRange?: string): { text: string; pages: number } | null {
  const tmpFile = join(tmpdir(), `pi-pdf-text-${randomUUID()}.txt`);
  try {
    let args = "";
    if (pageRange) {
      const parts = pageRange.split("-").map(Number);
      const first = parts[0] || 1;
      const last = parts[1] || parts[0] || 999;
      args = ` -f ${first} -l ${last}`;
    }
    execSync(`pdftotext${args} -layout "${pdfPath}" "${tmpFile}"`, { stdio: "ignore", timeout: 30_000 });
    if (!existsSync(tmpFile)) return null;
    const text = readFileSync(tmpFile, "utf8").trim();
    if (!text) return null;
    const lines = text.split("\n").filter(l => l.trim().length > 0);
    const totalChars = lines.reduce((sum, l) => sum + l.replace(/\s/g, "").length, 0);
    const pageCount = Math.max(1, lines.filter(l => /^\f/.test(l)).length + 1);
    const charsPerPage = totalChars / pageCount;
    if (charsPerPage < 30) return null;
    return { text, pages: pageCount };
  } catch { return null; }
  finally { try { rmSync(tmpFile, { force: true }); } catch {} }
}

function extractPDFPages(pdfPath: string, tempDir: string): string[] {
  let pageCount = 1;
  try {
    const info = execSync(`pdfinfo "${pdfPath}"`, { encoding: "utf8" });
    const match = info.match(/Pages:\s+(\d+)/);
    if (match) pageCount = parseInt(match[1], 10);
  } catch {}
  const pages = Math.min(pageCount, 20);
  execSync(`pdftoppm -png -r 150 -f 1 -l ${pages} "${pdfPath}" "${join(tempDir, "page")}"`, { stdio: "ignore" });
  const result: string[] = [];
  for (let i = 1; i <= pages; i++) {
    const p = join(tempDir, `page-${i}.png`);
    if (existsSync(p)) result.push(p);
  }
  return result;
}

// ─── Screenshot helpers ───────────────────────────────────────
const PS_ADD_TYPE = `Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class W32 { [DllImport(\\"user32.dll\\")] public static extern bool GetWindowRect(IntPtr h, out RECT r); [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L,T,R,B; } }'; Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing`;

function takeScreenshot(window?: string): string {
  const tmpFile = join(tmpdir(), `pi-screen-${randomUUID()}.png`);
  const safePath = tmpFile.replace(/'/g, "''");
  let ps: string;
  if (window) {
    const escaped = window.replace(/'/g, "''");
    ps = `${PS_ADD_TYPE}; ` +
      `$p = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -ne '' -and $_.MainWindowTitle -like '*${escaped}*' } | Select-Object -First 1;` +
      `if (-not $p) { Write-Error 'Window not found'; exit 1 };` +
      `$r = New-Object W32+RECT; [W32]::GetWindowRect($p.MainWindowHandle,[ref]$r)|Out-Null;` +
      `$w=$r.R-$r.L; $h=$r.B-$r.T;` +
      `$b=New-Object System.Drawing.Bitmap($w,$h); $g=[System.Drawing.Graphics]::FromImage($b);` +
      `$g.CopyFromScreen($r.L,$r.T,0,0,[System.Drawing.Size]::new($w,$h));` +
      `$b.Save('${safePath}'); $g.Dispose(); $b.Dispose()`;
  } else {
    ps = `${PS_ADD_TYPE}; ` +
      `$s=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;` +
      `$b=New-Object System.Drawing.Bitmap($s.Width,$s.Height); $g=[System.Drawing.Graphics]::FromImage($b);` +
      `$g.CopyFromScreen($s.Location,[System.Drawing.Point]::Empty,$s.Size);` +
      `$b.Save('${safePath}'); $g.Dispose(); $b.Dispose()`;
  }
  execSync(`powershell -Command "${ps}"`, { stdio: "pipe" });
  if (!existsSync(tmpFile)) throw new Error("Screenshot capture failed.");
  return tmpFile;
}

function listWindows(): { name: string; title: string }[] {
  try {
    const out = execSync(
      `powershell -Command "Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -ne '' } | Select-Object -First 15 Name,MainWindowTitle | ForEach-Object { \\"$( $_.Name )|$( $_.MainWindowTitle )\\" }"`,
      { encoding: "utf8" }
    );
    return out.trim().split("\n").filter(Boolean).map(line => {
      const [name, ...rest] = line.trim().split("|");
      return { name: name.trim(), title: rest.join("|").trim() };
    });
  } catch { return []; }
}

// ─── Main Extension ────────────────────────────────────────────
export default function (pi: ExtensionAPI) {
  let initialized = false;

  async function ensureReady(): Promise<string | null> {
    const ollamaKey = getOllamaCloudKey();
    const zaiKey = getZaiKey();
    if (!ollamaKey && !zaiKey)
      return "No vision API key found. Configure auth.json with 'ollama-cloud' or 'zai' key.";
    initialized = true;
    return null;
  }

  // ── Intercept images for non-vision models ──────────────────
  pi.on("input", async (event, ctx) => {
    if (ctx.model?.input?.includes("image")) return { action: "continue" };
    if (!event.images || event.images.length === 0) return { action: "continue" };

    const initErr = await ensureReady();
    if (initErr) {
      ctx.ui.notify("Vision API unavailable", "warning");
      return { action: "continue" };
    }

    ctx.ui.notify(`Analyzing ${event.images.length} image(s)...`, "info");

    const descriptions: string[] = [];
    for (let i = 0; i < event.images.length; i++) {
      try {
        const img = event.images[i];
        let imageSource: string;
        let cleanup = false;

        if (img.source?.data) {
          const tmpFile = join(tmpdir(), `pi-img-${randomUUID()}.png`);
          writeFileSync(tmpFile, Buffer.from(img.source.data, "base64"));
          imageSource = tmpFile;
          cleanup = true;
        } else if (typeof img === "string") {
          imageSource = img;
        } else if (img.source?.uri) {
          imageSource = img.source.uri;
        } else {
          descriptions.push(`[Image ${i + 1}: unsupported]`);
          continue;
        }

        const vr = await callVision(
          "Describe this image in detail, including all visible text, layout, and elements.",
          { images: [imageSource] }
        );
        if (cleanup) try { rmSync(imageSource, { force: true }); } catch {}
        descriptions.push(vr.ok ? `Image ${i + 1}: ${vr.text}` : `Image ${i + 1}: [${vr.error}]`);
      } catch (e: any) { descriptions.push(`Image ${i + 1}: [${e.message}]`); }
    }

    const prefix = event.text?.trim() || "Analyze these images:";
    ctx.ui.notify(`✓ ${event.images.length} image(s) analyzed`, "success");
    return { action: "transform", text: `${prefix}\n\n${descriptions.join("\n\n")}`, images: [] };
  });

  // ── pdf_read ────────────────────────────────────────────────
  pi.registerTool({
    name: "pdf_read",
    label: "PDF Text Reader",
    description:
      "Read and extract text from text-based PDF files using pdftotext. Fast, no vision model needed. Use for normal PDFs (papers, documents, presentations). For scanned/image-based PDFs, use pdf_read_ocr instead. Supports arbitrary page ranges (e.g. \"100-110\", \"5\"). Large outputs are truncated to 50KB with full content saved to a temp file that can be read with the read tool.",
    promptSnippet: "Read text-based PDF files using pdftotext",
    promptGuidelines: [
      "Use pdf_read to read content from PDF files when the user asks about a PDF document.",
      "Extraction is fast via pdftotext, no vision model needed. For image-based/scanned PDFs, use pdf_read_ocr instead.",
      "Supports arbitrary page ranges — you can read page 100-110 without reading the whole document.",
      "If output is truncated, a temp file path is provided. Use the read tool with offset/limit to access specific sections.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the PDF file" }),
      pages: Type.Optional(Type.String({ description: 'Page range, e.g. "1-5" or "3" or "100-110". Omit for all pages.' })),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      if (!existsSync(params.path)) return err(`PDF not found: ${params.path}`);
      if (!hasPdftotext()) return err("pdftotext not installed. Install poppler-utils.");

      const extracted = tryPdfTextExtract(params.path, params.pages);
      if (!extracted) return err("No text extracted. The PDF may be image-based/scanned. Try pdf_read_ocr instead.");

      const pageTexts = extracted.text.split("\f").filter((s: string) => s.trim());
      const labeled = pageTexts.length <= 1
        ? extracted.text
        : pageTexts.map((t: string, i: number) => `## Page ${i + 1}\n${t.trim()}`).join("\n\n---\n\n");
      return { content: [{ type: "text", text: truncateOutput(labeled) }] };
    },

    renderCall(args, theme) {
      return new Text(theme.fg("toolTitle", theme.bold("pdf_read ")) + theme.fg("accent", basename(args.path)), 0, 0);
    },
    renderResult(result, { isPartial }, theme) {
      if (isPartial) return new Text(theme.fg("warning", "Reading..."), 0, 0);
      if (result.isError) return new Text(theme.fg("error", "Failed"), 0, 0);
      const pages = (result.content?.[0]?.text?.match(/## Page/g) || []).length;
      const label = pages > 0 ? `✓ ${pages} pages read (text)` : "✓ Read";
      return new Text(theme.fg("success", label), 0, 0);
    },
  });

  // ── pdf_read_ocr ───────────────────────────────────────────
  pi.registerTool({
    name: "pdf_read_ocr",
    label: "PDF OCR Reader",
    description:
      "Read image-based/scanned PDF files using vision model OCR. Converts pages to images and extracts text. Use when pdf_read returns no text (scanned documents, image PDFs). Supports up to 20 pages per call. Requires pdftoppm (poppler-utils). Large outputs are truncated to 50KB with full content saved to a temp file.",
    promptSnippet: "Read scanned/image PDF files using vision OCR",
    promptGuidelines: [
      "Use pdf_read_ocr when pdf_read fails or returns empty results, indicating the PDF is image-based or scanned.",
      "This tool is slower and uses vision model tokens. Prefer pdf_read for text-based PDFs.",
      "If output is truncated, a temp file path is provided. Use the read tool with offset/limit to access specific sections.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the PDF file" }),
      pages: Type.Optional(Type.String({ description: 'Page range, e.g. "1-5" or "3". Default: first 20 pages. Max 20 pages per call.' })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.path)) return err(`PDF not found: ${params.path}`);
      if (!hasPdftoppm()) return err("pdftoppm not installed. Install poppler-utils.");

      const tempDir = join(tmpdir(), `pi-pdf-${randomUUID()}`);
      mkdirSync(tempDir, { recursive: true });
      try {
        const pageImages = extractPDFPages(params.path, tempDir);
        if (pageImages.length === 0) return err(`No pages extracted from ${basename(params.path)}. File may be corrupted.`);

        // If the active model can read images, pass page images through directly
        // so the model itself reads them — no external vision API needed.
        const modelSupportsImage = _ctx?.model?.input?.includes("image") ?? false;

        if (modelSupportsImage) {
          onUpdate?.({ content: [{ type: "text", text: `Passing ${pageImages.length} page(s) to model...` }] });
          const content: any[] = [];
          content.push({ type: "text", text: `PDF [${basename(params.path)}] — ${pageImages.length} page(s) rendered at 150 DPI. Read all text from these page images.` });
          if (params.prompt) content.push({ type: "text", text: `Instruction: ${params.prompt}` });
          pageImages.forEach((p, i) => {
            content.push({ type: "text", text: `— Page ${i + 1} —` });
            const block = imageBlock(p);
            if (block) content.push(block);
            else content.push({ type: "text", text: `[failed to read page image]` });
          });
          return { content };
        }

        const initErr = await ensureReady();
        if (initErr) return err(initErr);

        onUpdate?.({ content: [{ type: "text", text: `OCR: converting ${pageImages.length} pages via vision...` }] });

        // Use direct mode (no agent loop) for batch PDF OCR
        const vrList = await Promise.all(pageImages.map(imgPath =>
          callVision(
            "Extract ALL text content from this page verbatim. Preserve headings, body text, captions, formulas.",
            { images: [imgPath], direct: true }
          )
        ));

        const results: string[] = [];
        const errors: string[] = [];
        for (let i = 0; i < vrList.length; i++) {
          if (vrList[i].ok) results.push(`## Page ${i + 1}\n${vrList[i].text}`);
          else { results.push(`## Page ${i + 1}\n[extraction failed]`); errors.push(`Page ${i + 1}: ${vrList[i].error}`); }
        }

        const output = errors.length > 0
          ? results.join("\n\n---\n\n") + `\n\n---\n**Note:** ${errors.length} page(s) had issues: ${errors.join(", ")}`
          : results.join("\n\n---\n\n");
        return { content: [{ type: "text", text: output }] };
      } catch (e: any) { return err(`PDF OCR failed: ${e.message}`); }
      finally { try { rmSync(tempDir, { recursive: true, force: true }); } catch {} }
    },

    renderCall(args, theme) {
      return new Text(theme.fg("toolTitle", theme.bold("pdf_read_ocr ")) + theme.fg("accent", basename(args.path)), 0, 0);
    },
    renderResult(result, { isPartial }, theme) {
      if (isPartial) return new Text(theme.fg("warning", "OCR processing..."), 0, 0);
      if (result.isError) return new Text(theme.fg("error", "Failed"), 0, 0);
      const pages = (result.content?.[0]?.text?.match(/## Page/g) || []).length;
      return new Text(theme.fg("success", `✓ ${pages} pages read (OCR)`), 0, 0);
    },
  });

  // ── image_read ──────────────────────────────────────────────
  pi.registerTool({
    name: "image_read",
    label: "Image Reader",
    description: "Analyze image files (PNG, JPG, WEBP, GIF, BMP). Single-turn: each call is independent.",
    promptSnippet: "Analyze image files",
    promptGuidelines: [
      "prompt omitted → default comprehensive scan (Type/Subject/Content/Text/Layout/Details). prompt provided → replaces it entirely.",
      "Single-turn: each call is independent, no conversation memory. To analyze the same image from different angles, call again with a new prompt.",
      "OCR: explicitly request 'output text only, no translation, no explanation' to reduce filler.",
      "Coordinates: request format [[xmin,ymin,xmax,ymax]] normalized 0-999 (per-mille of image dimensions). Ask for one region at a time, or explicitly request 'every element's coordinates' — do not ask for multiple regions' coordinates at once.",
      "Frontend code: must specify scope (e.g. 'only replicate X, excluding Y') + tech stack, otherwise the model outputs placeholders/ellipsis.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the image file" }),
      prompt: Type.Optional(Type.String({ description: "Specific analysis instruction. If omitted: comprehensive scan. If provided: REPLACES default scan. Examples: 'Transcribe all text', 'Describe top-right corner', 'Bounding box of button as [[xmin,ymin,xmax,ymax]] 0-999'." })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.path)) return err(`Image not found: ${params.path}`);

      const modelSupportsImage = _ctx?.model?.input?.includes("image") ?? false;

      // If the active model can read images directly, pass the image through
      // and let the model see it — same as the built-in read tool does.
      // This skips the vision API round-trip entirely.
      if (modelSupportsImage) {
        const block = imageBlock(resolve(params.path));
        if (!block) return err(`Failed to read image: ${params.path}`);
        const note = params.prompt
          ? `Read image file [${block.mimeType}] — ${params.prompt}`
          : `Read image file [${block.mimeType}]`;
        return { content: [{ type: "text", text: note }, block] };
      }

      // Model can't see images — use vision API (gemma4 agent loop or zai fallback)
      const initErr = await ensureReady();
      if (initErr) return err(initErr);

      onUpdate?.({ content: [{ type: "text", text: "Analyzing image..." }] });

      const vr = await callVision(
        params.prompt || "",
        { images: [resolve(params.path)], onUpdate: (msg) => onUpdate?.({ content: [{ type: "text", text: msg }] }) }
      );
      if (!vr.ok) return err(vr.error);
      return { content: [{ type: "text", text: vr.text }] };
    },

    renderCall(args, theme) {
      return new Text(theme.fg("toolTitle", theme.bold("image_read ")) + theme.fg("accent", basename(args.path)), 0, 0);
    },
    renderResult(result, { isPartial }, theme) {
      if (isPartial) return new Text(theme.fg("warning", "Analyzing image..."), 0, 0);
      if (result.isError) return new Text(theme.fg("error", "Failed"), 0, 0);
      return new Text(theme.fg("success", "✓ Image analyzed"), 0, 0);
    },
  });

  // ── video_describe ──────────────────────────────────────────
  pi.registerTool({
    name: "video_describe",
    label: "Video Describer",
    description: "Analyze a video file (MP4/MOV/M4V, max 8MB). Single-turn.",
    promptSnippet: "Analyze video content",
    promptGuidelines: [
      "query omitted → default description. query provided → replaces it entirely.",
      "Single-turn: each call is independent.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the video file" }),
      query: Type.Optional(Type.String({ description: "What to look for in the video" })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.path)) return err(`Video not found: ${params.path}`);
      const size = statSync(params.path).size;
      if (size > 8 * 1024 * 1024) return err(`Video too large: ${(size / 1024 / 1024).toFixed(1)}MB (limit 8MB).`);
      const initErr = await ensureReady();
      if (initErr) return err(initErr);

      onUpdate?.({ content: [{ type: "text", text: "Analyzing video..." }] });

      // Video always uses zai (gemma4 doesn't support video)
      const vr = await callVision(
        params.query || "Describe what is happening in this video.",
        { video: resolve(params.path) }
      );
      if (!vr.ok) return err(vr.error);
      return { content: [{ type: "text", text: vr.text }] };
    },

    renderCall(args, theme) {
      return new Text(theme.fg("toolTitle", theme.bold("video_describe ")) + theme.fg("accent", basename(args.path)), 0, 0);
    },
    renderResult(result, { isPartial }, theme) {
      if (isPartial) return new Text(theme.fg("warning", "Analyzing video..."), 0, 0);
      if (result.isError) return new Text(theme.fg("error", "Failed"), 0, 0);
      return new Text(theme.fg("success", "✓ Video analyzed"), 0, 0);
    },
  });

  // ── screenshot ──────────────────────────────────────────────
  pi.registerTool({
    name: "screenshot",
    label: "Screenshot",
    description: "Capture a screenshot (full screen or specific window) and analyze it. Windows only.",
    promptSnippet: "Capture and analyze screen",
    promptGuidelines: [
      "window: window title substring (e.g. 'Chrome', 'VS Code') to capture a specific window. Omit for full screen.",
      "prompt omitted → default UI analysis. prompt provided → replaces it entirely.",
      "Coordinates: request format [[xmin,ymin,xmax,ymax]] 0-999. One region at a time.",
      "Windows only.",
    ],
    parameters: Type.Object({
      window: Type.Optional(Type.String({ description: "Window title substring to capture (e.g., 'Chrome', 'VS Code'). Omit for full screen." })),
      prompt: Type.Optional(Type.String({ description: "What to analyze (e.g., 'describe the UI layout', 'check for visual bugs')" })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      const modelSupportsImage = _ctx?.model?.input?.includes("image") ?? false;
      const target = params.window || null;
      onUpdate?.({ content: [{ type: "text", text: target ? `Capturing ${target}...` : "Capturing screen..." }] });

      let tmpFile: string;
      try { tmpFile = takeScreenshot(target || undefined); }
      catch (e: any) {
        if (e.message?.includes("Window not found")) {
          const windows = listWindows();
          const hint = windows.length > 0
            ? `Available windows: ${windows.map(w => `"${w.title}" (${w.name})`).join(", ")}`
            : "No visible windows found.";
          return err(`Window "${target}" not found. ${hint}`);
        }
        return err(`Screenshot failed: ${e.message}`);
      }

      // If the active model can read images, pass the screenshot through directly.
      if (modelSupportsImage) {
        const block = imageBlock(tmpFile);
        try { rmSync(tmpFile, { force: true }); } catch {}
        if (!block) return err(`Failed to read screenshot: ${tmpFile}`);
        const content: any[] = [];
        content.push({ type: "text", text: `Screenshot captured${params.window ? ` (${params.window})` : ""}.` });
        if (params.prompt) content.push({ type: "text", text: `Instruction: ${params.prompt}` });
        content.push(block);
        return { content };
      }

      const initErr = await ensureReady();
      if (initErr) { try { rmSync(tmpFile, { force: true }); } catch {} return err(initErr); }

      onUpdate?.({ content: [{ type: "text", text: "Analyzing UI..." }] });

      const vr = await callVision(
        params.prompt || "Analyze this screenshot for UI development. Describe: 1) Overall layout and structure 2) All visible UI components and states 3) Text content and labels 4) Colors and visual hierarchy 5) Any visual issues or inconsistencies",
        { images: [tmpFile], onUpdate: (msg) => onUpdate?.({ content: [{ type: "text", text: msg }] }) }
      );

      try { rmSync(tmpFile, { force: true }); } catch {}
      if (!vr.ok) return err(vr.error);
      return { content: [{ type: "text", text: vr.text }] };
    },

    renderCall(args, theme) {
      if (args.window) return new Text(theme.fg("toolTitle", theme.bold("screenshot ")) + theme.fg("dim", args.window), 0, 0);
      return new Text(theme.fg("toolTitle", theme.bold("screenshot")), 0, 0);
    },
    renderResult(result, { isPartial }, theme) {
      if (isPartial) {
        const t = result.content?.[0]?.text || "";
        if (t.includes("Analyzing") || t.includes("Cropping") || t.includes("round"))
          return new Text(theme.fg("warning", "Analyzing UI..."), 0, 0);
        return new Text(theme.fg("warning", "Capturing..."), 0, 0);
      }
      if (result.isError) return new Text(theme.fg("error", "Failed"), 0, 0);
      return new Text(theme.fg("success", "✓ Screen analyzed"), 0, 0);
    },
  });

  // ── ui_compare ──────────────────────────────────────────────
  pi.registerTool({
    name: "ui_compare",
    label: "UI Compare",
    description: "Capture a screenshot and compare it against a reference design image. Windows only.",
    promptSnippet: "Compare screen with design reference",
    promptGuidelines: [
      "reference: design image path. Screenshot is the current screen. Both sent together for comparison.",
      "window: window title substring to capture. Omit for full screen.",
      "prompt omitted → default diff analysis. prompt provided → replaces it entirely.",
      "Windows only.",
    ],
    parameters: Type.Object({
      reference: Type.String({ description: "Absolute path to the reference/design image file" }),
      window: Type.Optional(Type.String({ description: "Window title substring to capture. Omit for full screen." })),
      prompt: Type.Optional(Type.String({ description: "What aspects to compare (e.g., 'check layout consistency', 'find all visual differences')" })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.reference)) return err(`Reference image not found: ${params.reference}`);

      const modelSupportsImage = _ctx?.model?.input?.includes("image") ?? false;
      const target = params.window || null;
      onUpdate?.({ content: [{ type: "text", text: target ? `Capturing ${target}...` : "Capturing screen..." }] });

      let tmpFile: string;
      try { tmpFile = takeScreenshot(target || undefined); }
      catch (e: any) {
        if (e.message?.includes("Window not found")) {
          const windows = listWindows();
          const hint = windows.length > 0
            ? `Available windows: ${windows.map(w => `"${w.title}" (${w.name})`).join(", ")}`
            : "No visible windows found.";
          return err(`Window "${target}" not found. ${hint}`);
        }
        return err(`Screenshot failed: ${e.message}`);
      }

      // If the active model can read images, pass reference + screenshot through directly.
      if (modelSupportsImage) {
        const refBlock = imageBlock(resolve(params.reference));
        const shotBlock = imageBlock(tmpFile);
        try { rmSync(tmpFile, { force: true }); } catch {}
        if (!refBlock) return err(`Failed to read reference image: ${params.reference}`);
        if (!shotBlock) return err(`Failed to read screenshot: ${tmpFile}`);
        const content: any[] = [];
        content.push({ type: "text", text: `Compare the reference design against the live screenshot${params.window ? ` (${params.window})` : ""}.` });
        if (params.prompt) content.push({ type: "text", text: `Instruction: ${params.prompt}` });
        content.push({ type: "text", text: "Reference (expected):" });
        content.push(refBlock);
        content.push({ type: "text", text: "Screenshot (actual):" });
        content.push(shotBlock);
        return { content };
      }

      const initErr = await ensureReady();
      if (initErr) { try { rmSync(tmpFile, { force: true }); } catch {} return err(initErr); }

      onUpdate?.({ content: [{ type: "text", text: "Comparing UI..." }] });

      // Multi-image: use direct mode (crop targets first image only, not ideal for comparison)
      const vr = await callVision(
        params.prompt || "Compare the design reference (first image, expected) with the actual screenshot (second image). Identify all visual differences including: layout shifts, color differences, font/size mismatches, missing or extra elements, spacing issues.",
        { images: [resolve(params.reference), tmpFile], direct: true }
      );

      try { rmSync(tmpFile, { force: true }); } catch {}
      if (!vr.ok) return err(vr.error);
      return { content: [{ type: "text", text: vr.text }] };
    },

    renderCall(args, theme) {
      return new Text(theme.fg("toolTitle", theme.bold("ui_compare ")) + theme.fg("accent", basename(args.reference)), 0, 0);
    },
    renderResult(result, { isPartial }, theme) {
      if (isPartial) {
        const t = result.content?.[0]?.text || "";
        if (t.includes("Comparing")) return new Text(theme.fg("warning", "Comparing UI..."), 0, 0);
        return new Text(theme.fg("warning", "Capturing..."), 0, 0);
      }
      if (result.isError) return new Text(theme.fg("error", "Failed"), 0, 0);
      return new Text(theme.fg("success", "✓ UI compared"), 0, 0);
    },
  });
}
