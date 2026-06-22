import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { truncateHead, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, formatSize } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFileSync, existsSync, statSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve, basename, join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { execSync } from "node:child_process";
import { Text } from "@earendil-works/pi-tui";

// ─── GLM-4.1V-Thinking-Flash API Client ────────────────────────
// This model is trained on SINGLE-TURN dialogue only. No multi-turn context.
// Every call is independent: image + prompt → single response.
// For "follow-up" questions, the caller re-sends the image with a new prompt.
const ZAI_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions";
const ZAI_VISION_MODEL = "glm-4.1v-thinking-flash";

// Default prompt used when no specific question is given.
// Used ONLY when params.prompt is empty/omitted.
// Replaced entirely by params.prompt when one is provided.
const DEFAULT_SCAN_PROMPT =
  "Describe this image in detail. Cover what you see: subjects, objects, text (transcribe verbatim), colors, layout, and any notable details. Be thorough.";

// Truncate large output, save full content to temp file if truncated.
function truncateOutput(raw: string): string {
  const t = truncateHead(raw, { maxLines: DEFAULT_MAX_LINES, maxBytes: DEFAULT_MAX_BYTES });
  if (!t.truncated) return raw;
  const tmpFile = join(tmpdir(), `pi-pdf-${randomUUID()}.txt`);
  try { writeFileSync(tmpFile, raw, "utf8"); } catch {}
  return t.content + `\n\n[Output truncated: ${t.outputLines} of ${t.totalLines} lines (${formatSize(t.outputBytes)} of ${formatSize(t.totalBytes)}). Full content saved to: ${tmpFile}]`;
}

// Read file and return base64 string
function fileToBase64(filePath: string): string {
  const buf = readFileSync(filePath);
  return buf.toString("base64");
}

// Detect MIME type from file extension
function mimeFromPath(filePath: string): string {
  const ext = filePath.toLowerCase().split(".").pop() || "";
  const map: Record<string, string> = {
    png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
    gif: "image/gif", webp: "image/webp", bmp: "image/bmp",
  };
  return map[ext] || "image/png";
}

// Strip thinking-model artifacts from response text.
// With thinking enabled, the final answer is in `content` (may be wrapped in
// <|begin_of_box|>...<|end_of_box|>), reasoning is in separate `reasoning_content`.
function cleanResponse(text: string): string {
  return text
    .replace(/<\|begin_of_box\|>|<\|end_of_box\|>/g, "")
    .trim();
}

type VisionResult = { ok: true; text: string } | { ok: false; error: string };

/**
 * Call GLM-4.1V vision API — single-turn only.
 *
 * - prompt empty → uses DEFAULT_SCAN_PROMPT (comprehensive description)
 * - prompt provided → REPLACES default prompt entirely
 *
 * For follow-up questions on the same image, call again with the same image path
 * and a new specific prompt (e.g., "read the text in the top-right corner").
 */
async function callVision(
  prompt: string,
  opts: { images?: string[]; video?: string } = {}
): Promise<VisionResult> {
  const apiKey = getApiKey();
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
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: ZAI_VISION_MODEL,
        messages: [{ role: "user", content }],
        thinking: { type: "enabled" },
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      return { ok: false, error: `API ${resp.status}: ${errText.slice(0, 200)}` };
    }

    const data = await resp.json() as any;
    const rawText = data.choices?.[0]?.message?.content;
    if (!rawText) return { ok: false, error: "Empty response from vision model" };
    return { ok: true, text: cleanResponse(rawText) };
  } catch (e: any) {
    return { ok: false, error: e?.message || "Vision API call failed" };
  }
}

// ─── Error helper: return isError result instead of throwing ────
function err(msg: string) {
  return { content: [{ type: "text" as const, text: msg }], isError: true as const };
}

// ─── Helpers ─────────────────────────────────────────────────────
function getApiKey(): string {
  if (process.env.Z_AI_API_KEY) return process.env.Z_AI_API_KEY;
  try {
    const auth = JSON.parse(readFileSync(join(homedir(), ".pi", "agent", "auth.json"), "utf-8"));
    if (auth.zai?.key) return auth.zai.key;
  } catch {}
  return "";
}

function hasPdftoppm(): boolean {
  try { execSync("pdftoppm -v", { stdio: "ignore" }); return true; } catch { return false; }
}

function hasPdftotext(): boolean {
  try { const r = execSync("pdftotext -v", { encoding: "utf8", stdio: ["pipe","pipe","pipe"] }); return true; } catch (e: any) { return (e.stdout || "").includes("pdftotext") || (e.stderr || "").includes("pdftotext"); }
}

// Try extracting text directly. Returns null if text too sparse (image-based PDF).
function tryPdfTextExtract(pdfPath: string, pageRange?: string): { text: string; pages: number } | null {
  const tmpFile = join(tmpdir(), `pi-pdf-text-${randomUUID()}.txt`);
  try {
    let args = "";
    if (pageRange) {
      // pdftotext supports -f first -l last page
      const parts = pageRange.split("-").map(Number);
      const first = parts[0] || 1;
      const last = parts[1] || parts[0] || 999;
      args = ` -f ${first} -l ${last}`;
    }
    execSync(`pdftotext${args} -layout "${pdfPath}" "${tmpFile}"`, { stdio: "ignore", timeout: 30_000 });
    if (!existsSync(tmpFile)) return null;
    const text = readFileSync(tmpFile, "utf8").trim();
    if (!text) return null;
    // Heuristic: if less than 50 non-whitespace chars per expected page, treat as image-based
    const lines = text.split("\n").filter(l => l.trim().length > 0);
    const totalChars = lines.reduce((sum, l) => sum + l.replace(/\s/g, "").length, 0);
    const pageCount = Math.max(1, lines.filter(l => /^\f/.test(l)).length + 1); // form feeds separate pages
    const charsPerPage = totalChars / pageCount;
    if (charsPerPage < 30) return null; // too sparse, probably image-based
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

// ─── Main Extension ─────────────────────────────────────────────
export default function (pi: ExtensionAPI) {
  let initialized = false;

  async function ensureReady(): Promise<string | null> {
    const key = getApiKey();
    if (!key) return "ZAI API key not found. Configure auth.json (zai key).";
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
      "Read image-based/scanned PDF files using vision model OCR. Converts pages to images and extracts text via GLM-4.1V-Thinking-Flash. Use when pdf_read returns no text (scanned documents, image PDFs). Supports up to 20 pages per call. Requires pdftoppm (poppler-utils). Large outputs are truncated to 50KB with full content saved to a temp file.",
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
      const initErr = await ensureReady();
      if (initErr) return err(initErr);

      const tempDir = join(tmpdir(), `pi-pdf-${randomUUID()}`);
      mkdirSync(tempDir, { recursive: true });
      try {
        const pageImages = extractPDFPages(params.path, tempDir);
        if (pageImages.length === 0) return err(`No pages extracted from ${basename(params.path)}. File may be corrupted.`);

        onUpdate?.({ content: [{ type: "text", text: `OCR: converting ${pageImages.length} pages via vision...` }] });

        const vrList = await Promise.all(pageImages.map(imgPath =>
          callVision(
            "Extract ALL text content from this page verbatim. Preserve headings, body text, captions, formulas.",
            { images: [imgPath] }
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
    description:
      "Read and analyze image files using GLM-4.1V-Thinking-Flash vision model. Use when the current model does not support image input (PNG, JPG, WEBP, GIF, BMP). Single-turn: each call is independent.",
    promptSnippet: "Read image files using vision model",
    promptGuidelines: [
      "Use image_read when you need to analyze an image file but the current model does not support image input.",
      "When the built-in read tool returns 'model does not support images' for an image file, use image_read instead.",
      "FOLLOW-UP STRATEGY: This model is single-turn. If the first description lacks detail you need, call image_read AGAIN with the SAME image path and a NEW prompt targeting the specific aspect (e.g., 'read all text in the image', 'describe the object in the top-left corner').",
      "If you don't know what's in the image, omit the prompt — the tool will do a comprehensive scan. Then follow up with targeted prompts.",
      "If you know exactly what you need (e.g., 'extract all text', 'count objects'), provide it as the prompt — it replaces the default comprehensive scan.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the image file" }),
      prompt: Type.Optional(Type.String({ description: "What to analyze. If omitted, a comprehensive scan is performed. If provided, REPLACES the default scan — use for targeted questions like 'read all text', 'describe the top-right corner'." })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.path)) return err(`Image not found: ${params.path}`);
      const initErr = await ensureReady();
      if (initErr) return err(initErr);

      onUpdate?.({ content: [{ type: "text", text: "Analyzing image..." }] });

      const vr = await callVision(
        params.prompt || "",
        { images: [resolve(params.path)] }
      );
      if (!vr.ok) return err(vr.error);
      return { content: [{ type: "text", text: vr.text }] };
    },

    renderCall(args, theme) {
      if (args.prompt) return new Text(theme.fg("toolTitle", theme.bold("image_read ")) + theme.fg("accent", basename(args.path)), 0, 0);
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
    description: "Analyze a video file by sending it to GLM-4.1V-Thinking-Flash. Supports MP4/MOV/M4V (max 8MB).",
    promptSnippet: "Analyze video content via vision model",
    promptGuidelines: [
      "Use video_describe when the user asks about the content of a video file.",
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
    description: "Capture a screenshot and analyze it using GLM-4.1V-Thinking-Flash. Supports full screen or a specific window by title substring. Designed for UI development. Screenshot is temporary and deleted after analysis. Windows only (uses PowerShell for capture).",
    promptSnippet: "Capture and analyze screen for UI development",
    promptGuidelines: [
      "Use screenshot when you need to see what's currently on the user's screen, especially during UI development.",
      "If the user mentions a specific app or window, pass it as the 'window' parameter.",
      "Use this to understand layout, check component rendering, debug visual issues, or verify UI behavior.",
      "Windows only — uses PowerShell for screen capture.",
    ],
    parameters: Type.Object({
      window: Type.Optional(Type.String({ description: "Window title substring to capture (e.g., 'Chrome', 'VS Code'). Omit for full screen." })),
      prompt: Type.Optional(Type.String({ description: "What to analyze (e.g., 'describe the UI layout', 'check for visual bugs')" })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      const initErr = await ensureReady();
      if (initErr) return err(initErr);

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

      onUpdate?.({ content: [{ type: "text", text: "Analyzing UI..." }] });

      const vr = await callVision(
        params.prompt || "Analyze this screenshot for UI development. Describe: 1) Overall layout and structure 2) All visible UI components and states 3) Text content and labels 4) Colors and visual hierarchy 5) Any visual issues or inconsistencies",
        { images: [tmpFile] }
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
        if (t.includes("Analyzing")) return new Text(theme.fg("warning", "Analyzing UI..."), 0, 0);
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
    description: "Capture a screenshot and compare it against a reference design image using GLM-4.1V-Thinking-Flash. Supports specific window capture. Designed to compare design mockup with actual implementation. Windows only.",
    promptSnippet: "Compare screen with design reference for UI diff",
    promptGuidelines: [
      "Use ui_compare when you need to compare the current screen with a design mockup or reference image.",
      "ui_compare captures a screenshot and compares it with the reference image for visual diff analysis.",
      "Windows only — uses PowerShell for screen capture.",
    ],
    parameters: Type.Object({
      reference: Type.String({ description: "Absolute path to the reference/design image file" }),
      window: Type.Optional(Type.String({ description: "Window title substring to capture. Omit for full screen." })),
      prompt: Type.Optional(Type.String({ description: "What aspects to compare (e.g., 'check layout consistency', 'find all visual differences')" })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.reference)) return err(`Reference image not found: ${params.reference}`);
      const initErr = await ensureReady();
      if (initErr) return err(initErr);

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

      onUpdate?.({ content: [{ type: "text", text: "Comparing UI..." }] });

      const vr = await callVision(
        params.prompt || "Compare the design reference (first image, expected) with the actual screenshot (second image). Identify all visual differences including: layout shifts, color differences, font/size mismatches, missing or extra elements, spacing issues.",
        { images: [resolve(params.reference), tmpFile] }
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
