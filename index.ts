import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { ChildProcess, spawn } from "node:child_process";
import { readFileSync, existsSync, statSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve, basename, join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { execSync } from "node:child_process";
import { Text } from "@earendil-works/pi-tui";

// ─── MCP Client ─────────────────────────────────────────────────
let mcpProcess: ChildProcess | null = null;
let requestId = 0;
const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
let initPromise: Promise<void> | null = null;

function ensureMCP(apiKey: string): Promise<void> {
  if (initPromise) return initPromise;
  initPromise = new Promise((resolveInit, rejectInit) => {
    mcpProcess = spawn("npx", ["-y", "@z_ai/mcp-server"], {
      env: { ...process.env, Z_AI_API_KEY: apiKey, Z_AI_MODE: "ZAI" },
      shell: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let buffer = "";
    mcpProcess.stdout!.on("data", (chunk: Buffer) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const msg = JSON.parse(trimmed);
          if (msg.id != null && pending.has(msg.id)) {
            const { resolve, reject } = pending.get(msg.id)!;
            pending.delete(msg.id);
            if (msg.error) reject(new Error(msg.error.message || "MCP error"));
            else resolve(msg.result);
          }
        } catch {}
      }
    });
    mcpProcess.stderr!.on("data", () => {});
    mcpProcess.on("error", (err) => {
      for (const [, p] of pending) p.reject(new Error(`MCP spawn failed: ${err.message}`));
      pending.clear(); initPromise = null; mcpProcess = null;
    });
    mcpProcess.on("exit", (code) => {
      for (const [, p] of pending) p.reject(new Error(`MCP exited unexpectedly (code ${code})`));
      pending.clear(); initPromise = null; mcpProcess = null;
    });
    sendMCP("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "pi-zai-vision-bridge", version: "1.3.0" },
    }).then(() => {
      sendNotification("notifications/initialized", {});
      resolveInit();
    }).catch(rejectInit);
  });
  return initPromise;
}

function sendMCP(method: string, params: any): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!mcpProcess || mcpProcess.killed) return reject(new Error("MCP not running"));
    const id = ++requestId;
    pending.set(id, { resolve, reject });
    mcpProcess.stdin!.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    setTimeout(() => { if (pending.delete(id)) reject(new Error("MCP timeout")); }, 120_000);
  });
}

function sendNotification(method: string, params: any) {
  if (!mcpProcess || mcpProcess.killed) return;
  mcpProcess.stdin!.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
}

type VisionResult = { ok: true; text: string } | { ok: false; error: string };

async function callVisionTool(name: string, args: Record<string, string>): Promise<VisionResult> {
  try {
    const result = await sendMCP("tools/call", { name, arguments: args });
    if (result?.content) {
      const texts = result.content.filter((c: any) => c.type === "text").map((c: any) => c.text);
      if (texts.length > 0) return { ok: true, text: texts.join("\n") };
    }
    return { ok: false, error: `${name}: empty response from vision model` };
  } catch (e: any) {
    const msg = e?.message || "unknown error";
    if (msg.includes("timeout")) return { ok: false, error: `${name}: vision model timed out (120s)` };
    if (msg.includes("not running") || msg.includes("exited")) return { ok: false, error: `${name}: MCP server process not running` };
    return { ok: false, error: `${name}: ${msg}` };
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
  try { execSync("pdftotext -v", { stdio: "ignore" }); return true; } catch { return false; }
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
    if (!key) return "ZAI API key not found. Set Z_AI_API_KEY env or configure auth.json.";
    if (!initialized) {
      try { await ensureMCP(key); initialized = true; }
      catch (e: any) { return `Vision MCP init failed: ${e?.message || "unknown"}`; }
    }
    return null;
  }

  // ── Intercept images for non-vision models ──────────────────
  pi.on("input", async (event, ctx) => {
    if (ctx.model?.input?.includes("image")) return { action: "continue" };
    if (!event.images || event.images.length === 0) return { action: "continue" };

    const initErr = await ensureReady();
    if (initErr) {
      ctx.ui.notify("Vision MCP unavailable", "warning");
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

        const vr = await callVisionTool("analyze_image", {
          image_source: imageSource,
          prompt: "Describe this image in detail, including all visible text, layout, and elements.",
        });
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
    label: "PDF Reader",
    description:
      "Read and extract text/content from PDF files using ZAI Vision MCP. Supports up to 20 pages per call. Requires pdftoppm (poppler-utils).",
    promptSnippet: "Read PDF files using vision model",
    promptGuidelines: [
      "Use pdf_read to read content from PDF files when the user asks about a PDF document.",
      "For text-based PDFs, extraction is fast via pdftotext. For image-based/scanned PDFs, pages are converted to images and processed with a vision model.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the PDF file" }),
      pages: Type.Optional(Type.String({ description: 'Page range, e.g. "1-5" or "3". Default: first 20 pages' })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.path)) return err(`PDF not found: ${params.path}`);

      // Step 1: try direct text extraction first (fast, no MCP needed)
      if (hasPdftotext()) {
        const extracted = tryPdfTextExtract(params.path, params.pages);
        if (extracted) {
          // Split by form-feed to label pages
          const pageTexts = extracted.text.split("\f").filter((s: string) => s.trim());
          const labeled = pageTexts.length <= 1
            ? extracted.text
            : pageTexts.map((t: string, i: number) => `## Page ${i + 1}\n${t.trim()}`).join("\n\n---\n\n");
          return { content: [{ type: "text", text: labeled }] };
        }
      }

      // Step 2: text extraction failed or too sparse — fall back to image-based OCR
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
          callVisionTool("extract_text_from_screenshot", {
            image_source: imgPath,
            prompt: "Extract ALL text content from this page verbatim. Preserve headings, body text, captions, formulas.",
          })
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
      } catch (e: any) { return err(`PDF reading failed: ${e.message}`); }
      finally { try { rmSync(tempDir, { recursive: true, force: true }); } catch {} }
    },

    renderCall(args, theme) {
      return new Text(theme.fg("toolTitle", theme.bold("pdf_read ")) + theme.fg("accent", basename(args.path)), 0, 0);
    },
    renderResult(result, { isPartial }, theme) {
      if (isPartial) return new Text(theme.fg("warning", "Processing..."), 0, 0);
      if (result.isError) return new Text(theme.fg("error", "Failed"), 0, 0);
      const pages = (result.content?.[0]?.text?.match(/## Page/g) || []).length;
      return new Text(theme.fg("success", `✓ ${pages} pages read`), 0, 0);
    },
  });

  // ── image_read ──────────────────────────────────────────────
  pi.registerTool({
    name: "image_read",
    label: "Image Reader",
    description: "Read and analyze image files using ZAI Vision MCP. Use when the current model does not support image input (PNG, JPG, WEBP, GIF, BMP). Returns a detailed text description.",
    promptSnippet: "Read image files using vision model",
    promptGuidelines: [
      "Use image_read when you need to analyze an image file but the current model does not support image input.",
      "When the built-in read tool returns 'model does not support images' for an image file, use image_read instead.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the image file" }),
      prompt: Type.Optional(Type.String({ description: "What to analyze (e.g., 'describe the scene', 'extract all text', 'identify objects')" })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      if (!existsSync(params.path)) return err(`Image not found: ${params.path}`);
      const initErr = await ensureReady();
      if (initErr) return err(initErr);

      onUpdate?.({ content: [{ type: "text", text: "Analyzing image..." }] });

      const vr = await callVisionTool("analyze_image", {
        image_source: resolve(params.path),
        prompt: params.prompt || "Describe this image in detail, including all visible text, layout, colors, objects, and elements.",
      });
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
    description: "Analyze a video file by sending it to ZAI Vision MCP. Supports MP4/MOV/M4V (max 8MB).",
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

      const vr = await callVisionTool("analyze_video", {
        video_source: resolve(params.path),
        prompt: params.query || "Describe what is happening in this video.",
      });
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
    description: "Capture a screenshot and analyze it using ZAI Vision MCP. Supports full screen or a specific window by title substring. Designed for UI development. Screenshot is temporary and deleted after analysis.",
    promptSnippet: "Capture and analyze screen for UI development",
    promptGuidelines: [
      "Use screenshot when you need to see what's currently on the user's screen, especially during UI development.",
      "If the user mentions a specific app or window, pass it as the 'window' parameter.",
      "Use this to understand layout, check component rendering, debug visual issues, or verify UI behavior.",
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

      const vr = await callVisionTool("analyze_image", {
        image_source: tmpFile,
        prompt: params.prompt || "Analyze this screenshot for UI development. Describe: 1) Overall layout and structure 2) All visible UI components and states 3) Text content and labels 4) Colors and visual hierarchy 5) Any visual issues or inconsistencies",
      });

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
    description: "Capture a screenshot and compare it against a reference design image using ZAI Vision MCP. Supports specific window capture. Designed to compare design mockup with actual implementation.",
    promptSnippet: "Compare screen with design reference for UI diff",
    promptGuidelines: [
      "Use ui_compare when you need to compare the current screen with a design mockup or reference image.",
      "ui_compare captures a screenshot and compares it with the reference image for visual diff analysis.",
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

      const vr = await callVisionTool("ui_diff_check", {
        expected_image_source: resolve(params.reference),
        actual_image_source: tmpFile,
        prompt: params.prompt || "Compare the design reference (expected) with the actual screenshot. Identify all visual differences including: layout shifts, color differences, font/size mismatches, missing or extra elements, spacing issues.",
      });

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

  // ── /zai-vision-login ───────────────────────────────────────
  pi.registerCommand("zai-vision-login", {
    description: "Set ZAI API key for vision MCP",
    handler: async (args, ctx) => {
      const t = (args ?? "").trim();
      let key: string;
      if (t) { key = t; } else {
        const input = await ctx.ui.input("ZAI Vision Login - API Key:");
        if (!input?.trim()) return ctx.ui.notify("Cancelled.", "warning");
        key = input.trim();
      }
      process.env.Z_AI_API_KEY = key;
      ctx.ui.notify("✓ API key set.", "success");
    },
  });
}
