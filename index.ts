import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { ChildProcess, spawn } from "node:child_process";
import { readFileSync, existsSync, statSync, mkdirSync, rmSync } from "node:fs";
import { extname, resolve, basename, join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { execSync } from "node:child_process";

// ─── MCP Client (newline-delimited JSON) ─────────────────────────
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
      // Process complete lines (newline-delimited JSON)
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // keep incomplete last line
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const msg = JSON.parse(trimmed);
          if (msg.id != null && pending.has(msg.id)) {
            const { resolve, reject } = pending.get(msg.id)!;
            pending.delete(msg.id);
            if (msg.error) reject(new Error(msg.error.message || JSON.stringify(msg.error)));
            else resolve(msg.result);
          }
        } catch { /* ignore parse errors */ }
      }
    });

    mcpProcess.stderr!.on("data", () => { /* suppress MCP server logs */ });

    mcpProcess.on("error", (err) => {
      rejectInit(err);
      initPromise = null;
      mcpProcess = null;
    });

    mcpProcess.on("exit", () => {
      initPromise = null;
      mcpProcess = null;
    });

    // Initialize MCP
    sendMCP("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "pi-zai-vision-bridge", version: "1.0.0" },
    }).then(() => {
      // Send initialized notification
      sendNotification("notifications/initialized", {});
      resolveInit();
    }).catch(rejectInit);
  });

  return initPromise;
}

function sendMCP(method: string, params: any): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!mcpProcess || mcpProcess.killed) {
      reject(new Error("MCP server not running"));
      return;
    }
    const id = ++requestId;
    pending.set(id, { resolve, reject });
    const msg = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
    mcpProcess.stdin!.write(msg);
    // Timeout after 120s
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error(`MCP call timeout: ${method}`));
      }
    }, 120_000);
  });
}

function sendNotification(method: string, params: any) {
  if (!mcpProcess || mcpProcess.killed) return;
  const msg = JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n";
  mcpProcess.stdin!.write(msg);
}

async function callTool(name: string, args: Record<string, string>): Promise<string> {
  const result = await sendMCP("tools/call", { name, arguments: args });
  if (result?.content) {
    return result.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text)
      .join("\n");
  }
  return JSON.stringify(result);
}

// ─── Helpers ─────────────────────────────────────────────────────
function imageToBase64(path: string): string {
  const buf = readFileSync(path);
  return buf.toString("base64");
}

function imageMediaType(path: string): string {
  const ext = extname(path).toLowerCase();
  const map: Record<string, string> = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
  };
  return map[ext] || "image/png";
}

function getApiKey(): string {
  if (process.env.Z_AI_API_KEY) return process.env.Z_AI_API_KEY;
  try {
    const authPath = join(homedir(), ".pi", "agent", "auth.json");
    const auth = JSON.parse(readFileSync(authPath, "utf-8"));
    if (auth.zai?.key) return auth.zai.key;
  } catch { /* ignore */ }
  return "";
}

const MAX_PAGES_PDF = 20;

function hasPdftoppm(): boolean {
  try { execSync("pdftoppm -v", { stdio: "ignore" }); return true; } catch { return false; }
}

function extractPDFPages(pdfPath: string, tempDir: string): string[] {
  let pageCount = 1;
  try {
    const info = execSync(`pdfinfo "${pdfPath}"`, { encoding: "utf8" });
    const match = info.match(/Pages:\s+(\d+)/);
    if (match) pageCount = parseInt(match[1], 10);
  } catch { /* default to 1 */ }

  const pages = Math.min(pageCount, MAX_PAGES_PDF);
  const outPrefix = join(tempDir, "page");
  execSync(`pdftoppm -png -r 150 -f 1 -l ${pages} "${pdfPath}" "${outPrefix}"`, { stdio: "ignore" });

  const result: string[] = [];
  for (let i = 1; i <= pages; i++) {
    const p = `${outPrefix}-${i}.png`;
    if (existsSync(p)) result.push(p);
  }
  return result;
}

// ─── Main Extension ─────────────────────────────────────────────
export default function (pi: ExtensionAPI) {
  let initialized = false;

  async function ensureReady(): Promise<string> {
    const key = getApiKey();
    if (!key) throw new Error("ZAI API key not configured. Set Z_AI_API_KEY or run /zai-vision-login.");
    if (!initialized) {
      await ensureMCP(key);
      initialized = true;
    }
    return key;
  }

  // ── Intercept images for non-vision models ──────────────────
  pi.on("input", async (event, ctx) => {
    if (ctx.model?.input?.includes("image")) return { action: "continue" };
    if (!event.images || event.images.length === 0) return { action: "continue" };

    try {
      await ensureReady();
    } catch (err: any) {
      ctx.ui.notify(`Vision MCP: ${err.message}`, "error");
      return { action: "continue" };
    }

    ctx.ui.notify(`Analyzing ${event.images.length} image(s) via ZAI Vision MCP...`, "info");

    const descriptions: string[] = [];
    for (let i = 0; i < event.images.length; i++) {
      try {
        let b64: string;
        const img = event.images[i];
        if (img.source?.data) {
          b64 = img.source.data;
        } else if (typeof img === "string") {
          b64 = img;
        } else {
          descriptions.push(`[Image ${i + 1}]: unable to extract`);
          continue;
        }

        const result = await callTool("analyze_image", {
          image_source: `data:image/png;base64,${b64}`,
          prompt: "Describe this image in complete detail, including all visible text, layout, colors, and elements.",
        });
        descriptions.push(`## Image ${i + 1}\n${result}`);
      } catch (err: any) {
        descriptions.push(`## Image ${i + 1}\n**Failed**: ${err.message}`);
      }
    }

    const prefix = event.text?.trim() || "Analyze these images:";
    const newText = `${prefix}\n\n<image_descriptions>\n${descriptions.join("\n\n---\n\n")}\n</image_descriptions>`;

    ctx.ui.notify(`✓ ${event.images.length} image(s) analyzed`, "success");
    return { action: "transform", text: newText, images: [] };
  });

  // ── pdf_read tool ───────────────────────────────────────────
  pi.registerTool({
    name: "pdf_read",
    label: "PDF Reader",
    description: "Read and extract text/content from PDF files using ZAI Vision MCP. Supports up to 20 pages per call. Requires pdftoppm (poppler-utils).",
    promptSnippet: "Read PDF files using vision model (text extraction from PDF pages)",
    promptGuidelines: [
      "Use pdf_read to read content from PDF files when the user asks about a PDF document.",
      "pdf_read converts PDF pages to images and processes them with a vision model for text extraction.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the PDF file" }),
      pages: Type.Optional(Type.String({ description: 'Page range, e.g. "1-5" or "3". Default: first 20 pages' })),
    }),
    async execute(_id, params, _signal, onUpdate, ctx) {
      const pdfPath = params.path;
      if (!existsSync(pdfPath)) throw new Error(`File not found: ${pdfPath}`);

      if (!hasPdftoppm()) {
        return {
          content: [{ type: "text", text: "pdftoppm is not installed. PDF reading requires poppler-utils.\n\nInstall: https://poppler.freedesktop.org/" }],
          isError: true,
        };
      }

      await ensureReady();

      const tempDir = join(tmpdir(), `pi-pdf-${randomUUID()}`);
      mkdirSync(tempDir, { recursive: true });

      try {
        const pageImages = extractPDFPages(pdfPath, tempDir);
        onUpdate?.({ content: [{ type: "text", text: `Processing ${pageImages.length} page(s) via ZAI Vision MCP...` }] });

        const results: string[] = [];
        for (let i = 0; i < pageImages.length; i++) {
          onUpdate?.({ content: [{ type: "text", text: `Reading page ${i + 1}/${pageImages.length}...` }] });
          const b64 = readFileSync(pageImages[i]).toString("base64");
          const text = await callTool("extract_text_from_screenshot", {
            image_source: `data:image/png;base64,${b64}`,
            prompt: "Extract ALL text content from this page verbatim. Include headings, body text, captions — everything readable.",
          });
          results.push(`## Page ${i + 1}\n${text}`);
        }

        return {
          content: [{ type: "text", text: results.join("\n\n---\n\n") }],
          details: { pages: results.length, fileName: basename(pdfPath) },
        };
      } finally {
        try { rmSync(tempDir, { recursive: true, force: true }); } catch { /* ok */ }
      }
    },
  });

  // ── video_describe tool ─────────────────────────────────────
  pi.registerTool({
    name: "video_describe",
    label: "Video Describer",
    description: "Analyze a video file by extracting keyframes and sending them to ZAI Vision MCP for description. Requires ffmpeg. Supports MP4/MOV/M4V (max 8MB).",
    promptSnippet: "Analyze video content via keyframe extraction and vision model",
    promptGuidelines: [
      "Use video_describe when the user asks about the content of a video file.",
      "video_describe extracts keyframes and describes them with a vision model.",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the video file" }),
      query: Type.Optional(Type.String({ description: "What to look for in the video (e.g., 'describe the scene', 'find timestamps with people')" })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      const videoPath = params.path;
      if (!existsSync(videoPath)) throw new Error(`File not found: ${videoPath}`);

      const stat = statSync(videoPath);
      if (stat.size > 8 * 1024 * 1024) {
        return {
          content: [{ type: "text", text: "Video file exceeds 8MB limit for MCP vision analysis." }],
          isError: true,
        };
      }

      await ensureReady();
      onUpdate?.({ content: [{ type: "text", text: "Analyzing video via ZAI Vision MCP..." }] });

      const query = params.query || "Describe what is happening in this video.";
      const result = await callTool("analyze_video", {
        video_source: resolve(videoPath),
        prompt: query,
      });

      return {
        content: [{ type: "text", text: result }],
        details: { fileName: basename(videoPath) },
      };
    },
  });

  // ── /zai-vision-login ───────────────────────────────────────
  pi.registerCommand("zai-vision-login", {
    description: "Set ZAI API key for vision MCP (saved globally)",
    handler: async (args, ctx) => {
      const t = (args ?? "").trim();
      let key: string;
      if (t) {
        key = t;
      } else {
        const input = await ctx.ui.input("ZAI Vision Login — API Key:");
        if (!input?.trim()) return ctx.ui.notify("Cancelled.", "warning");
        key = input.trim();
      }
      process.env.Z_AI_API_KEY = key;
      ctx.ui.notify("✓ ZAI Vision API key set. MCP server will use it on next call.", "success");
    },
  });
}
