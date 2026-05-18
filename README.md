# zai-vision-bridge

Pi Coding Agent extension that bridges [ZAI Vision MCP](https://docs.bigmodel.cn/cn/coding-plan/mcp/vision-mcp-server) (GLM-4.6V) for multimodal understanding.

Provides image analysis, PDF reading, video description, screenshot capture, and UI comparison — all via the official ZAI MCP server, no local vision model needed.

## Install

```bash
pi install https://github.com/inouemoby/pi-zai-vision-bridge.git
```

## Setup

The extension automatically reads your ZAI API key from `~/.pi/agent/auth.json`. If you already have a `zai` provider configured in pi, no extra setup needed.

Otherwise, set the API key manually:

```
/zai-vision-login <your-api-key>
```

## Features

### Auto Image Intercept

When using a **non-vision model** (e.g. glm-5.1) and you paste an image, the extension automatically:

1. Intercepts the image
2. Sends it to ZAI Vision MCP for analysis
3. Replaces the image with a detailed text description

When using a **vision model** (e.g. glm-5v-turbo), images pass through natively — no interception.

### Tools

| Tool | Description |
|------|-------------|
| `image_read` | Analyze image files when the model doesn't support vision |
| `pdf_read` | Read and extract text from PDF files (up to 20 pages, requires [poppler-utils](https://poppler.freedesktop.org/)) |
| `video_describe` | Analyze video content (MP4/MOV/M4V, max 8MB) |
| `screenshot` | Capture screen (full or specific window) and analyze UI layout **\*Windows only\*** |
| `ui_compare` | Capture screen and compare with a design reference image **\*Windows only\*** |

### Commands

| Command | Description |
|---------|-------------|
| `/zai-vision-login` | Set ZAI API key |

### Output Isolation

All tools isolate their internal workings from the pi interface:

- **UI display**: Only shows minimal status (e.g., `Processing...` → `✓ 5 pages read` or `Failed`)
- **AI access**: Full extracted text/results are available to the AI for answering questions
- **Errors**: Descriptive messages for the AI to diagnose (e.g., `PDF not found: /path/to/file`), but no raw MCP protocol data, base64, or stack traces are ever exposed
- **Screenshots**: Temporary files are deleted immediately after analysis

## Platform Notes

- **screenshot** and **ui_compare** use Windows PowerShell for screen capture and are **Windows-only**. The `window` parameter captures a specific window by title substring (e.g., `"Chrome"`, `"VS Code"`).
- **pdf_read** requires [poppler-utils](https://poppler.freedesktop.org/) (`pdftoppm`) to be installed. On Windows, add it to your PATH.
- Other tools (`image_read`, `video_describe`, auto image intercept) work on all platforms.

## Quota

Vision MCP calls share your Coding Plan's **5-hour token pool** — the same pool shown by `5h:xx%` in the footer (via [zai-usage](https://github.com/inouemoby/pi-zai-usage)). No separate call limit for vision.

## Related

- [pi-zai-usage](https://github.com/inouemoby/pi-zai-usage) — ZAI quota monitor
- [pi-ollama-usage](https://github.com/inouemoby/pi-ollama-usage) — Ollama Cloud quota monitor

## License

MIT
