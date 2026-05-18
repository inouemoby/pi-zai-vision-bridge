# zai-vision-bridge

Pi Coding Agent extension that bridges [ZAI Vision MCP](https://docs.bigmodel.cn/cn/coding-plan/mcp/vision-mcp-server) (GLM-4.6V) for multimodal understanding.

Provides image analysis, PDF reading, and video description via the official ZAI MCP server — no local vision model needed.

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
| `pdf_read` | Read and extract text from PDF files |
| `video_describe` | Analyze video content (MP4/MOV/M4V, max 8MB) |

### Commands

| Command | Description |
|---------|-------------|
| `/zai-vision-login` | Set ZAI API key |

## Quota

Vision MCP calls share your Coding Plan's **5-hour token pool** — the same pool shown by `5h:xx%` in the footer (via [zai-usage](https://github.com/inouemoby/pi-zai-usage)). No separate call limit for vision.

## Related

- [pi-zai-usage](https://github.com/inouemoby/pi-zai-usage) — ZAI quota monitor
- [pi-ollama-usage](https://github.com/inouemoby/pi-ollama-usage) — Ollama Cloud quota monitor
