---
title: mse_ai_api
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
short_description: Free OpenAI API via DuckDuckGo AI Chat
---

# ⚡ mse_ai_api — Free OpenAI-Compatible API

A production-ready **FastAPI** server powered by **DuckDuckGo AI Chat** — no login, no API key needed.

## 🌐 Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /dashboard` | Web dashboard |
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /v1/responses` | Responses API |
| `GET /v1/models` | List models |
| `GET /docs` | Swagger UI |

## 🤖 Available Models

| Model ID | Name |
|---|---|
| `gpt-5-mini` | GPT-5 mini default |
| `gpt-4o-mini` | GPT-4o mini |
| `gpt-oss-120b` | GPT-OSS 120B |
| `llama-4-scout` | Llama 4 Scout |
| `claude-haiku-4.5` | Claude Haiku 4.5 |
| `mistral-small-3` | Mistral Small 3 |

## API Usage

Base URL: `https://nopoh22-mse-ai-api.hf.space/v1`

Header: `Authorization: Bearer YOUR_API_SECRET_KEY`