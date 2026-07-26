# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MomentSearch is a visual video search and RAG system. Users upload videos or paste YouTube URLs; workers sample keyframes, dedup them, embed with CLIP, and index them per-user in Qdrant. Questions retrieve the most relevant moments and optionally have a vision LLM write a cited answer from the actual frames.

Live at momentsearch.fly.dev. Apache 2.0.

## Commands

```bash
# Full stack (preferred — runs clip service, seed gate, API, worker)
cp .env.example .env   # fill DATABASE_URL + PREFECT_API_URL/KEY at minimum
docker compose up --build

# Bare processes (each in a separate terminal)
uvicorn src.app:app --reload --port 8000     # API + UI
python -m src.worker                          # ingest worker
uvicorn src.clip_service:app --port 8001      # CLIP service (optional)
python -m src.seed                            # one-shot sample seeder

# Syntax check (no test suite exists)
python -m py_compile src/**/*.py
```

Requires **Python 3.11+**, **FFmpeg**, and **Node** (for yt-dlp YouTube extraction). The Docker image bundles all three.

## Architecture

One Docker image, four entrypoints (the command picks which): API, worker, CLIP service, seed gate. All Python lives under `src/`.

**Write path** (slow, background): Browser → presigned PUT to object storage → `POST /api/videos` registers and returns 202 → dispatcher admits round-robin (WFQ) → Prefect worker runs the pipeline: fetch → sample (ffmpeg) → dedup (pHash) → embed (CLIP) → upsert to Qdrant. YouTube videos also get a transcript branch (captions → bge embeddings → second Qdrant collection).

**Read path** (fast): `POST /api/ask` → CLIP text-embed query → parallel kNN on both Qdrant collections (visual + text) → RRF fusion with time-window grouping and cross-modal boost → confidence gate (abstain before LLM if both branches below threshold) → vision LLM generates cited answer from frames + transcript.

**Design rule**: stateful = managed services (Qdrant, Neon Postgres, Prefect Cloud, object storage), stateless = this repo's code. Nothing on local disk in production.

### Key modules

| Module | Role |
|---|---|
| `src/app.py` | FastAPI app, mounts video + search routers |
| `src/config.py` | Every env knob — single source of truth |
| `src/worker.py` | Prefect worker entrypoint, self-heals on serve() crashes |
| `src/dispatcher.py` | WFQ fair scheduler — round-robin across users |
| `src/clip_service.py` | Warm CLIP model behind HTTP — "embedding is a URL" |
| `src/ingest/pipeline.py` | Prefect flow: fetch → sample → embed → transcript |
| `src/rag/search.py` | 2-branch retrieve → RRF fusion → gate → cited answer |
| `src/rag/vector_store.py` | Multi-tenant Qdrant (visual + text collections) |
| `src/rag/embeddings.py` | CLIP (image+text) + transcript embeddings (bge/OpenAI) |
| `src/storage.py` | Object storage abstraction (aws/gcp/gcp_native/flyio/local) |
| `src/db.py` | Postgres: manifest, status, per-user LLM config, WFQ claims |
| `src/llm.py` | Provider-agnostic vision LLM (OpenAI-compat/NVIDIA/Anthropic) |
| `src/api/videos.py` | Write path: presign, register, status, retry, delete |
| `src/api/search.py` | Read path: /api/ask, /api/llm, config, media, UI |
| `ui/index.html` | Single-file frontend — no build step, no framework |

### Multi-tenancy

Every bucket key, Postgres row, and Qdrant point is `user_id`-tagged. Tenant is `X-User-Id` header (default `"default"`). Qdrant uses one shared collection with a tenant payload index, not collection-per-user.

## Conventions

- **No test suite** — verify with `python -m py_compile src/**/*.py` and manual testing.
- **Single-file UI** — `ui/index.html` has no build step. Keep it that way.
- **Retrieval stays keyless** — CLIP runs locally with no API key. The LLM is only for the final answer; search must work without any API key.
- **Visual-first, multimodal for YouTube** — uploaded files are visual-only (no audio transcription). YouTube adds a transcript branch that fuses with the visual branch.
- Type hints and short docstrings explaining *why*, not *what*.
- `config.py` is the single place for all env-driven configuration.

## Deploy

Three Fly.io process groups from `fly.toml`: `api` (auto-stops when idle), `worker`, `clip`. `release_command` runs the seed gate before each deploy. CD via `.github/workflows/fly-deploy.yml` on push to `dev`.

## Environment

Minimum required: `DATABASE_URL`, `PREFECT_API_URL`, `PREFECT_API_KEY`. Storage defaults to local (`STORAGE_PROVIDER=local`). Qdrant defaults to embedded local (single-process only). LLM key is optional — search works without it, answers degrade to visual-similarity summaries. See `.env.example` for the full reference.
