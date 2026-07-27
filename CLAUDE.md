# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MomentSearch is a visual video search and RAG system. Users upload videos, paste YouTube URLs, or upload documents (PDF papers, PPTX slide decks); workers sample keyframes (videos) or parse text chunks (documents), embed them with CLIP / bge, and index them per-user in Qdrant. Questions retrieve the most relevant moments — across videos *and* documents — and optionally have a vision LLM write a cited answer from the actual frames. Document citations carry page/slide numbers instead of timestamps.

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

# Eval suite (requires running API at localhost:8000)
python eval/run_evals.py

# Syntax check
python -m py_compile src/**/*.py
```

Requires **Python 3.11+**, **FFmpeg**, and **Node** (for yt-dlp YouTube extraction). The Docker image bundles all three.

## Architecture

One Docker image, four entrypoints (the command picks which): API, worker, CLIP service, seed gate. All Python lives under `src/`.

**Video write path** (slow, background): Browser → presigned PUT to object storage → `POST /api/videos` registers and returns 202 → dispatcher admits round-robin (WFQ) → Prefect worker runs the pipeline: fetch → sample (ffmpeg) → dedup (pHash) → embed (CLIP) → upsert to Qdrant. YouTube videos also get a transcript branch (captions → bge embeddings → second Qdrant collection). Prefect scheduling runs in a background thread so `register` stays under ~250ms.

**Document write path** (same pattern): `POST /api/documents/presign` → PUT to storage → `POST /api/documents` registers (auto-detects `kind` from extension: `.pdf` → paper, `.pptx` → deck) → worker parses into page/slide-aware text chunks → bge/OpenAI text embeddings → upsert to `moments_text` with `kind`, `page`/`slide`, `doc_id` payload. Citations carry `file_url` for in-browser viewing (PDF `#page=N`) or download (PPTX).

**Read path** (fast): `POST /api/ask` → CLIP text-embed query → parallel kNN on both Qdrant collections (visual + text) → RRF fusion with time-window grouping and cross-modal boost → confidence gate (abstain before LLM if both branches below threshold) → vision LLM generates cited answer from frames + transcript. Results are cross-source: video citations show timestamps + thumbnails, document citations show page/slide numbers + file links, all ranked together by fused RRF score.

**Design rule**: stateful = managed services (Qdrant, Neon Postgres, Prefect Cloud, object storage), stateless = this repo's code. Nothing on local disk in production.

### Key modules

| Module | Role |
|---|---|
| `src/app.py` | FastAPI app, mounts video + document + search routers |
| `src/config.py` | Every env knob — single source of truth |
| `src/worker.py` | Prefect worker entrypoint, self-heals on serve() crashes |
| `src/dispatcher.py` | WFQ fair scheduler — round-robin across users |
| `src/clip_service.py` | Warm CLIP model behind HTTP — "embedding is a URL" |
| `src/jobs.py` | Prefect Cloud trigger — schedules runs from the API side |
| `src/ingest/pipeline.py` | Prefect flow: fetch → sample → embed → transcript |
| `src/ingest/doc_pipeline.py` | Prefect flow for documents: fetch → parse → embed → upsert |
| `src/ingest/paper.py` | PDF paper parser (PyMuPDF, ~500-word page-aware chunks) |
| `src/ingest/deck.py` | Slide deck parser (PPTX slides or PDF pages, one chunk each) |
| `src/rag/search.py` | 2-branch retrieve → RRF fusion → gate → cross-source cited answer |
| `src/rag/vector_store.py` | Multi-tenant Qdrant (visual + text collections) |
| `src/rag/embeddings.py` | CLIP (image+text) + transcript embeddings (bge/OpenAI) |
| `src/storage.py` | Object storage abstraction (aws/gcp/gcp_native/flyio/local) |
| `src/db.py` | Postgres: manifest, status, per-user LLM config, WFQ claims |
| `src/llm.py` | Provider-agnostic vision LLM (OpenAI-compat/NVIDIA/Anthropic) |
| `src/api/videos.py` | Video write path: presign, register, status, retry, delete |
| `src/api/documents.py` | Document write path: presign, register, status, file serving |
| `src/api/admin.py` | Unified /admin/sources — cross-type listing, get, delete |
| `src/api/search.py` | Read path: /api/ask, /api/llm, config, media, UI |
| `ui/index.html` | Single-file frontend — no build step, no framework |

### Multi-tenancy

Every bucket key, Postgres row, and Qdrant point is `user_id`-tagged. Tenant is `X-User-Id` header (default `"default"`). Qdrant uses one shared collection with a tenant payload index, not collection-per-user.

## Conventions

- **Eval suite** — `python eval/run_evals.py` (31 evals: API contracts, ingest pipeline, search functional, performance KPIs). Also `python -m py_compile src/**/*.py` for syntax checks.
- **Single-file UI** — `ui/index.html` has no build step. Keep it that way.
- **Retrieval stays keyless** — CLIP runs locally with no API key. The LLM is only for the final answer; search must work without any API key.
- **Visual-first, multimodal for YouTube** — uploaded video files are visual-only (no audio transcription). YouTube adds a transcript branch that fuses with the visual branch. Documents (PDF/PPTX) are text-only and rank alongside videos via the same RRF fusion.
- Type hints and short docstrings explaining *why*, not *what*.
- `config.py` is the single place for all env-driven configuration.

## Deploy

Three Fly.io process groups from `fly.toml`: `api` (auto-stops when idle), `worker`, `clip`. `release_command` runs the seed gate before each deploy. CD via `.github/workflows/fly-deploy.yml` on push to `dev`.

## Environment

Minimum required: `DATABASE_URL`, `PREFECT_API_URL`, `PREFECT_API_KEY`. Storage defaults to local (`STORAGE_PROVIDER=local`). Qdrant defaults to embedded local (single-process only). LLM key is optional — search works without it, answers degrade to visual-similarity summaries. Document uploads accept PDF and PPTX (`ALLOWED_DOC_TYPES`), capped at `MAX_UPLOAD_MB` (default 2048). See `.env.example` for the full reference.
