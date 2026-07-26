"""Ingest worker entrypoint — serves both Prefect flows (video + document).

    python -m src.worker

Registers two deployments in Prefect Cloud:
  - ms-ingest-video/ingest        (videos — YouTube + uploads)
  - ms-ingest-document/ingest-doc (papers + decks)

Both long-poll for scheduled runs — outbound HTTPS only, no ports. Scale
horizontally by running more replicas of this process.

Sample seeding is NOT done here — it's a one-shot startup gate (seed.py /
src/seeding.py) that the whole stack waits on, so the app never serves a
half-indexed corpus. This worker only handles user uploads + YouTube adds +
document ingestion.

Embedding goes to the warm CLIP service when CLIP_SERVICE_URL is set
(docker-compose default); unset, each run loads the model in-process.
"""
import os
import time

from prefect import serve
from prefect.flows import EntrypointType

from .db import init_schema
from .ingest.doc_pipeline import ingest_document
from .ingest.pipeline import ingest_video


def main():
    init_schema()
    from .rag import vector_store
    vector_store.ensure_collection()
    from . import dispatcher
    dispatcher.start_in_background()
    limit = int(os.getenv("WORKER_CONCURRENCY", "2"))
    while True:
        try:
            print(f"[worker] serving video + document deployments (concurrency {limit})")
            _ep = EntrypointType.MODULE_PATH
            serve(
                ingest_video.to_deployment(name="ingest", entrypoint_type=_ep),
                ingest_document.to_deployment(name="ingest-doc", entrypoint_type=_ep),
                limit=limit,
            )
            break
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[worker] serve crashed: {type(exc).__name__}: {exc} — retrying in 15s")
            time.sleep(15)


if __name__ == "__main__":
    main()
