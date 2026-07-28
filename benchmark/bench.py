#!/usr/bin/env python3
"""MomentSearch benchmark — latency, throughput, and resilience testing.

Usage:
    python benchmark/bench.py                     # default: latency + throughput
    python benchmark/bench.py --resilience        # adds resilience tests
    python benchmark/bench.py --base-url http://host:8000 --concurrency 20

Modes:
  default     — search latency percentiles (p50/p95/p99), accept latency,
                throughput (queries/sec) under concurrency
  --resilience — search-during-ingest stability, error rate under burst,
                 concurrent mixed-type operations, recovery after failure
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
BENCH_USER = "bench_runner"
CONCURRENCY = 10
SEARCH_ROUNDS = 50
BURST_SIZE = 30

QUERIES = [
    "What is retrieval augmented generation?",
    "How does vector similarity search work?",
    "Explain the attention mechanism",
    "What are transformer architectures?",
    "How do embeddings represent text?",
    "What is reciprocal rank fusion?",
    "Explain confidence gating in RAG",
    "How does CLIP encode images?",
    "What is perceptual hashing?",
    "Describe the difference between HNSW and brute-force search",
]


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def api(method, path, user=BENCH_USER, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["X-User-Id"] = user
    return requests.request(method, BASE_URL + path, headers=headers,
                            timeout=30, **kwargs)


def make_test_pdf(n_pages=3):
    import fitz
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        tw = fitz.TextWriter(page.rect)
        tw.append((72, 100), f"Benchmark test page {i+1} about neural networks and deep learning.")
        tw.write_text(page)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def upload_doc(file_bytes, filename, content_type, title, user=BENCH_USER):
    r = api("POST", "/api/documents/presign", user=user,
            json={"filename": filename, "content_type": content_type,
                  "size": len(file_bytes)})
    if r.status_code != 200:
        return None, f"presign failed: {r.status_code}"
    p = r.json()
    url = p["url"] if p["url"].startswith("http") else BASE_URL + p["url"]
    put_headers = {**p["headers"], "X-User-Id": user}
    r2 = requests.put(url, headers=put_headers, data=file_bytes, timeout=30)
    if r2.status_code not in (200, 201):
        return None, f"upload failed: {r2.status_code}"
    r3 = api("POST", "/api/documents", user=user,
             json={"doc_id": p["doc_id"], "key": p["key"], "title": title})
    if r3.status_code != 202:
        return None, f"register failed: {r3.status_code}"
    return p["doc_id"], "ok"


def cleanup():
    try:
        docs = api("GET", "/api/documents").json().get("documents", [])
        for d in docs:
            api("DELETE", f"/api/documents/{d['id']}")
    except Exception:
        pass
    try:
        vids = api("GET", "/api/videos").json().get("videos", [])
        for v in vids:
            try:
                api("DELETE", f"/api/videos/{v['id']}")
            except Exception:
                pass
    except Exception:
        pass


# ── Percentile helper ────────────────────────────────────────────────────────

def percentile(data, p):
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(data_sorted) else f
    d = k - f
    return data_sorted[f] + d * (data_sorted[c] - data_sorted[f])


# ── Benchmark: Search Latency ────────────────────────────────────────────────

def bench_search_latency(concurrency):
    print("\n  SEARCH LATENCY")
    print("  " + "-" * 60)

    latencies = []
    errors = 0

    def do_search(q):
        t0 = time.time()
        try:
            r = api("POST", "/api/ask", user="default",
                    json={"question": q})
            elapsed = (time.time() - t0) * 1000
            if r.status_code == 200:
                return elapsed, None
            return elapsed, f"status={r.status_code}"
        except Exception as e:
            return (time.time() - t0) * 1000, str(e)

    queries = [QUERIES[i % len(QUERIES)] for i in range(SEARCH_ROUNDS)]

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(do_search, q) for q in queries]
        for f in as_completed(futs):
            ms, err = f.result()
            if err:
                errors += 1
            else:
                latencies.append(ms)

    if latencies:
        print(f"  Requests:    {len(latencies)} ok, {errors} errors")
        print(f"  p50:         {percentile(latencies, 50):.0f} ms")
        print(f"  p95:         {percentile(latencies, 95):.0f} ms")
        print(f"  p99:         {percentile(latencies, 99):.0f} ms")
        print(f"  min/max:     {min(latencies):.0f} / {max(latencies):.0f} ms")
        qps = len(latencies) / (sum(latencies) / 1000 / concurrency) if latencies else 0
        print(f"  Throughput:  {qps:.1f} queries/sec (concurrency={concurrency})")
    else:
        print(f"  All {errors} requests failed.")

    return {"latencies": latencies, "errors": errors}


# ── Benchmark: Accept Latency ────────────────────────────────────────────────

def bench_accept_latency():
    print("\n  ACCEPT LATENCY (document register)")
    print("  " + "-" * 60)

    pdf_bytes = make_test_pdf(1)
    latencies = []
    doc_ids = []

    # Pre-stage uploads
    staged = []
    for i in range(10):
        r = api("POST", "/api/documents/presign",
                json={"filename": f"bench_{i}.pdf",
                      "content_type": "application/pdf",
                      "size": len(pdf_bytes)})
        if r.status_code != 200:
            continue
        p = r.json()
        url = p["url"] if p["url"].startswith("http") else BASE_URL + p["url"]
        r2 = requests.put(url, headers={**p["headers"], "X-User-Id": BENCH_USER},
                          data=pdf_bytes, timeout=30)
        if r2.status_code in (200, 201):
            staged.append(p)

    # Burst register calls
    for p in staged:
        t0 = time.time()
        r = api("POST", "/api/documents",
                json={"doc_id": p["doc_id"], "key": p["key"],
                      "title": "Bench Accept Test"})
        ms = (time.time() - t0) * 1000
        if r.status_code == 202:
            latencies.append(ms)
            doc_ids.append(p["doc_id"])

    if latencies:
        p95 = percentile(latencies, 95)
        print(f"  Samples:     {len(latencies)}")
        print(f"  p50:         {percentile(latencies, 50):.0f} ms")
        print(f"  p95:         {p95:.0f} ms")
        print(f"  Target:      < 500 ms  {'PASS' if p95 < 500 else 'FAIL'}")
    else:
        print("  No successful registrations.")

    for did in doc_ids:
        api("DELETE", f"/api/documents/{did}")

    return {"latencies": latencies}


# ── Benchmark: Throughput ────────────────────────────────────────────────────

def bench_throughput(concurrency):
    print("\n  SEARCH THROUGHPUT")
    print("  " + "-" * 60)

    queries = [QUERIES[i % len(QUERIES)] for i in range(SEARCH_ROUNDS)]
    t_start = time.time()
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        def do_q(q):
            r = api("POST", "/api/ask", user="default", json={"question": q})
            return r.status_code == 200

        futs = [ex.submit(do_q, q) for q in queries]
        for f in as_completed(futs):
            if f.result():
                completed += 1
            else:
                errors += 1

    wall = time.time() - t_start
    qps = completed / wall if wall > 0 else 0
    print(f"  Completed:   {completed}/{len(queries)} in {wall:.1f}s")
    print(f"  Errors:      {errors}")
    print(f"  QPS:         {qps:.1f}")

    return {"qps": qps, "wall_s": wall, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════════════
# RESILIENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def resilience_search_during_ingest():
    """Search must remain responsive while ingest is running."""
    print("\n  RESILIENCE: search during ingest")
    print("  " + "-" * 60)

    pdf_bytes = make_test_pdf(5)
    doc_ids = []

    # Start several ingests (don't wait for completion)
    for i in range(3):
        did, msg = upload_doc(pdf_bytes, f"resilience_{i}.pdf",
                              "application/pdf", f"Resilience Doc {i}")
        if did:
            doc_ids.append(did)

    # Immediately hammer search
    latencies = []
    errors = 0
    for i in range(20):
        t0 = time.time()
        try:
            r = api("POST", "/api/ask", user="default",
                    json={"question": QUERIES[i % len(QUERIES)]})
            ms = (time.time() - t0) * 1000
            if r.status_code == 200:
                latencies.append(ms)
            else:
                errors += 1
        except Exception:
            errors += 1

    if latencies:
        p95 = percentile(latencies, 95)
        print(f"  Searches:    {len(latencies)} ok, {errors} errors")
        print(f"  p95:         {p95:.0f} ms (during active ingest)")
        print(f"  Result:      {'PASS' if errors == 0 and p95 < 5000 else 'FAIL'}")
    else:
        print(f"  All searches failed ({errors} errors)  FAIL")

    # Cleanup
    for did in doc_ids:
        api("DELETE", f"/api/documents/{did}")

    return {"latencies": latencies, "errors": errors}


def resilience_burst_register():
    """Burst concurrent register calls — API must not crash or lose data."""
    print("\n  RESILIENCE: burst register")
    print("  " + "-" * 60)

    pdf_bytes = make_test_pdf(1)
    n = 10
    staged = []

    for i in range(n):
        r = api("POST", "/api/documents/presign",
                json={"filename": f"burst_{i}.pdf",
                      "content_type": "application/pdf",
                      "size": len(pdf_bytes)})
        if r.status_code != 200:
            continue
        p = r.json()
        url = p["url"] if p["url"].startswith("http") else BASE_URL + p["url"]
        requests.put(url, headers={**p["headers"], "X-User-Id": BENCH_USER},
                     data=pdf_bytes, timeout=30)
        staged.append(p)

    # Burst all register calls concurrently
    results = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        def do_register(p):
            r = api("POST", "/api/documents",
                    json={"doc_id": p["doc_id"], "key": p["key"],
                          "title": "Burst Test"})
            return r.status_code

        futs = {ex.submit(do_register, p): p for p in staged}
        for f in as_completed(futs):
            results.append(f.result())

    accepted = sum(1 for r in results if r == 202)
    print(f"  Staged:      {len(staged)}")
    print(f"  Accepted:    {accepted}/{len(results)} returned 202")
    print(f"  Result:      {'PASS' if accepted == len(staged) else 'FAIL'}")

    # Verify all appear in listing
    docs = api("GET", "/api/documents").json().get("documents", [])
    doc_ids = {d["id"] for d in docs}
    found = sum(1 for p in staged if p["doc_id"] in doc_ids)
    print(f"  In listing:  {found}/{len(staged)} (no data loss)")

    for p in staged:
        api("DELETE", f"/api/documents/{p['doc_id']}")

    return {"staged": len(staged), "accepted": accepted, "found": found}


def resilience_mixed_operations():
    """Concurrent reads and writes across different source types."""
    print("\n  RESILIENCE: mixed concurrent operations")
    print("  " + "-" * 60)

    pdf_bytes = make_test_pdf(1)
    errors = 0
    successes = 0

    def search_op():
        r = api("POST", "/api/ask", user="default",
                json={"question": "What is deep learning?"})
        return "search", r.status_code == 200

    def list_op():
        r = api("GET", "/api/videos", user="default")
        return "list_videos", r.status_code == 200

    def list_docs_op():
        r = api("GET", "/api/documents", user="default")
        return "list_docs", r.status_code == 200

    def admin_op():
        r = api("GET", "/admin/sources", user="default")
        return "admin_sources", r.status_code == 200

    def health_op():
        r = requests.get(BASE_URL + "/api/health", timeout=10)
        return "health", r.status_code == 200

    ops = [search_op, list_op, list_docs_op, admin_op, health_op] * 4

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(op) for op in ops]
        for f in as_completed(futs):
            name, ok = f.result()
            if ok:
                successes += 1
            else:
                errors += 1

    total = len(ops)
    print(f"  Operations:  {total} ({successes} ok, {errors} errors)")
    print(f"  Error rate:  {errors/total*100:.1f}%")
    print(f"  Result:      {'PASS' if errors == 0 else 'FAIL'}")

    return {"total": total, "successes": successes, "errors": errors}


def resilience_recovery():
    """System recovers after a bad request — no cascading failures."""
    print("\n  RESILIENCE: recovery after bad requests")
    print("  " + "-" * 60)

    bad_requests = [
        ("POST", "/api/ask", {"question": ""}),
        ("POST", "/api/documents/presign",
         {"filename": "x", "content_type": "text/plain", "size": 1}),
        ("GET", "/api/videos/nonexistent_xyz", None),
        ("DELETE", "/api/documents/nonexistent_xyz", None),
        ("POST", "/api/documents",
         {"doc_id": "fake", "key": "bad/key", "title": "x"}),
    ]

    # Fire bad requests
    for method, path, body in bad_requests:
        try:
            if body:
                api(method, path, json=body)
            else:
                api(method, path)
        except Exception:
            pass

    # Now verify the system still works
    ok = 0
    total = 5
    checks = [
        ("health", lambda: requests.get(BASE_URL + "/api/health", timeout=5).status_code == 200),
        ("config", lambda: requests.get(BASE_URL + "/api/config", timeout=5).status_code == 200),
        ("list_videos", lambda: api("GET", "/api/videos", user="default").status_code == 200),
        ("list_docs", lambda: api("GET", "/api/documents", user="default").status_code == 200),
        ("search", lambda: api("POST", "/api/ask", user="default",
                               json={"question": "test"}).status_code == 200),
    ]
    for name, check in checks:
        try:
            if check():
                ok += 1
            else:
                print(f"    {name}: FAILED")
        except Exception as e:
            print(f"    {name}: ERROR ({e})")

    print(f"  Post-error checks: {ok}/{total} passed")
    print(f"  Result:      {'PASS' if ok == total else 'FAIL'}")

    return {"ok": ok, "total": total}


def resilience_sse_stream():
    """Verify /api/ask_stream returns valid SSE events."""
    print("\n  RESILIENCE: SSE /api/ask_stream")
    print("  " + "-" * 60)

    try:
        r = requests.post(BASE_URL + "/api/ask_stream",
                          headers={"X-User-Id": "default",
                                   "Content-Type": "application/json"},
                          json={"question": "What is deep learning?"},
                          stream=True, timeout=30)
        if r.status_code != 200:
            print(f"  Status: {r.status_code}  FAIL")
            return {"ok": False}

        events = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())

        has_citations = "citations" in events
        has_token = "token" in events
        has_done = "done" in events
        print(f"  Events:      {events}")
        print(f"  citations:   {'yes' if has_citations else 'MISSING'}")
        print(f"  token:       {'yes' if has_token else 'MISSING'}")
        print(f"  done:        {'yes' if has_done else 'MISSING'}")
        ok = has_citations and has_token and has_done
        print(f"  Result:      {'PASS' if ok else 'FAIL'}")
        return {"ok": ok, "events": events}
    except Exception as e:
        print(f"  Error: {e}  FAIL")
        return {"ok": False}


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results, resilience=False, concurrency=CONCURRENCY, rounds=SEARCH_ROUNDS):
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    sl = results.get("search_latency", {})
    if sl.get("latencies"):
        lats = sl["latencies"]
        print(f"  Search p50:          {percentile(lats, 50):.0f} ms")
        print(f"  Search p95:          {percentile(lats, 95):.0f} ms")
        print(f"  Search errors:       {sl['errors']}")

    tp = results.get("throughput", {})
    if tp.get("qps"):
        print(f"  Throughput:          {tp['qps']:.1f} QPS")

    al = results.get("accept_latency", {})
    if al.get("latencies"):
        print(f"  Accept p95:          {percentile(al['latencies'], 95):.0f} ms")

    resilience_results = {}
    if resilience:
        print(f"\n  Resilience tests:")
        for key in ("search_during_ingest", "burst_register",
                    "mixed_ops", "recovery", "sse_stream"):
            r = results.get(key, {})
            if key == "search_during_ingest":
                ok = r.get("errors", 1) == 0
            elif key == "burst_register":
                ok = r.get("accepted", 0) == r.get("staged", 1)
            elif key == "mixed_ops":
                ok = r.get("errors", 1) == 0
            elif key == "recovery":
                ok = r.get("ok", 0) == r.get("total", 1)
            elif key == "sse_stream":
                ok = r.get("ok", False)
            else:
                ok = False
            resilience_results[key] = ok
            print(f"    {key:30s} {'PASS' if ok else 'FAIL'}")

    print("=" * 70)

    # Save JSON report
    now = datetime.now(timezone.utc)
    report = {
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "base_url": BASE_URL,
        "config": {"concurrency": concurrency, "rounds": rounds,
                    "resilience": resilience},
        "search_latency": {
            "p50_ms": percentile(sl.get("latencies", []), 50),
            "p95_ms": percentile(sl.get("latencies", []), 95),
            "p99_ms": percentile(sl.get("latencies", []), 99),
            "min_ms": min(sl["latencies"]) if sl.get("latencies") else None,
            "max_ms": max(sl["latencies"]) if sl.get("latencies") else None,
            "errors": sl.get("errors", 0),
        },
        "accept_latency": {
            "p50_ms": percentile(al.get("latencies", []), 50),
            "p95_ms": percentile(al.get("latencies", []), 95),
            "samples": len(al.get("latencies", [])),
            "pass": percentile(al.get("latencies", []), 95) < 500,
        },
        "throughput": {
            "qps": tp.get("qps", 0),
            "wall_s": tp.get("wall_s", 0),
            "errors": tp.get("errors", 0),
        },
    }
    if resilience:
        report["resilience"] = resilience_results

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"bench_{now.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global BASE_URL, SEARCH_ROUNDS
    parser = argparse.ArgumentParser(description="MomentSearch benchmark")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--rounds", type=int, default=SEARCH_ROUNDS)
    parser.add_argument("--resilience", action="store_true",
                        help="Run resilience tests (search during ingest, burst, recovery)")
    args = parser.parse_args()

    BASE_URL = args.base_url
    SEARCH_ROUNDS = args.rounds

    # Verify stack is up
    print("=" * 70)
    print(f"MomentSearch Benchmark — {BASE_URL}")
    print(f"Concurrency: {args.concurrency}  Rounds: {args.rounds}  "
          f"Resilience: {'yes' if args.resilience else 'no'}")
    print("=" * 70)

    try:
        r = requests.get(BASE_URL + "/api/health", timeout=5)
        if r.status_code != 200:
            print("Stack not healthy — aborting.")
            sys.exit(1)
    except Exception:
        print("Stack unreachable — aborting.")
        sys.exit(1)

    results = {}

    results["search_latency"] = bench_search_latency(args.concurrency)
    results["accept_latency"] = bench_accept_latency()
    results["throughput"] = bench_throughput(args.concurrency)

    if args.resilience:
        print("\n" + "=" * 70)
        print("RESILIENCE TESTS")
        print("=" * 70)
        results["search_during_ingest"] = resilience_search_during_ingest()
        results["burst_register"] = resilience_burst_register()
        results["mixed_ops"] = resilience_mixed_operations()
        results["recovery"] = resilience_recovery()
        results["sse_stream"] = resilience_sse_stream()

    cleanup()
    print_report(results, resilience=args.resilience,
                 concurrency=args.concurrency, rounds=args.rounds)


if __name__ == "__main__":
    main()
