"""
Measure the loading-screen wait: time from the first /health call until the server says it's ready.

This is what the frontend does on page load, so it's the wait users see. For a true cold start,
no container may be running: the deployment scales to zero after 7 idle minutes (scaledown_window=420),
so wait at least that long with no traffic before each measurement (--idle).

    python eval/coldstart.py --url https://...modal.run --runs 1 --idle 0   --out eval/runs/coldstart_v1.jsonl
    python eval/coldstart.py --url https://...modal.run --runs 2 --idle 540 --out eval/runs/coldstart_v1.jsonl

Each run is appended to --out as one JSON line. Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.request

POLL_INTERVAL_S = 1.5   # same as the frontend
GIVE_UP_AFTER_S = 900


def wait_until_ready(url: str) -> tuple[float, int, dict | None]:
    t0 = time.perf_counter()
    attempts = 0
    while time.perf_counter() - t0 < GIVE_UP_AFTER_S:
        attempts += 1
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=GIVE_UP_AFTER_S) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("status") == "ok":
                return round(time.perf_counter() - t0, 1), attempts, body
        except Exception as e:  # cold container, gateway timeout, etc. -- retry like the frontend
            print(f"  attempt {attempts}: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(POLL_INTERVAL_S)
    return round(time.perf_counter() - t0, 1), attempts, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--idle", type=int, default=540, help="seconds of no traffic before each run")
    args = p.parse_args()
    url = args.url.rstrip("/")

    for run in range(1, args.runs + 1):
        if args.idle:
            print(f"Run {run}: idling {args.idle}s so the container scales to zero ...", flush=True)
            time.sleep(args.idle)
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        print(f"Run {run}: polling {url}/health from {started} ...", flush=True)
        seconds, attempts, body = wait_until_ready(url)
        record = {"started_at": started, "idle_before_s": args.idle, "seconds_to_ready": seconds,
                  "attempts": attempts, "ready": body is not None}
        print(f"Run {run}: {'ready' if body else 'GAVE UP'} after {seconds}s ({attempts} attempt(s))", flush=True)
        with open(args.out, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
