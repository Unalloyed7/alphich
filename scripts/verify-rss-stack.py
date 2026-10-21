#!/usr/bin/env python3
"""Offline, bill-free six-service RSS Docker lifecycle acceptance.

Requires already-running 5-service Alphich with DEMO_A imported.
The collector's test-only Compose override suspends real external polling.
No paid model endpoints are called.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "-f", "compose.yaml",
           "-f", "scripts/rss-smoke.compose.yml", "--profile", "news"]
SIX = {"postgres", "api", "worker", "analysis-worker", "frontend", "rss-collector"}


def command(*args, capture=False):
    completed = subprocess.run([*COMPOSE, *args], cwd=ROOT, check=True,
                               stdout=subprocess.PIPE if capture else None, text=True)
    return (completed.stdout or "").strip()


def run(company):
    test_id = uuid.uuid4().hex[:12]
    print("Starting controlled sixth service (no live RSS / no LLM)...", flush=True)
    command("up", "-d", "--build", "rss-collector")
    services = set(command("ps", "--services", "--status", "running", capture=True).splitlines())
    if missing := SIX - services:
        raise RuntimeError("Missing running services: " + ", ".join(sorted(missing)))
    try:
        def phase(name):
            command("run", "--rm", "-T", "--no-deps", "rss-collector",
                    "python", "-m", "app.rss_smoke", "--test-id", test_id,
                    "--phase", name, "--company", company)
        phase("first")
        print("Restarting rss-collector (retains PostgreSQL state)...", flush=True)
        command("restart", "rss-collector")
        phase("replay")
        print("PASS RSS -> job -> event -> source -> 304 -> no duplicates -> restart", flush=True)
    finally:
        # Do not leave a suspended test collector running in the deployment.
        command("stop", "rss-collector")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="DEMO_A")
    args = parser.parse_args()
    try:
        run(args.company)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print("FAIL " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
