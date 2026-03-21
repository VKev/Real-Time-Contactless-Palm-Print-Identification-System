#!/bin/sh
set -eu

python - <<'PY'
import os
import socket
import sys
import time


def wait_for_tcp(host: str, port: int, label: str, retries: int = 90, delay: float = 2.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((host, port), timeout=2.0):
                print(f"[startup] {label} is reachable at {host}:{port}", flush=True)
                return
        except OSError:
            print(f"[startup] waiting for {label} at {host}:{port} ({attempt}/{retries})", flush=True)
            time.sleep(delay)

    print(f"[startup] timed out waiting for {label} at {host}:{port}", file=sys.stderr, flush=True)
    raise SystemExit(1)


triton_grpc = os.getenv("TRITON_GRPC_URL", "triton:8001").strip()
if ":" in triton_grpc:
    triton_host, triton_port = triton_grpc.rsplit(":", 1)
else:
    triton_host, triton_port = triton_grpc, "8001"

qdrant_host = os.getenv("QDRANT_HOST", "qdrant").strip()
qdrant_port = os.getenv("QDRANT_PORT", "6333").strip()

wait_for_tcp(triton_host or "triton", int(triton_port), "Triton gRPC")
wait_for_tcp(qdrant_host or "qdrant", int(qdrant_port), "Qdrant")
PY

exec uvicorn backend:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-7001}"
