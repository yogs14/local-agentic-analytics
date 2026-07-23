"""
Entry point untuk menjalankan API server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080
    python scripts/run_api.py --host 127.0.0.1
    python scripts/run_api.py --reload   # dev mode
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MAX_PORT_RETRIES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Agentic Analytics API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true", help="Dev mode auto-reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn tidak terinstall. Jalankan: pip install -r requirements.txt")
        sys.exit(1)

    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    port = args.port

    for _ in range(MAX_PORT_RETRIES):
        if port != args.port:
            print(f"Info: Port {port - 1} sudah terpakai, mencoba port {port}...")

        try:
            print(f"Local Agentic Analytics API")
            print(f"  URL:  http://{display_host}:{port}")
            print(f"  Docs: http://{display_host}:{port}/docs")
            print("Ctrl+C untuk berhenti.")

            uvicorn.run(
                "local_agentic_analytics.api.app:app",
                host=args.host,
                port=port,
                reload=args.reload,
                log_level="info",
            )
            return
        except SystemExit:
            return
        except OSError:
            port += 1

    print(f"Error: Gagal bind port {args.port} s/d {port - 1}. Semua terpakai.")
    sys.exit(1)


if __name__ == "__main__":
    main()
