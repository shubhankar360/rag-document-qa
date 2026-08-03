#!/usr/bin/env python3
"""Run the HTTP API.

    pip install "fastapi[standard]"
    python serve.py --index sample_docs
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, help="index this path at startup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn

        from ragqa.api import create_app
    except ImportError:
        print('the HTTP API needs FastAPI: pip install "fastapi[standard]"')
        return 1

    from ragqa import RagPipeline

    pipeline = RagPipeline()
    if args.index:
        print(f"indexed {pipeline.index_path(args.index)} chunks from {args.index}")

    uvicorn.run(create_app(pipeline), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
