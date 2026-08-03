#!/usr/bin/env python3
"""Ask questions about a folder of documents from the terminal.

    python cli.py sample_docs "how do I rotate an API key?"
    python cli.py sample_docs            # interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragqa import RagPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="file or directory to index")
    parser.add_argument("question", nargs="*", help="question; omit for interactive mode")
    parser.add_argument("-k", type=int, default=4, help="passages to retrieve (default 4)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"no such path: {args.path}", file=sys.stderr)
        return 1

    pipeline = RagPipeline(top_k=args.k)
    count = pipeline.index_path(args.path)
    print(
        f"indexed {count} chunks from {args.path} "
        f"(embedder: {pipeline.store.embedder.name}, answerer: {pipeline.answerer.name})\n"
    )

    if args.question:
        print(pipeline.ask(" ".join(args.question)))
        return 0

    print("Ask a question, or press Ctrl-D to exit.\n")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question:
            print(f"\n{pipeline.ask(question)}\n")


if __name__ == "__main__":
    raise SystemExit(main())
