from __future__ import annotations

import argparse
from pathlib import Path

from parquet_loader import export_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Nemotron Personas Korea parquet files into JSONL for RAG/LangChain workflows."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Nemotron-Personas-Korea/data"),
        help="Directory containing Nemotron parquet files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("Nemotron-Personas-Korea/data/jsonl/personas.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional maximum number of documents to export (useful for quick samples).",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="nvidia/Nemotron-Personas-Korea",
        help="Hugging Face dataset repository used if local parquet files are not available.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Dataset split to use from Hugging Face when falling back.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading parquet files from: {args.input_dir}")
    print(f"Writing JSONL to: {args.output_path}")
    if args.sample:
        print(f"Exporting sample of {args.sample} documents")

    export_jsonl(
        input_dir=args.input_dir,
        output_path=args.output_path,
        max_docs=args.sample,
        hf_repo_id=args.hf_repo,
        hf_split=args.hf_split,
    )


if __name__ == "__main__":
    main()
