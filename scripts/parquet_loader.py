from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from pyarrow.lib import ArrowInvalid


@dataclass
class PersonaDocument:
    id: str
    text: str
    metadata: Dict[str, Any]


DEFAULT_PERSONA_FIELDS = [
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
]

DEFAULT_METADATA_FIELDS = [
    "uuid",
    "sex",
    "age",
    "marital_status",
    "military_status",
    "family_type",
    "housing_type",
    "education_level",
    "bachelors_field",
    "occupation",
    "district",
    "province",
    "country",
]


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def list_parquet_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.parquet") if p.is_file()])


def read_parquet_file(path: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    return table.to_pandas()


def read_parquet_row_groups(path: Path, columns: Optional[Iterable[str]] = None) -> Generator[pd.DataFrame, None, None]:
    parquet_file = pq.ParquetFile(path)
    for group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(group_index, columns=list(columns) if columns is not None else None)
        yield table.to_pandas()


def build_text_content(row: Dict[str, Any], persona_fields: Optional[List[str]] = None) -> str:
    persona_fields = persona_fields or DEFAULT_PERSONA_FIELDS
    parts: List[str] = []
    for field in persona_fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            label = field.replace("_", " ").capitalize()
            parts.append(f"{label}: {value}")
    return "\n".join(parts).strip()


def build_metadata(row: Dict[str, Any], exclude_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    exclude_fields = set(exclude_fields or [])
    metadata: Dict[str, Any] = {}
    for key, value in row.items():
        if key in exclude_fields:
            continue
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        metadata[key] = _to_json_safe(value)
    return metadata


def iter_persona_docs_from_hf(
    repo_id: str = "nvidia/Nemotron-Personas-Korea",
    split: str = "train",
    persona_fields: Optional[List[str]] = None,
) -> Generator[PersonaDocument, None, None]:
    dataset = load_dataset(repo_id, split=split)
    for row in dataset:
        text = build_text_content(row, persona_fields=persona_fields)
        metadata = build_metadata(row, exclude_fields=["persona"])
        yield PersonaDocument(
            id=str(row.get("uuid", "")),
            text=text,
            metadata=metadata,
        )


def iter_persona_docs(
    input_dir: Path,
    persona_fields: Optional[List[str]] = None,
    metadata_fields: Optional[List[str]] = None,
) -> Generator[PersonaDocument, None, None]:
    files = list_parquet_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    for parquet_path in files:
        for df in read_parquet_row_groups(parquet_path):
            for row in df.to_dict(orient="records"):
                text = build_text_content(row, persona_fields=persona_fields)
                metadata = build_metadata(row, exclude_fields=["persona"])
                yield PersonaDocument(
                    id=str(row.get("uuid", "")),
                    text=text,
                    metadata=metadata,
                )


def _iter_persona_docs_with_fallback(
    input_dir: Path,
    persona_fields: Optional[List[str]] = None,
    hf_repo_id: str = "nvidia/Nemotron-Personas-Korea",
    hf_split: str = "train",
) -> Generator[PersonaDocument, None, None]:
    try:
        yield from iter_persona_docs(input_dir, persona_fields=persona_fields)
    except (FileNotFoundError, ArrowInvalid, OSError) as exc:
        print(f"Local parquet failed: {exc}")
        print("Falling back to Hugging Face remote dataset loader...")
        yield from iter_persona_docs_from_hf(repo_id=hf_repo_id, split=hf_split, persona_fields=persona_fields)


def export_jsonl(
    input_dir: Path,
    output_path: Path,
    persona_fields: Optional[List[str]] = None,
    max_docs: Optional[int] = None,
    hf_repo_id: str = "nvidia/Nemotron-Personas-Korea",
    hf_split: str = "train",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for doc in _iter_persona_docs_with_fallback(
            input_dir,
            persona_fields=persona_fields,
            hf_repo_id=hf_repo_id,
            hf_split=hf_split,
        ):
            line = json.dumps(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                },
                ensure_ascii=False,
            )
            fp.write(line + "\n")
            count += 1
            if max_docs is not None and count >= max_docs:
                break
    print(f"Exported {count} documents to {output_path}")
