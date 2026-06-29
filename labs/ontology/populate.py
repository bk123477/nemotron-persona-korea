"""
Stage 5: JSONL → RDF 트리플 변환

페르소나 JSONL 데이터를 읽어 OWL 인스턴스로 변환하고 .ttl 파일로 저장한다.

사용법:
    cd /home/minkih/nemotron-persona-korea

    # 샘플 10,000건 변환 (기본)
    .venv/bin/python3 -m labs.ontology.populate

    # 전체 변환
    .venv/bin/python3 -m labs.ontology.populate --all

    # 출력 경로 지정
    .venv/bin/python3 -m labs.ontology.populate --max-docs 5000 --output /tmp/personas.ttl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import quote

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from labs.common.config import JSONL_PATH
from labs.ontology.schema import NEMO, NEMO_DATA, build_schema

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_TTL_PATH = _DATA_DIR / "personas.ttl"


def _uri_safe(value: str) -> str:
    """URI에 안전한 문자열로 변환."""
    return quote(str(value).strip().replace(" ", "_"), safe="")


def _age_group_key(age: int) -> str:
    if age < 30:
        return "youth"
    if age < 50:
        return "middle"
    if age < 65:
        return "senior"
    return "elderly"


def populate(
    jsonl_path: Path = JSONL_PATH,
    max_docs: int | None = None,
    output_path: Path = _TTL_PATH,
) -> Graph:
    """JSONL → RDF 그래프 변환 후 TTL 파일로 저장.

    Args:
        jsonl_path : 입력 JSONL 파일 경로
        max_docs   : 변환할 최대 레코드 수 (None = 전체)
        output_path: 출력 TTL 파일 경로

    Returns:
        생성된 RDF 그래프
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g = build_schema()

    # 카테고리 인스턴스 캐시 (value → URI) — 중복 트리플 방지
    _cache: dict[str, dict[str, URIRef]] = {
        "prov": {}, "dist": {}, "occ": {}, "edu": {},
        "mar": {}, "fam": {}, "hou": {}, "mil": {},
    }

    def _cat_uri(prefix: str, value: str, rdf_class: URIRef,
                 extra_prop: URIRef | None = None) -> URIRef:
        if value not in _cache[prefix]:
            uri = NEMO_DATA[f"{prefix}_{_uri_safe(value)}"]
            g.add((uri, RDF.type, rdf_class))
            g.add((uri, RDFS.label, Literal(value, lang="ko")))
            if extra_prop:
                g.add((uri, extra_prop, Literal(value, lang="ko")))
            _cache[prefix][value] = uri
        return _cache[prefix][value]

    count = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if max_docs and count >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = rec.get("metadata", {})
            uid = meta.get("uuid") or rec.get("id", f"persona_{count}")
            p_uri = NEMO_DATA[f"persona_{_uri_safe(str(uid))}"]

            g.add((p_uri, RDF.type, NEMO.Persona))
            g.add((p_uri, NEMO.hasUUID, Literal(str(uid), datatype=XSD.string)))

            # 나이
            try:
                age = int(meta.get("age", 0))
                g.add((p_uri, NEMO.hasAge, Literal(age, datatype=XSD.integer)))
                g.add((p_uri, NEMO.hasAgeGroup, NEMO_DATA[f"age_{_age_group_key(age)}"]))
            except (ValueError, TypeError):
                pass

            # 성별
            sex = str(meta.get("sex", "")).strip()
            if sex in ("남자", "여자"):
                g.add((p_uri, NEMO.hasSex, NEMO_DATA[f"sex_{sex}"]))

            # 시도 (Province)
            province = str(meta.get("province", "")).strip()
            if province:
                prov_uri = _cat_uri("prov", province, NEMO.Province, NEMO.hasProvinceName)
                g.add((p_uri, NEMO.livesInProvince, prov_uri))

            # 시군구 (District)
            district = str(meta.get("district", "")).strip()
            if district:
                dist_key = f"{province}_{district}" if province else district
                dist_uri = _cat_uri("dist", dist_key, NEMO.District, NEMO.hasDistrictName)
                g.add((p_uri, NEMO.livesInDistrict, dist_uri))
                if province and province in _cache["prov"]:
                    g.add((dist_uri, NEMO.partOf, _cache["prov"][province]))

            # 직업
            occ = str(meta.get("occupation", "")).strip()
            if occ:
                occ_uri = _cat_uri("occ", occ, NEMO.OccupationCategory, NEMO.hasOccupationName)
                g.add((p_uri, NEMO.hasOccupation, occ_uri))

            # 학력
            edu = str(meta.get("education_level", "")).strip()
            if edu:
                edu_uri = _cat_uri("edu", edu, NEMO.EducationLevel, NEMO.hasEducationName)
                g.add((p_uri, NEMO.hasEducation, edu_uri))

            # 전공 계열
            bachelors = str(meta.get("bachelors_field", "")).strip()
            if bachelors:
                g.add((p_uri, NEMO.hasBachelorsField, Literal(bachelors, lang="ko")))

            # 혼인상태
            marital = str(meta.get("marital_status", "")).strip()
            if marital:
                mar_uri = _cat_uri("mar", marital, NEMO.MaritalStatus)
                g.add((p_uri, NEMO.hasMaritalStatus, mar_uri))

            # 가구형태
            family = str(meta.get("family_type", "")).strip()
            if family:
                fam_uri = _cat_uri("fam", family, NEMO.FamilyType)
                g.add((p_uri, NEMO.hasFamilyType, fam_uri))

            # 주거형태
            housing = str(meta.get("housing_type", "")).strip()
            if housing:
                hou_uri = _cat_uri("hou", housing, NEMO.HousingType)
                g.add((p_uri, NEMO.hasHousingType, hou_uri))

            # 병역상태
            military = str(meta.get("military_status", "")).strip()
            if military:
                mil_uri = _cat_uri("mil", military, NEMO.MilitaryStatus)
                g.add((p_uri, NEMO.hasMilitaryStatus, mil_uri))

            count += 1
            if count % 1000 == 0:
                logger.info("%d건 처리됨 (트리플: %d)", count, len(g))

    g.serialize(destination=str(output_path), format="turtle")
    print(f"[완료] {count:,}건 → {output_path}  (트리플: {len(g):,}개)")
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL → RDF 트리플 변환 (Stage 5 온톨로지)")
    parser.add_argument("--max-docs", type=int, default=10_000,
                        help="변환할 최대 레코드 수 (기본: 10,000)")
    parser.add_argument("--all", action="store_true",
                        help="전체 JSONL 변환 (--max-docs 무시)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 TTL 파일 경로")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    output = Path(args.output) if args.output else _TTL_PATH
    max_docs = None if args.all else args.max_docs

    print(f"[JSONL 경로] {JSONL_PATH}")
    print(f"[출력 경로]  {output}")
    print(f"[변환 대수]  {'전체' if max_docs is None else f'{max_docs:,}건'}")

    populate(max_docs=max_docs, output_path=output)


if __name__ == "__main__":
    main()
