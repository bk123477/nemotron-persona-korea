"""
Stage 5: SPARQL 질의 예시 모음

사전 생성된 TTL 파일에 대해 다양한 SPARQL SELECT 질의를 실행한다.

사용법:
    cd /home/minkih/nemotron-persona-korea

    # 전체 예시 실행
    .venv/bin/python3 -m labs.ontology.query

    # 특정 질의 번호만 실행
    .venv/bin/python3 -m labs.ontology.query --query 2

    # TTL 파일 경로 지정
    .venv/bin/python3 -m labs.ontology.query --ttl /tmp/personas.ttl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rdflib import Graph

_TTL_PATH = Path(__file__).parent / "data" / "personas.ttl"

_PFX = """
PREFIX nemo: <http://nemotron.persona.kr/ontology#>
PREFIX data: <http://nemotron.persona.kr/data/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

QUERIES: dict[int, dict] = {
    1: {
        "title": "시도별 페르소나 수",
        "sparql": _PFX + """
SELECT ?provinceName (COUNT(?p) AS ?count)
WHERE {
  ?p a nemo:Persona ; nemo:livesInProvince ?prov .
  ?prov nemo:hasProvinceName ?provinceName .
}
GROUP BY ?provinceName
ORDER BY DESC(?count)
""",
    },
    2: {
        "title": "직업별 평균 나이 (상위 20개)",
        "sparql": _PFX + """
SELECT ?occName (AVG(?age) AS ?avgAge) (COUNT(?p) AS ?count)
WHERE {
  ?p a nemo:Persona ; nemo:hasOccupation ?occ ; nemo:hasAge ?age .
  ?occ nemo:hasOccupationName ?occName .
}
GROUP BY ?occName
ORDER BY DESC(?count)
LIMIT 20
""",
    },
    3: {
        "title": "학력 × 혼인상태 교차 분석",
        "sparql": _PFX + """
SELECT ?eduName ?marLabel (COUNT(?p) AS ?count)
WHERE {
  ?p a nemo:Persona ;
     nemo:hasEducation ?edu ;
     nemo:hasMaritalStatus ?mar .
  ?edu nemo:hasEducationName ?eduName .
  ?mar rdfs:label ?marLabel .
  FILTER(LANG(?marLabel) = "ko")
}
GROUP BY ?eduName ?marLabel
ORDER BY DESC(?count)
LIMIT 30
""",
    },
    4: {
        "title": "성별 × 연령대 분포",
        "sparql": _PFX + """
SELECT ?sexLabel ?ageLabel (COUNT(?p) AS ?count)
WHERE {
  ?p a nemo:Persona ;
     nemo:hasSex ?sex ;
     nemo:hasAgeGroup ?ag .
  ?sex rdfs:label ?sexLabel .
  ?ag  rdfs:label ?ageLabel .
  FILTER(LANG(?sexLabel) = "ko" && LANG(?ageLabel) = "ko")
}
GROUP BY ?sexLabel ?ageLabel
ORDER BY ?sexLabel ?ageLabel
""",
    },
    5: {
        "title": "서울 거주 고령층 직업 분포",
        "sparql": _PFX + """
SELECT ?occName (COUNT(?p) AS ?count)
WHERE {
  ?p a nemo:Persona ;
     nemo:livesInProvince ?prov ;
     nemo:hasAgeGroup data:age_elderly ;
     nemo:hasOccupation ?occ .
  ?prov nemo:hasProvinceName "서울"@ko .
  ?occ  nemo:hasOccupationName ?occName .
}
GROUP BY ?occName
ORDER BY DESC(?count)
LIMIT 15
""",
    },
    6: {
        "title": "가구형태별 분포",
        "sparql": _PFX + """
SELECT ?famLabel (COUNT(?p) AS ?count)
WHERE {
  ?p a nemo:Persona ; nemo:hasFamilyType ?fam .
  ?fam rdfs:label ?famLabel .
  FILTER(LANG(?famLabel) = "ko")
}
GROUP BY ?famLabel
ORDER BY DESC(?count)
""",
    },
    7: {
        "title": "고학력 미혼 20~35세 페르소나 목록",
        "sparql": _PFX + """
SELECT ?p ?age ?provName ?occName
WHERE {
  ?p a nemo:Persona ;
     nemo:hasAge ?age ;
     nemo:hasMaritalStatus ?mar ;
     nemo:hasEducation ?edu ;
     nemo:hasOccupation ?occ ;
     nemo:livesInProvince ?prov .
  ?edu nemo:hasEducationName ?eduName .
  ?mar rdfs:label ?marLabel .
  ?occ nemo:hasOccupationName ?occName .
  ?prov nemo:hasProvinceName ?provName .
  FILTER(LANG(?marLabel) = "ko")
  FILTER(CONTAINS(?marLabel, "미혼"))
  FILTER(CONTAINS(?eduName, "4년제") || CONTAINS(?eduName, "대학원"))
  FILTER(?age >= 20 && ?age <= 35)
}
ORDER BY ?age
LIMIT 20
""",
    },
    8: {
        "title": "시도별 1인 가구 비율",
        "sparql": _PFX + """
SELECT ?provName
       (COUNT(?p) AS ?total)
       (SUM(IF(CONTAINS(?famLabel, "혼자"), 1, 0)) AS ?single)
WHERE {
  ?p a nemo:Persona ;
     nemo:livesInProvince ?prov ;
     nemo:hasFamilyType ?fam .
  ?prov nemo:hasProvinceName ?provName .
  ?fam  rdfs:label ?famLabel .
  FILTER(LANG(?famLabel) = "ko")
}
GROUP BY ?provName
ORDER BY DESC(?single)
""",
    },
}


def load_graph(ttl_path: Path = _TTL_PATH) -> Graph:
    """TTL 파일에서 RDF 그래프를 로드한다."""
    if not ttl_path.exists():
        raise FileNotFoundError(
            f"TTL 파일이 없습니다: {ttl_path}\n"
            "먼저 다음을 실행하세요:\n"
            "  .venv/bin/python3 -m labs.ontology.populate --max-docs 10000"
        )
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    return g


def run_query(g: Graph, query_no: int) -> list:
    """단일 SPARQL 질의를 실행하고 결과 rows를 반환한다."""
    if query_no not in QUERIES:
        raise KeyError(f"질의 번호 {query_no}가 없습니다. 사용 가능: {sorted(QUERIES)}")
    return list(g.query(QUERIES[query_no]["sparql"]))


def print_query(g: Graph, query_no: int) -> None:
    """질의를 실행하고 결과를 출력한다."""
    q = QUERIES[query_no]
    print(f"\n{'═' * 60}")
    print(f"[질의 {query_no}] {q['title']}")
    print('─' * 60)
    rows = run_query(g, query_no)
    if not rows:
        print("  결과 없음")
        return
    for row in rows[:20]:
        print("  " + " | ".join(str(v) for v in row))
    if len(rows) > 20:
        print(f"  ... (총 {len(rows)}건 중 20건 표시)")


def main() -> None:
    parser = argparse.ArgumentParser(description="SPARQL 질의 예시 (Stage 5 온톨로지)")
    parser.add_argument("--query", type=int, default=None,
                        help=f"실행할 질의 번호 (사용 가능: {sorted(QUERIES.keys())})")
    parser.add_argument("--ttl", type=str, default=None,
                        help="TTL 파일 경로 (기본: labs/ontology/data/personas.ttl)")
    args = parser.parse_args()

    ttl_path = Path(args.ttl) if args.ttl else _TTL_PATH
    print(f"[그래프 로딩] {ttl_path}")
    try:
        g = load_graph(ttl_path)
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    print(f"[트리플 수] {len(g):,}개")

    if args.query:
        print_query(g, args.query)
    else:
        for qno in sorted(QUERIES):
            print_query(g, qno)


if __name__ == "__main__":
    main()
