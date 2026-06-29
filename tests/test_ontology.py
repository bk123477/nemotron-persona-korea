"""
단위/통합 테스트: Stage 5 온톨로지
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import RDF, RDFS, Graph, Literal, URIRef


# ── 스키마 테스트 ─────────────────────────────────────────────────────

class TestSchema:
    def test_build_schema_returns_graph(self):
        from labs.ontology.schema import build_schema
        g = build_schema()
        assert isinstance(g, Graph)
        assert len(g) > 0

    def test_persona_class_defined(self):
        from labs.ontology.schema import build_schema, NEMO
        g = build_schema()
        assert (NEMO.Persona, RDF.type, None) in g or \
               any(True for _ in g.triples((NEMO.Persona, RDF.type, None)))

    def test_province_subclass_of_geographic(self):
        from labs.ontology.schema import build_schema, NEMO
        g = build_schema()
        assert (NEMO.Province, RDFS.subClassOf, NEMO.Geographic) in g

    def test_district_subclass_of_geographic(self):
        from labs.ontology.schema import build_schema, NEMO
        g = build_schema()
        assert (NEMO.District, RDFS.subClassOf, NEMO.Geographic) in g

    def test_sex_instances_created(self):
        from labs.ontology.schema import build_schema, NEMO, NEMO_DATA
        g = build_schema()
        assert (NEMO_DATA["sex_남자"], RDF.type, NEMO.SexCategory) in g
        assert (NEMO_DATA["sex_여자"], RDF.type, NEMO.SexCategory) in g

    def test_age_group_instances_created(self):
        from labs.ontology.schema import build_schema, NEMO_DATA, NEMO
        g = build_schema()
        for key in ("youth", "middle", "senior", "elderly"):
            assert (NEMO_DATA[f"age_{key}"], RDF.type, NEMO.AgeGroup) in g

    def test_object_properties_have_domain_range(self):
        from rdflib import OWL
        from labs.ontology.schema import build_schema, NEMO
        g = build_schema()
        for prop in (NEMO.hasSex, NEMO.hasOccupation, NEMO.hasEducation):
            assert (prop, RDFS.domain, None) in g or \
                   any(True for _ in g.triples((prop, RDFS.domain, None)))

    def test_has_age_is_datatype_property(self):
        from rdflib import OWL
        from labs.ontology.schema import build_schema, NEMO
        g = build_schema()
        assert (NEMO.hasAge, RDF.type, OWL.DatatypeProperty) in g


# ── 인구사 테스트 ─────────────────────────────────────────────────────

class TestPopulate:
    def test_populate_small_sample(self, tmp_path, sample_jsonl):
        from labs.ontology.schema import NEMO, NEMO_DATA
        from labs.ontology.populate import populate

        out = tmp_path / "test.ttl"
        g = populate(jsonl_path=sample_jsonl, max_docs=10, output_path=out)

        assert out.exists()
        assert len(g) > 0
        # 10건의 페르소나가 생성됐는지 확인
        personas = list(g.subjects(RDF.type, NEMO.Persona))
        assert len(personas) == 10

    def test_persona_has_age(self, tmp_path, sample_jsonl):
        from labs.ontology.schema import NEMO
        from labs.ontology.populate import populate

        out = tmp_path / "test.ttl"
        g = populate(jsonl_path=sample_jsonl, max_docs=5, output_path=out)

        personas = list(g.subjects(RDF.type, NEMO.Persona))
        for p in personas[:3]:
            ages = list(g.objects(p, NEMO.hasAge))
            assert len(ages) > 0

    def test_persona_has_sex(self, tmp_path, sample_jsonl):
        from labs.ontology.schema import NEMO
        from labs.ontology.populate import populate

        out = tmp_path / "test.ttl"
        g = populate(jsonl_path=sample_jsonl, max_docs=5, output_path=out)

        personas = list(g.subjects(RDF.type, NEMO.Persona))
        has_sex = 0
        for p in personas:
            if list(g.objects(p, NEMO.hasSex)):
                has_sex += 1
        assert has_sex > 0

    def test_province_instances_created(self, tmp_path, sample_jsonl):
        from labs.ontology.schema import NEMO
        from labs.ontology.populate import populate

        out = tmp_path / "test.ttl"
        g = populate(jsonl_path=sample_jsonl, max_docs=10, output_path=out)

        provinces = list(g.subjects(RDF.type, NEMO.Province))
        assert len(provinces) > 0

    def test_output_file_is_valid_turtle(self, tmp_path, sample_jsonl):
        from labs.ontology.populate import populate

        out = tmp_path / "test.ttl"
        populate(jsonl_path=sample_jsonl, max_docs=5, output_path=out)

        # 파일을 다시 파싱해 유효한 Turtle인지 확인
        g2 = Graph()
        g2.parse(str(out), format="turtle")
        assert len(g2) > 0

    def test_max_docs_limit(self, tmp_path, sample_jsonl):
        from labs.ontology.schema import NEMO
        from labs.ontology.populate import populate

        out = tmp_path / "test.ttl"
        g = populate(jsonl_path=sample_jsonl, max_docs=3, output_path=out)

        personas = list(g.subjects(RDF.type, NEMO.Persona))
        assert len(personas) == 3


# ── SPARQL 질의 테스트 ────────────────────────────────────────────────

class TestSPARQLQueries:
    @pytest.fixture
    def populated_graph(self, tmp_path, sample_jsonl):
        from labs.ontology.populate import populate
        out = tmp_path / "test.ttl"
        return populate(jsonl_path=sample_jsonl, max_docs=10, output_path=out)

    def test_query_1_province_count(self, populated_graph):
        from labs.ontology.query import QUERIES
        rows = list(populated_graph.query(QUERIES[1]["sparql"]))
        assert len(rows) > 0  # 시도가 하나 이상 있어야 함
        counts = [int(row[1]) for row in rows]
        assert sum(counts) == 10  # 전체 합이 10건

    def test_query_2_occupation_avg_age(self, populated_graph):
        from labs.ontology.query import QUERIES
        rows = list(populated_graph.query(QUERIES[2]["sparql"]))
        assert len(rows) > 0

    def test_query_4_sex_age_distribution(self, populated_graph):
        from labs.ontology.query import QUERIES
        rows = list(populated_graph.query(QUERIES[4]["sparql"]))
        assert len(rows) > 0

    def test_load_graph_raises_on_missing_file(self, tmp_path):
        from labs.ontology.query import load_graph
        with pytest.raises(FileNotFoundError):
            load_graph(tmp_path / "nonexistent.ttl")

    def test_run_query_returns_list(self, populated_graph, tmp_path):
        from labs.ontology.query import run_query

        # populated_graph를 파일로 저장 후 load_graph로 불러오는 방식 대신
        # 직접 그래프를 사용해 run_query 테스트
        from labs.ontology.query import QUERIES
        rows = run_query(populated_graph, 1)
        assert isinstance(rows, list)

    def test_run_query_invalid_number_raises(self, populated_graph):
        from labs.ontology.query import run_query
        with pytest.raises(KeyError):
            run_query(populated_graph, 9999)
