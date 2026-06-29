"""
Stage 5: OWL/RDF 온톨로지 스키마

Nemotron-Personas-Korea 데이터셋의 인구통계 구조를 OWL 클래스/프로퍼티로 모델링한다.

사용법:
    from labs.ontology.schema import build_schema, NEMO, NEMO_DATA
    g = build_schema()
    g.serialize("schema.ttl", format="turtle")
"""
from __future__ import annotations

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, XSD

NEMO = Namespace("http://nemotron.persona.kr/ontology#")
NEMO_DATA = Namespace("http://nemotron.persona.kr/data/")


def build_schema() -> Graph:
    """OWL 온톨로지 스키마 그래프를 생성한다."""
    g = Graph()
    g.bind("nemo", NEMO)
    g.bind("data", NEMO_DATA)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # ── 온톨로지 선언 ──────────────────────────────────────────────────
    g.add((NEMO.NemotronPersonaKorea, RDF.type, OWL.Ontology))
    g.add((NEMO.NemotronPersonaKorea, RDFS.label,
           Literal("Nemotron Personas Korea Ontology", lang="en")))
    g.add((NEMO.NemotronPersonaKorea, RDFS.comment,
           Literal("NVIDIA Nemotron-Personas-Korea 데이터셋 온톨로지", lang="ko")))

    # ── 클래스 정의 ────────────────────────────────────────────────────
    _classes = [
        (NEMO.Persona,            "페르소나",    "Korean synthetic persona"),
        (NEMO.Geographic,         "지리",        "Geographic entity"),
        (NEMO.Province,           "시도",        "Korean province (시도)"),
        (NEMO.District,           "시군구",      "Korean district (시군구)"),
        (NEMO.Country,            "국가",        "Country"),
        (NEMO.OccupationCategory, "직업",        "Occupation category"),
        (NEMO.EducationLevel,     "학력",        "Education level"),
        (NEMO.MaritalStatus,      "혼인상태",    "Marital status"),
        (NEMO.FamilyType,         "가구형태",    "Family / household type"),
        (NEMO.HousingType,        "주거형태",    "Housing type"),
        (NEMO.MilitaryStatus,     "병역상태",    "Military service status"),
        (NEMO.SexCategory,        "성별",        "Sex / gender category"),
        (NEMO.AgeGroup,           "연령대",      "Age group"),
    ]
    for cls, label_ko, label_en in _classes:
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.label, Literal(label_ko, lang="ko")))
        g.add((cls, RDFS.label, Literal(label_en, lang="en")))

    # ── 클래스 계층 ────────────────────────────────────────────────────
    g.add((NEMO.Province, RDFS.subClassOf, NEMO.Geographic))
    g.add((NEMO.District, RDFS.subClassOf, NEMO.Geographic))
    g.add((NEMO.Country,  RDFS.subClassOf, NEMO.Geographic))

    # ── 오브젝트 프로퍼티 ──────────────────────────────────────────────
    _obj_props = [
        (NEMO.hasSex,           NEMO.Persona,   NEMO.SexCategory,        "성별"),
        (NEMO.hasOccupation,    NEMO.Persona,   NEMO.OccupationCategory, "직업"),
        (NEMO.livesInDistrict,  NEMO.Persona,   NEMO.District,           "거주 시군구"),
        (NEMO.livesInProvince,  NEMO.Persona,   NEMO.Province,           "거주 시도"),
        (NEMO.hasEducation,     NEMO.Persona,   NEMO.EducationLevel,     "학력"),
        (NEMO.hasMaritalStatus, NEMO.Persona,   NEMO.MaritalStatus,      "혼인상태"),
        (NEMO.hasFamilyType,    NEMO.Persona,   NEMO.FamilyType,         "가구형태"),
        (NEMO.hasHousingType,   NEMO.Persona,   NEMO.HousingType,        "주거형태"),
        (NEMO.hasMilitaryStatus,NEMO.Persona,   NEMO.MilitaryStatus,     "병역상태"),
        (NEMO.hasAgeGroup,      NEMO.Persona,   NEMO.AgeGroup,           "연령대"),
        (NEMO.partOf,           NEMO.Geographic,NEMO.Geographic,         "상위 지역"),
    ]
    for prop, domain, range_, label_ko in _obj_props:
        g.add((prop, RDF.type, OWL.ObjectProperty))
        g.add((prop, RDFS.domain, domain))
        g.add((prop, RDFS.range, range_))
        g.add((prop, RDFS.label, Literal(label_ko, lang="ko")))

    # ── 데이터 프로퍼티 ────────────────────────────────────────────────
    _data_props = [
        (NEMO.hasUUID,           NEMO.Persona,            XSD.string,  "고유 UUID"),
        (NEMO.hasAge,            NEMO.Persona,            XSD.integer, "나이"),
        (NEMO.hasPersonaText,    NEMO.Persona,            XSD.string,  "페르소나 원문"),
        (NEMO.hasBachelorsField, NEMO.Persona,            XSD.string,  "전공 계열"),
        (NEMO.hasProvinceName,   NEMO.Province,           XSD.string,  "시도명"),
        (NEMO.hasDistrictName,   NEMO.District,           XSD.string,  "시군구명"),
        (NEMO.hasOccupationName, NEMO.OccupationCategory, XSD.string,  "직업명"),
        (NEMO.hasEducationName,  NEMO.EducationLevel,     XSD.string,  "학력명"),
    ]
    for prop, domain, range_, label_ko in _data_props:
        g.add((prop, RDF.type, OWL.DatatypeProperty))
        g.add((prop, RDFS.domain, domain))
        g.add((prop, RDFS.range, range_))
        g.add((prop, RDFS.label, Literal(label_ko, lang="ko")))

    # ── 성별 인스턴스 (고정값) ─────────────────────────────────────────
    for name in ("남자", "여자"):
        inst = NEMO_DATA[f"sex_{name}"]
        g.add((inst, RDF.type, NEMO.SexCategory))
        g.add((inst, RDFS.label, Literal(name, lang="ko")))

    # ── 연령대 인스턴스 (고정값) ──────────────────────────────────────
    _age_groups = [
        ("youth",   "청년 (18~29세)"),
        ("middle",  "중년 (30~49세)"),
        ("senior",  "장년 (50~64세)"),
        ("elderly", "고령 (65세 이상)"),
    ]
    for key, label in _age_groups:
        inst = NEMO_DATA[f"age_{key}"]
        g.add((inst, RDF.type, NEMO.AgeGroup))
        g.add((inst, RDFS.label, Literal(label, lang="ko")))

    return g
