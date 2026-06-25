"""
페르소나 통계 집계 + LLM 위키 문서 자동 생성

JSONL 전체를 단일 패스로 집계하고, LLM이 각 주제(지역·연령대·직업)에 대한
마크다운 위키 문서를 생성한다. 생성 결과는 labs/wiki_lab/wiki/ 에 저장된다.

사용법:
    cd /home/minkih/nemotron-persona-korea

    # 전체 생성 (지역 17개 + 연령대 + 직업 개요)
    .venv/bin/python3 -m labs.wiki_lab.generate_wiki

    # LLM 없이 통계만 확인 (빠른 검증)
    .venv/bin/python3 -m labs.wiki_lab.generate_wiki --dry-run

    # 특정 지역만
    .venv/bin/python3 -m labs.wiki_lab.generate_wiki --region 서울

    # 특정 타입만 (regions / age_groups / occupations)
    .venv/bin/python3 -m labs.wiki_lab.generate_wiki --type age_groups
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401
from labs.common.config import (
    JSONL_PATH, LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, _PROXY,
)

_WIKI_DIR = Path(__file__).parent / "wiki"

_AGE_GROUPS = {
    "청년": (18, 29),
    "중년": (30, 49),
    "장년": (50, 64),
    "고령": (65, 120),
}


# ── 통계 집계 ────────────────────────────────────────────────────────

def _empty_bucket() -> dict:
    return {
        "total": 0,
        "sex": Counter(),
        "ages": [],
        "occupation": Counter(),
        "education": Counter(),
        "marital_status": Counter(),
        "family_type": Counter(),
        "housing_type": Counter(),
    }


def aggregate_all(jsonl_path: Path) -> dict:
    """JSONL 단일 패스로 지역·연령대·직업 전체 통계를 한 번에 집계한다."""
    province_stats: dict[str, dict] = defaultdict(_empty_bucket)
    age_group_stats: dict[str, dict] = defaultdict(_empty_bucket)
    occupation_stats: dict[str, dict] = defaultdict(_empty_bucket)
    global_stats = _empty_bucket()

    print(f"[집계 중] {jsonl_path}")
    count = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            meta = record.get("metadata", {})

            province = meta.get("province") or "미상"
            age_raw = meta.get("age")
            age = int(age_raw) if age_raw else 0
            sex = meta.get("sex") or "미상"
            occ = meta.get("occupation") or "미상"
            edu = meta.get("education_level") or "미상"
            marital = meta.get("marital_status") or "미상"
            family = meta.get("family_type") or "미상"
            housing = meta.get("housing_type") or "미상"

            # 연령대 결정
            age_group = "기타"
            for g, (lo, hi) in _AGE_GROUPS.items():
                if lo <= age <= hi:
                    age_group = g
                    break

            def _update(bucket: dict) -> None:
                bucket["total"] += 1
                bucket["sex"][sex] += 1
                if age:
                    bucket["ages"].append(age)
                bucket["occupation"][occ] += 1
                bucket["education"][edu] += 1
                bucket["marital_status"][marital] += 1
                bucket["family_type"][family] += 1
                bucket["housing_type"][housing] += 1

            _update(global_stats)
            _update(province_stats[province])
            _update(age_group_stats[age_group])

            # 직업 대분류 버킷: 첫 3글자로 그룹화 (조잡하지만 합리적인 근사)
            occ_prefix = occ[:4] if len(occ) >= 4 else occ
            _update(occupation_stats[occ_prefix])

            count += 1
            if count % 50000 == 0:
                print(f"  {count:,}건 처리 중...")

    print(f"[완료] 총 {count:,}건 집계")
    return {
        "global": global_stats,
        "province": dict(province_stats),
        "age_group": dict(age_group_stats),
        "occupation": dict(occupation_stats),
    }


# ── 통계 → 텍스트 ────────────────────────────────────────────────────

def _format_counter(counter: Counter, top_n: int = 7) -> str:
    total = sum(counter.values())
    if total == 0:
        return "데이터 없음"
    lines = []
    for k, v in counter.most_common(top_n):
        pct = v / total * 100
        lines.append(f"  - {k}: {v:,}명 ({pct:.1f}%)")
    return "\n".join(lines)


def _format_age_stats(ages: list[int]) -> str:
    if not ages:
        return "데이터 없음"
    avg = sum(ages) / len(ages)
    decades = Counter((a // 10) * 10 for a in ages)
    dist = ", ".join(f"{d}대: {c:,}명" for d, c in sorted(decades.items()))
    return f"평균 {avg:.1f}세 | 연령 분포: {dist}"


def _bucket_to_text(label: str, bucket: dict) -> str:
    total = bucket["total"]
    age_text = _format_age_stats(bucket["ages"])
    return f"""대상: {label}
총 페르소나 수: {total:,}명

[성별 분포]
{_format_counter(bucket['sex'], top_n=5)}

[나이 통계]
{age_text}

[직업 상위 7개]
{_format_counter(bucket['occupation'], top_n=7)}

[학력 분포]
{_format_counter(bucket['education'], top_n=6)}

[혼인 상태]
{_format_counter(bucket['marital_status'], top_n=5)}

[가구 형태]
{_format_counter(bucket['family_type'], top_n=5)}

[주거 형태]
{_format_counter(bucket['housing_type'], top_n=5)}"""


# ── LLM 문서 생성 ────────────────────────────────────────────────────

_WIKI_SYSTEM = (
    "당신은 한국 사회 인구통계 전문 데이터 저널리스트입니다. "
    "아래 통계 데이터를 바탕으로 위키 문서를 한국어 마크다운으로 작성하세요.\n\n"
    "요구사항:\n"
    "- ## 소제목으로 구조화하세요 (개요 / 인구 구조 / 직업 특성 / 생활 특성 / 요약)\n"
    "- 숫자와 비율을 구체적으로 인용하세요\n"
    "- 데이터에 없는 내용은 절대 추측하지 마세요\n"
    "- 500~700자 분량으로 작성하세요\n"
    "- 문서 상단에 '# 제목'을 반드시 포함하세요"
)


def _make_llm():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY가 없습니다.")
    import httpx
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        http_client=httpx.Client(proxy=_PROXY, verify=False),
        temperature=0.3,
        max_tokens=1024,
    )


def _generate_article(label: str, stats_text: str, llm) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=_WIKI_SYSTEM),
        HumanMessage(content=f"다음 통계 데이터로 '{label}' 위키 문서를 작성하세요:\n\n{stats_text}"),
    ]
    return llm.invoke(messages).content


def _save_wiki(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  → 저장: {path.relative_to(_ROOT)}")


# ── 지역 위키 ────────────────────────────────────────────────────────

def generate_region_wikis(
    all_stats: dict,
    llm=None,
    dry_run: bool = False,
    only_region: Optional[str] = None,
) -> None:
    province_stats = all_stats["province"]
    global_total = all_stats["global"]["total"]

    targets = (
        {only_region: province_stats[only_region]}
        if only_region and only_region in province_stats
        else province_stats
    )

    print(f"\n[지역 위키] {len(targets)}개 지역 생성")
    for province, bucket in sorted(targets.items(), key=lambda x: -x[1]["total"]):
        if province == "미상":
            continue
        pct = bucket["total"] / global_total * 100
        stats_text = _bucket_to_text(f"{province} 지역 페르소나", bucket)
        stats_text += f"\n\n전체 대비 비율: {pct:.1f}%"

        if dry_run:
            print(f"  [DRY] {province}: {bucket['total']:,}명 ({pct:.1f}%)")
            _save_wiki(
                _WIKI_DIR / "regions" / f"{province}.md",
                f"# {province} 지역 인구통계 위키\n\n```\n{stats_text}\n```\n",
            )
            continue

        print(f"  [{province}] 문서 생성 중... ({bucket['total']:,}명)")
        article = _generate_article(f"{province} 지역 페르소나 특성", stats_text, llm)
        _save_wiki(_WIKI_DIR / "regions" / f"{province}.md", article)


# ── 연령대 위키 ──────────────────────────────────────────────────────

def generate_age_group_wikis(
    all_stats: dict,
    llm=None,
    dry_run: bool = False,
) -> None:
    age_group_stats = all_stats["age_group"]
    print(f"\n[연령대 위키] {len(_AGE_GROUPS)}개 연령대 생성")

    articles = []
    for group in ["청년", "중년", "장년", "고령"]:
        bucket = age_group_stats.get(group)
        if not bucket or bucket["total"] == 0:
            print(f"  [{group}] 데이터 없음 — 건너뜀")
            continue

        stats_text = _bucket_to_text(f"{group} (연령대)", bucket)

        if dry_run:
            print(f"  [DRY] {group}: {bucket['total']:,}명")
            articles.append(f"## {group}\n\n```\n{stats_text}\n```\n")
            continue

        print(f"  [{group}] 문서 생성 중... ({bucket['total']:,}명)")
        article = _generate_article(f"{group} 연령대 한국인 페르소나 특성", stats_text, llm)
        articles.append(article)

    if articles:
        combined = "\n\n---\n\n".join(articles)
        _save_wiki(_WIKI_DIR / "demographics" / "age_groups.md", combined)


# ── 직업 개요 위키 ───────────────────────────────────────────────────

def generate_occupation_wiki(
    all_stats: dict,
    llm=None,
    dry_run: bool = False,
) -> None:
    global_bucket = all_stats["global"]
    occ_counter = global_bucket["occupation"]
    total = global_bucket["total"]

    # 상위 30개 직업만 다룸
    top_occs = occ_counter.most_common(30)
    stats_text = f"총 페르소나 수: {total:,}명\n\n[직업 분포 상위 30개]\n"
    for occ, cnt in top_occs:
        pct = cnt / total * 100
        stats_text += f"  - {occ}: {cnt:,}명 ({pct:.1f}%)\n"

    print(f"\n[직업 개요 위키] 상위 30개 직업")
    if dry_run:
        print("  [DRY] 직업 개요 — 통계 저장")
        _save_wiki(
            _WIKI_DIR / "occupations" / "overview.md",
            f"# 직업별 페르소나 분포 개요\n\n```\n{stats_text}\n```\n",
        )
        return

    print("  [직업 개요] 문서 생성 중...")
    article = _generate_article("한국인 직업 분포 및 특성 개요", stats_text, llm)
    _save_wiki(_WIKI_DIR / "occupations" / "overview.md", article)


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="위키 문서 자동 생성")
    parser.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 통계만 저장")
    parser.add_argument("--region", type=str, default=None, help="특정 지역만 생성")
    parser.add_argument(
        "--type", choices=["regions", "age_groups", "occupations"], default=None,
        help="생성할 위키 타입 (기본: 전체)",
    )
    args = parser.parse_args()

    if not JSONL_PATH.exists():
        print(f"[오류] JSONL 파일이 없습니다: {JSONL_PATH}")
        print("scripts/convert_personas_jsonl.py를 먼저 실행하세요.")
        sys.exit(1)

    # 통계 집계 (단일 패스)
    all_stats = aggregate_all(JSONL_PATH)

    # LLM 초기화
    llm = None
    if not args.dry_run:
        llm = _make_llm()

    generate_type = args.type

    if generate_type in (None, "regions"):
        generate_region_wikis(all_stats, llm=llm, dry_run=args.dry_run, only_region=args.region)

    if generate_type in (None, "age_groups") and not args.region:
        generate_age_group_wikis(all_stats, llm=llm, dry_run=args.dry_run)

    if generate_type in (None, "occupations") and not args.region:
        generate_occupation_wiki(all_stats, llm=llm, dry_run=args.dry_run)

    print(f"\n[완료] 위키 문서가 {_WIKI_DIR} 에 저장되었습니다.")
    print("위키 RAG를 사용하려면:")
    print("  .venv/bin/python3 -m labs.wiki_lab.wiki_rag --build-index  # 최초 1회")
    print("  .venv/bin/python3 -m labs.wiki_lab.wiki_rag                # Q&A 시작")


if __name__ == "__main__":
    main()
