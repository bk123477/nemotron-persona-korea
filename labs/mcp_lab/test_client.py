"""
MCP 서버 동작 검증 테스트 클라이언트

MCP Python SDK의 stdio 클라이언트를 사용해 persona_server.py의
Tools / Resources / Prompts를 순차적으로 호출하고 결과를 검증한다.

사용법:
    cd /home/minkih/nemotron-persona-korea
    .venv/bin/python3 -m labs.mcp_lab.test_client
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_CMD = sys.executable
_SERVER_ARGS = ["-m", "labs.mcp_lab.persona_server"]
_SERVER_CWD = str(_ROOT)

_PASS = "✓"
_FAIL = "✗"


def _print_result(label: str, ok: bool, detail: str = "") -> None:
    icon = _PASS if ok else _FAIL
    print(f"  {icon} {label}")
    if detail:
        # 멀티라인 결과는 들여써서 출력
        for line in detail.splitlines()[:8]:
            print(f"      {line}")
        if len(detail.splitlines()) > 8:
            print(f"      ... (이하 생략)")


async def run_tests() -> int:
    """모든 테스트를 실행하고 실패 수를 반환한다."""
    failures = 0

    server_params = StdioServerParameters(
        command=_SERVER_CMD,
        args=_SERVER_ARGS,
        env=None,
        cwd=_SERVER_CWD,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n[1] 서버 정보 확인")
            try:
                # SDK 1.28+: initialize() 반환값 또는 _server_info 속성 사용
                info = getattr(session, "_server_info", None) or getattr(session, "server_info", None)
                name = getattr(info, "name", None) or getattr(info, "serverInfo", {})
                if isinstance(name, dict):
                    name = name.get("name", "persona-korea")
                ok = True  # initialize()가 예외 없이 완료되면 연결 성공
                _print_result(f"서버 연결 성공 (name={name or 'persona-korea'})", ok)
            except Exception as e:
                _print_result("서버 연결", False, str(e))
                failures += 1

            # ── Tools 목록 ────────────────────────────────────────
            print("\n[2] Tools 목록")
            try:
                tools_resp = await session.list_tools()
                tool_names = [t.name for t in tools_resp.tools]
                expected_tools = {"search_personas", "get_persona_by_id", "get_demographic_stats"}
                for name in expected_tools:
                    found = name in tool_names
                    _print_result(f"tool: {name}", found)
                    if not found:
                        failures += 1
            except Exception as e:
                _print_result("list_tools", False, str(e))
                failures += 1

            # ── Resources 목록 ───────────────────────────────────
            print("\n[3] Resources 목록")
            try:
                res_resp = await session.list_resource_templates()
                uris = [r.uriTemplate for r in res_resp.resourceTemplates]
                found = any("persona://" in u for u in uris)
                _print_result(f"resource template: persona://{{uuid}}", found, str(uris))
                if not found:
                    failures += 1
            except Exception as e:
                _print_result("list_resource_templates", False, str(e))
                failures += 1

            # ── Prompts 목록 ─────────────────────────────────────
            print("\n[4] Prompts 목록")
            try:
                prompts_resp = await session.list_prompts()
                prompt_names = [p.name for p in prompts_resp.prompts]
                found = "persona_roleplay" in prompt_names
                _print_result("prompt: persona_roleplay", found)
                if not found:
                    failures += 1
            except Exception as e:
                _print_result("list_prompts", False, str(e))
                failures += 1

            # ── Tool: search_personas ────────────────────────────
            print("\n[5] Tool 호출: search_personas")
            try:
                resp = await session.call_tool("search_personas", {
                    "query": "광주 70대 하역 노동자",
                    "top_k": 3,
                })
                raw = resp.content[0].text if resp.content else ""
                data = json.loads(raw)
                ok = data.get("count", 0) > 0
                detail = f"count={data.get('count')} | 첫 결과: {data['results'][0]['profile'] if data.get('results') else 'N/A'}"
                _print_result("search_personas (기본 검색)", ok, detail)
                if not ok:
                    failures += 1

                # 필터 포함
                resp2 = await session.call_tool("search_personas", {
                    "query": "전문직 여행 좋아하는 사람",
                    "province": "서울",
                    "top_k": 2,
                })
                raw2 = resp2.content[0].text if resp2.content else ""
                data2 = json.loads(raw2)
                ok2 = "results" in data2
                _print_result(
                    f"search_personas (province=서울 필터)",
                    ok2,
                    f"count={data2.get('count')}",
                )
                if not ok2:
                    failures += 1

                # 첫 결과의 uuid 저장 (이후 테스트에 사용)
                first_uuid = (
                    data["results"][0]["id"]
                    if data.get("results") and data["results"][0].get("id")
                    else None
                )
            except Exception as e:
                _print_result("search_personas", False, str(e))
                failures += 1
                first_uuid = None

            # ── Tool: get_persona_by_id ──────────────────────────
            print("\n[6] Tool 호출: get_persona_by_id")
            if first_uuid:
                try:
                    resp = await session.call_tool("get_persona_by_id", {"uuid": first_uuid})
                    raw = resp.content[0].text if resp.content else ""
                    data = json.loads(raw)
                    ok = "profile" in data and "error" not in data
                    _print_result(
                        f"get_persona_by_id (uuid={first_uuid[:8]}...)",
                        ok,
                        data.get("profile", data.get("error", "")),
                    )
                    if not ok:
                        failures += 1
                except Exception as e:
                    _print_result("get_persona_by_id", False, str(e))
                    failures += 1

                # 존재하지 않는 ID
                try:
                    resp = await session.call_tool("get_persona_by_id", {"uuid": "nonexistent-id"})
                    raw = resp.content[0].text if resp.content else ""
                    data = json.loads(raw)
                    ok = "error" in data
                    _print_result("get_persona_by_id (없는 ID → error 반환)", ok)
                    if not ok:
                        failures += 1
                except Exception as e:
                    _print_result("get_persona_by_id (없는 ID)", False, str(e))
                    failures += 1
            else:
                _print_result("get_persona_by_id (skip — uuid 없음)", False)
                failures += 1

            # ── Tool: get_demographic_stats ──────────────────────
            print("\n[7] Tool 호출: get_demographic_stats")
            try:
                resp = await session.call_tool("get_demographic_stats", {})
                raw = resp.content[0].text if resp.content else ""
                data = json.loads(raw)
                ok = "matched" in data and data["matched"] > 0
                _print_result(
                    "get_demographic_stats (전체)",
                    ok,
                    f"matched={data.get('matched')} | sex={data.get('sex')}",
                )
                if not ok:
                    failures += 1

                resp2 = await session.call_tool("get_demographic_stats", {
                    "province": "서울",
                    "age_group": "청년",
                })
                raw2 = resp2.content[0].text if resp2.content else ""
                data2 = json.loads(raw2)
                ok2 = "matched" in data2
                _print_result(
                    "get_demographic_stats (서울 + 청년)",
                    ok2,
                    f"matched={data2.get('matched')} | age={data2.get('age')}",
                )
                if not ok2:
                    failures += 1
            except Exception as e:
                _print_result("get_demographic_stats", False, str(e))
                failures += 1

            # ── Resource: persona://{uuid} ───────────────────────
            print("\n[8] Resource 조회: persona://{uuid}")
            if first_uuid:
                try:
                    resp = await session.read_resource(f"persona://{first_uuid}")
                    raw = resp.contents[0].text if resp.contents else ""
                    data = json.loads(raw)
                    ok = "profile" in data
                    _print_result(f"persona://{first_uuid[:8]}...", ok, data.get("profile", ""))
                    if not ok:
                        failures += 1
                except Exception as e:
                    _print_result("resource persona://", False, str(e))
                    failures += 1
            else:
                _print_result("resource persona:// (skip)", False)
                failures += 1

            # ── Prompt: persona_roleplay ─────────────────────────
            print("\n[9] Prompt 호출: persona_roleplay")
            try:
                resp = await session.get_prompt("persona_roleplay", {
                    "query": "광주 70대 남성 노동자",
                })
                messages = resp.messages
                ok = len(messages) > 0
                first_msg = messages[0].content.text if messages else ""
                _print_result(
                    "persona_roleplay",
                    ok,
                    first_msg[:200] if first_msg else "(내용 없음)",
                )
                if not ok:
                    failures += 1
            except Exception as e:
                _print_result("persona_roleplay", False, str(e))
                failures += 1

    return failures


def main() -> None:
    print("=" * 60)
    print("Nemotron-Personas-Korea MCP 서버 테스트")
    print("=" * 60)

    failures = asyncio.run(run_tests())

    print("\n" + "=" * 60)
    if failures == 0:
        print(f"결과: 전체 통과 {_PASS}")
    else:
        print(f"결과: {failures}개 실패 {_FAIL}")
    print("=" * 60)
    sys.exit(failures)


if __name__ == "__main__":
    main()
