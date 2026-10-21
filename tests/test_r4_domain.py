"""도메인 적용 평가 노드 — 담당: R4

조립·결합 가설은 `test_r4_evaluate.py` 가 본다. 여기서는 domain 만의 검색 방식 — papers_core
실증 근거와 context 배경 자료를 둘 다 조회 — 을 본다.
"""

from types import SimpleNamespace

from src.agents import domain as agent
from tests.test_r4_evaluate import drafts


def test_both_collections_are_queried(monkeypatch):
    calls = []

    def bind(role, collection=None):
        target = "context" if collection == "context" else "core"
        return SimpleNamespace(invoke=lambda p: calls.append((target, p["technology"])) or [])

    monkeypatch.setattr(agent, "bind_document_search", bind)
    drafts(monkeypatch, {"turboquant": "근거 없음.", "itme": "근거 없음.", "gaps": []})

    agent.domain_assessment({"run_id": "r4-test", "run_config": {}})

    assert {("core", "TurboQuant"), ("core", "ITME"), ("context", "both")} <= set(calls)
