"""TRL 평가 노드 — 담당: R4

조립은 `test_r4_evaluate.py` 가 본다. 여기서는 maturity 만의 검색 방식 — 기술 근거가 하나도
없으면 기술별 질의로 1회 보완 — 을 본다.
"""

from types import SimpleNamespace

from src.agents import maturity as agent
from tests.test_r4_evaluate import ITME_ID, TQ_ID, chunk, drafts


def test_supplemental_search_runs_once_when_a_technology_is_entirely_unretrieved(monkeypatch):
    tq = chunk(TQ_ID, applies_to=["TurboQuant"])
    itme = chunk(ITME_ID, applies_to=["ITME"])
    # 기본 2회 질의 동안 ITME 는 빈 결과, 보완 검색(1회)에서만 회수된다
    queue = {"TurboQuant": [[tq], []], "ITME": [[], [], [itme]]}
    calls = {"TurboQuant": 0, "ITME": 0}

    def invoke(payload):
        tech = payload["technology"]
        calls[tech] += 1
        return list(queue[tech][calls[tech] - 1]) if calls[tech] <= len(queue[tech]) else []

    monkeypatch.setattr(agent, "bind_document_search", lambda *a, **k: SimpleNamespace(invoke=invoke))
    drafts(monkeypatch, {"turboquant": f"TRL 4 [{TQ_ID}].", "itme": f"TRL 3 [{ITME_ID}].", "gaps": []})

    out = agent.maturity({"run_id": "r4-test", "run_config": {}})

    assert out["trace"][0]["supplemental_searches"] == 1
    assert out["maturity"]["status"] == "completed"
