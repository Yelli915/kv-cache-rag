"""기술 조사 노드 — 담당: R4

조립(인용 사슬·status·수치 재작성)은 `test_r4_evaluate.py` 가 본다. 여기서는 research 만의
검색 방식 — direct 근거가 없는 기술이 있으면 WIDEN 질의로 1회 보완 — 을 본다.
"""

from types import SimpleNamespace

from src.agents import research as agent
from tests.test_r4_evaluate import ITME_ID, TQ_ID, chunk, drafts


def test_supplemental_search_runs_once_when_a_technology_is_entirely_unretrieved(monkeypatch):
    tq = chunk(TQ_ID, applies_to=["TurboQuant"])
    itme = chunk(ITME_ID, applies_to=["ITME"])
    # ITME 는 기본 질의 3회 동안 빈 결과, 보완 검색에서만 회수된다
    queue = {"TurboQuant": [[tq], [], []], "ITME": [[], [], [], [itme]]}
    calls = {"TurboQuant": 0, "ITME": 0}

    def invoke(payload):
        tech = payload["technology"]
        calls[tech] += 1
        return list(queue[tech][calls[tech] - 1]) if calls[tech] <= len(queue[tech]) else []

    monkeypatch.setattr(agent, "bind_document_search", lambda *a, **k: SimpleNamespace(invoke=invoke))
    drafts(monkeypatch, {"turboquant": f"압축한다 [{TQ_ID}].", "itme": f"가속한다 [{ITME_ID}].", "gaps": []})

    out = agent.research({"run_id": "r4-test", "run_config": {}})

    assert out["trace"][0]["supplemental_searches"] == 3
    assert out["research"]["status"] == "completed"
