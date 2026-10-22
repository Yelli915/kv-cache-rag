"""실행 진입점과 비용 기록 — 담당: R1 (`app.py`, `src/llm.py`)

run.json 의 토큰·캐시 적중 집계, 재개 시 구간 보존, 종료 상태를 본다. 그래프는 mock 이다.
"""

import json
import os
from types import SimpleNamespace

import pytest
from langchain_core.globals import set_llm_cache

import app
from src import llm
from src.graph import MergeConflict


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """tmp sqlite 캐시. langchain 전역 캐시는 monkeypatch 가 모르므로 직접 되돌린다 —
    안 그러면 지워진 tmp 파일을 가리키는 캐시가 다음 테스트로 샌다."""
    monkeypatch.setattr(llm, "_cache", None)
    yield llm.enable_cache(str(tmp_path / "llm.sqlite"))
    set_llm_cache(None)


def test_llm_cache_counts_hits_and_misses(cache):
    """재현성은 모델이 아니라 캐시가 담당한다. 적중 수를 세지 못하면 비용 기록이 거짓이 된다."""
    from langchain_core.outputs import Generation

    assert cache.lookup("같은 프롬프트", "모델") is None        # miss
    cache.update("같은 프롬프트", "모델", [Generation(text="응답")])
    assert cache.lookup("같은 프롬프트", "모델")[0].text == "응답"   # hit

    assert (cache.hits, cache.misses) == (1, 1)
    assert llm.llm_report() | {"cache_hits": 1, "cache_misses": 1} == llm.llm_report()


def test_cache_hit_counts_zero_tokens(cache):
    """저장된 응답은 원래 호출의 토큰 수를 들고 있다. 재사용분을 이번 실행 비용으로 세면 안 된다."""
    from langchain_core.callbacks import get_usage_metadata_callback
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    reply = lambda: AIMessage("응답", response_metadata={"model_name": "m"},
                              usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    model = GenericFakeChatModel(messages=iter([reply(), reply()]))

    with get_usage_metadata_callback() as usage:
        model.invoke("같은 프롬프트")      # miss — 실제 호출
        model.invoke("같은 프롬프트")      # hit — 호출 없음

    assert usage.usage_metadata["m"]["total_tokens"] == 15


def test_run_records_token_usage_and_call_counts():
    """§6 — 실제 비용은 토큰 사용량과 실행 기록으로 확인한다."""
    state = {"trace": [
        {"node": "research", "status": "ok", "attempt": 1},
        {"node": "stakeholder", "action": "draft_validation", "status": "failed", "attempt": 1},
        {"node": "stakeholder", "action": "draft_validation", "status": "ok", "attempt": 2},
        {"tool": "search_market_signals", "node": "stakeholder", "attempt": 1},
        {"node": "synthesis", "status": "ok", "attempt": 1},
        {"node": "report", "status": "ok", "attempt": 1},
    ]}
    usage = SimpleNamespace(usage_metadata={
        "gemini-2.5-flash": {"input_tokens": 41000, "output_tokens": 7000, "total_tokens": 48000},
        "gemini-2.5-pro": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    })

    record = app._accounting(state, usage)

    assert record["node_events"] == {"research": 1, "stakeholder": 2, "synthesis": 1}
    assert record["tool_calls"] == 1          # tool 이 붙은 이벤트만 센다
    assert record["retries"] == 1             # attempt > 1
    assert record["total_tokens"] == 48120 and record["input_tokens"] == 41100
    assert "report" not in record["node_events"]   # 생성 단계가 아니다


def test_accounting_without_a_callback_still_records_the_trace():
    record = app._accounting({"trace": [{"node": "market", "attempt": 1}]}, None)
    assert record["node_events"] == {"market": 1} and "total_tokens" not in record


def test_resume_keeps_the_first_leg_in_run_json(tmp_path):
    """재개가 run.json 을 덮어쓰면 검토 대기 전까지 쓴 토큰 기록이 사라진다(§6)."""
    import json

    state = {"run_id": "r1-resume", "run_config": {"runs_dir": str(tmp_path)}}
    path = app.start_run(state) / "run.json"
    first = json.loads(path.read_text()) | {"run_status": "partial", "llm": {"total_tokens": 900}}
    path.write_text(json.dumps(first))

    app.start_run(state)      # 재개
    app.start_run(state)      # 한 번 더 재개

    record = json.loads(path.read_text())
    assert record["run_status"] == "running"
    assert [leg["run_status"] for leg in record["legs"]] == ["partial", "running"]
    assert record["legs"][0]["llm"] == {"total_tokens": 900}


def test_importing_app_leaves_the_working_directory_alone():
    """chdir 은 main() 안에서만 한다. import 가 cwd 를 옮기면 상대경로 테스트가 실행 순서에 묶인다."""
    import importlib

    before = os.getcwd()
    importlib.reload(app)
    assert os.getcwd() == before


def test_merge_conflict_is_recorded_as_a_failed_run(tmp_path, monkeypatch):
    """병합 오류는 예외 대신 failed 로 끝난다 — 종료 코드 1 과 run.json errors 가 그 기록이다."""
    def conflict(graph, state):
        raise MergeConflict("market: e-1 가 기존 내용과 다르다")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "_ROOT", tmp_path)
    monkeypatch.setattr(app, "load_dotenv", lambda **k: None)
    monkeypatch.setattr(app, "enable_cache", lambda: None)
    monkeypatch.setattr(app, "settings", lambda: {"run": {"runs_dir": "runs"}, "llm": {}, "limits": {}})
    monkeypatch.setattr(app, "build_graph", lambda **k: None)
    monkeypatch.setattr(app, "invoke", conflict)
    monkeypatch.setattr("sys.argv", ["app.py", "--run-id", "r1-conflict"])

    assert app.main() == "failed"
    record = json.loads((tmp_path / "runs" / "r1-conflict" / "run.json").read_text())
    assert record["run_status"] == "failed" and "e-1" in record["errors"][0]
