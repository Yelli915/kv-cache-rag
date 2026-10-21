"""그래프 경로 — 담당: R1 (설계서 §8, 부록 A)

전 노드를 mock 으로 두고 그래프가 끝까지 도는지, 경로마다 run_status·trace·
runs/<run_id>/ 가 남는지 본다. 실제 LLM·검색은 타지 않는다.
"""

import json

import pytest

from src.graph import MergeConflict, build_graph, invoke, resume_state, setup as real_setup
from tests import mock_nodes

CONFIG = {"domain": "데이터센터/클라우드", "technologies": ["TurboQuant", "ITME"]}


@pytest.fixture
def run(tmp_path):
    """mock 노드로 그래프를 돌리고 최종 State 와 실행 디렉토리를 돌려준다."""
    def go(*, start="setup", state=None, **mock_kw):
        initial = state or {"run_id": "r1-test", "trace": [],
                            "run_config": CONFIG | {"runs_dir": str(tmp_path)}}
        final = invoke(build_graph(start=start, **mock_nodes.all_nodes(**mock_kw)), initial)
        return final, tmp_path / initial["run_id"]
    return go


def test_completed_path(run):
    final, directory = run()
    assert final["run_status"] == "completed"
    assert final["report"].startswith("# 합성 보고서")
    assert final["report_paths"] == [str(directory / "report.md")]
    assert {"setup", "research", "maturity", "market", "stakeholder", "domain_assessment",
            "collect_evidence", "synthesis", "review", "final_check", "report",
            "publish"} == {t["node"] for t in final["trace"]}
    assert directory.is_dir()


def test_five_assessments_merge_into_two_registries(run):
    """기술 조사 근거도 병합한다 — 참고문헌을 실제 인용에서 역으로 만들어야 한다(§9)."""
    final, _ = run()
    assert list(final["source_registry"]) == ["web-mock-1"]
    assert list(final["evidence_registry"]) == ["e-web-mock-1"]
    assert [g["role"] for g in final["gaps"]] == ["synthesis"]   # 종합 후 순차 병합


def test_gaps_merge_sequentially(run):
    final, _ = run(statuses={"market": "partial"})
    assert [g["role"] for g in final["gaps"]] == ["market", "synthesis"]


def test_merge_conflict_fails_the_run(run, tmp_path):
    """같은 ID 에 다른 내용이 오면 종합·검토로 넘기지 않는다(§7)."""
    with pytest.raises(MergeConflict, match="e-web-mock-1"):
        run(market={"quote": "다른 인용문"})
    errors = json.loads((tmp_path / "r1-test" / "merge-errors.json").read_text())["errors"]
    assert errors and "market" in errors[0]


def test_review_pending_saves_draft_and_stops(run):
    """검토 대기면 초안만 저장하고 멈춘다. 사람 검토 없이 review 로 되돌아가지 않는다."""
    final, directory = run(review_status="pending")
    assert final["run_status"] == "partial"
    assert "report" not in final
    draft = json.loads((directory / "draft.json").read_text())
    assert draft["source_registry"] and draft["review_status"] == "pending"


def test_resume_from_reviewed_draft(run, tmp_path):
    """부록 A 의 `검토 결과 반영 후 재개` — 내용 검토부터 이어 간다."""
    _, directory = run(review_status="pending")
    draft = resume_state("r1-test", tmp_path)
    assert draft["review_status"] == "pending"

    final, _ = run(start="review", state=draft)
    assert final["run_status"] == "completed" and final["report"]
    assert [t["node"] for t in final["trace"]] == ["review", "final_check", "report", "publish"]


def test_unresolved_errors_block_submission(run):
    final, directory = run(review_errors=["없는 인용 ID"])
    assert final["run_status"] == "failed"
    assert "report" not in final
    assert json.loads((directory / "validation-errors.json").read_text())["errors"]


def test_failed_assessment_fails_the_run(run):
    final, _ = run(statuses={"stakeholder": "failed"})
    assert final["run_status"] == "failed"


def test_partial_assessment_still_reports(run):
    """근거 공백은 실패가 아니다. 보고서는 내고 run_status 로 표시한다(§7)."""
    final, _ = run(statuses={"market": "partial"})
    assert final["run_status"] == "partial" and final["report"]


def test_setup_requires_run_id_domain_and_technologies(tmp_path):
    graph = build_graph(**mock_nodes.all_nodes() | {"setup": real_setup})
    base = {"trace": [], "run_config": CONFIG | {"runs_dir": str(tmp_path)}}
    with pytest.raises(ValueError, match="run_id"):
        invoke(graph, base | {"run_id": ""})
    with pytest.raises(ValueError, match="technologies"):
        invoke(graph, base | {"run_id": "r1-test",
                              "run_config": {"domain": "x", "runs_dir": str(tmp_path)}})


def test_smoke_script_reproduces_every_path(tmp_path):
    """증빙 스크립트가 CI 에서도 돌아야 한다 — 증빙과 테스트가 갈라지지 않게."""
    from scripts.r1_smoke import run

    cases = run(tmp_path / "smoke")
    assert [c["case"] for c in cases] == ["completed", "review_pending", "resume_after_review",
                                          "unresolved_errors", "merge_conflict"]
    assert all(c["passed"] for c in cases)


def test_corpus_hash_mismatch_stops_the_run(tmp_path):
    """설계서 §3 — 적재할 때 SHA-256 을 다시 확인한다. 조용히 넘어가면 안 된다."""
    import hashlib

    from src.graph import verify_corpus

    original = tmp_path / "paper.pdf"
    original.write_bytes(b"original")
    entry = {"id": "itme-paper", "local_path": str(original),
             "sha256": hashlib.sha256(b"original").hexdigest()}

    assert verify_corpus({"sources": [entry]}) == []

    original.write_bytes(b"upstream changed the file")
    assert "SHA-256 불일치" in verify_corpus({"sources": [entry]})[0]

    original.unlink()
    assert "원문이 없다" in verify_corpus({"sources": [entry]})[0]


def test_llm_cache_counts_hits_and_misses(tmp_path, monkeypatch):
    """재현성은 모델이 아니라 캐시가 담당한다. 적중 수를 세지 못하면 비용 기록이 거짓이 된다."""
    from langchain_core.outputs import Generation

    from src import llm

    monkeypatch.setattr(llm, "_cache", None)
    cache = llm.enable_cache(str(tmp_path / "llm.sqlite"))

    assert cache.lookup("같은 프롬프트", "모델") is None        # miss
    cache.update("같은 프롬프트", "모델", [Generation(text="응답")])
    assert cache.lookup("같은 프롬프트", "모델")[0].text == "응답"   # hit

    assert (cache.hits, cache.misses) == (1, 1)
    assert llm.llm_report() | {"cache_hits": 1, "cache_misses": 1} == llm.llm_report()


def test_cache_hit_counts_zero_tokens(tmp_path, monkeypatch):
    """저장된 응답은 원래 호출의 토큰 수를 들고 있다. 재사용분을 이번 실행 비용으로 세면 안 된다."""
    from langchain_core.callbacks import get_usage_metadata_callback
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from src import llm

    monkeypatch.setattr(llm, "_cache", None)
    llm.enable_cache(str(tmp_path / "llm.sqlite"))
    reply = lambda: AIMessage("응답", response_metadata={"model_name": "m"},
                              usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    model = GenericFakeChatModel(messages=iter([reply(), reply()]))

    with get_usage_metadata_callback() as usage:
        model.invoke("같은 프롬프트")      # miss — 실제 호출
        model.invoke("같은 프롬프트")      # hit — 호출 없음

    assert usage.usage_metadata["m"]["total_tokens"] == 15


def test_run_records_token_usage_and_call_counts():
    """§6 — 실제 비용은 토큰 사용량과 실행 기록으로 확인한다."""
    from types import SimpleNamespace

    import app

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
    import app

    record = app._accounting({"trace": [{"node": "market", "attempt": 1}]}, None)
    assert record["node_events"] == {"market": 1} and "total_tokens" not in record


def test_two_column_split_uses_the_page_bbox():
    """CropBox 가 0 에서 시작하지 않는 PDF — 좌표를 페이지 bbox 기준으로 자른다."""
    from src.rag import index

    crops = []

    class Page:
        bbox, width = (100, 0, 700, 800), 600

        def extract_words(self, **_):          # 가운데(400)를 가로지르는 단어가 없다 → 2단
            return [{"x0": 120, "x1": 300}, {"x0": 450, "x1": 650}]

        def crop(self, box):
            crops.append(box)
            return type("Half", (), {"extract_text": lambda self, **_: "half"})()

    assert index._page_text(Page()) == "half\nhalf"
    assert crops == [(100, 0, 400.0, 800), (400.0, 0, 700, 800)]


def test_page_cache_reuses_extraction_until_the_source_changes(tmp_path, monkeypatch):
    """PDF 추출 캐시 — 같은 원문이면 다시 파싱하지 않고, 해시가 바뀌면 다시 파싱한다."""
    from src.rag import index

    calls = []
    monkeypatch.setattr(index, "PAGE_CACHE", tmp_path)
    monkeypatch.setattr(index, "_extract_pdf", lambda path: calls.append(path) or [(1, "본문", [["a", "b"]])])
    entry = {"id": "paper", "sha256": "aaa", "local_path": "paper.pdf"}

    assert index._cached_pdf(entry) == [(1, "본문", [["a", "b"]])]
    assert index._cached_pdf(entry) == [(1, "본문", [["a", "b"]])] and len(calls) == 1
    index._cached_pdf(entry | {"sha256": "bbb"})
    assert len(calls) == 2


def test_vector_cache_reuses_embeddings_until_a_chunk_changes(tmp_path, monkeypatch):
    """청크 벡터 캐시 — chunk_id(본문 해시)가 하나라도 바뀌면 다시 임베딩한다."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from src.rag import index

    class Fake(Embeddings):
        calls = 0

        def embed_documents(self, texts):
            Fake.calls += 1
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    monkeypatch.setattr(index, "VECTOR_CACHE", tmp_path)
    docs = [Document(page_content="a", metadata={"chunk_id": "c1"})]

    index._faiss(docs, Fake())
    store = index._faiss(docs, Fake())
    assert Fake.calls == 1 and store.index.ntotal == 1
    index._faiss([Document(page_content="b", metadata={"chunk_id": "c2"})], Fake())
    assert Fake.calls == 2
