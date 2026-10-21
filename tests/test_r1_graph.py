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


def test_merge_conflict_fails_the_run(run):
    """같은 ID 에 다른 내용이 오면 종합·검토로 넘기지 않는다(§7)."""
    with pytest.raises(MergeConflict, match="market: e-web-mock-1"):
        run(market={"quote": "다른 인용문"})


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
    final, _ = run(review_errors=["없는 인용 ID"])
    assert final["run_status"] == "failed"
    assert "report" not in final
    assert final["trace"][-1]["node"] == "fail" and final["trace"][-1]["errors"]


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
