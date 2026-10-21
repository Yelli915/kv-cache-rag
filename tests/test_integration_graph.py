"""실물 노드로 그래프를 끝까지 돌린다 — 담당: R1

`tests/test_r1_graph.py` 는 mock 노드로 **배선**을 본다. 이쪽은 `DEFAULT_NODES` 의
실물을 그대로 쓰고 검색·LLM·색인만 fixture 로 막아 **계약이 맞물리는지** 본다.
R4 의 Assessment → R1 의 병합 → R5 의 종합·검토·보고서·제출본이 한 줄로 이어져야 한다.

`stakeholder` 만 stub 이다. 네트워크와 실행별 저장소를 쓰는 노드라 R3 의
`scripts/r3_smoke.py` 가 따로 검증한다.
"""

import hashlib
import json

import pytest

from src.agents import common, domain, market, maturity, research
from src.graph import build_graph, invoke
from src.output import pdf
from src.state import ASSESSMENT_NODES
from src.schema import Assessment, Synthesis
from tests.helpers import PAPERS, fill_review, search

ECOSYSTEM = [
    {"chunk_id": "cccccccccccc", "source_id": "cxl-report", "collection": "ecosystem",
     "page": 2, "text": "CXL memory modules shipped by multiple vendors.",
     "applies_to": ["ITME"], "scope": "ecosystem", "cosine_score": 0.5},
    {"chunk_id": "dddddddddddd", "source_id": "quant-report", "collection": "ecosystem",
     "page": 4, "text": "Serving frameworks announce KV cache quantization support.",
     "applies_to": ["TurboQuant"], "scope": "ecosystem", "cosine_score": 0.5},
]

PAPERS_DRAFT = {
    "turboquant": "TurboQuant 는 3.5비트 양자화를 보고한다 [aaaaaaaaaaaa].",
    "itme": "ITME 는 NVMe-oF 대비 1.80배 처리량을 보고한다 [bbbbbbbbbbbb].",
    "gaps": [{"technology": "both", "item": "결합 실측 자료 없음"}],
}
MARKET_TEXT = (
    "## ① 시장 규모/성장성\n정량 리포트 미확인 [dddddddddddd].\n\n"
    "## ② 상용화/채택 현황\n발표 단계, 배포 확인 미확인 [cccccccccccc].\n\n"
    "## ③ 생태계 지지\n일반 CXL 근거이며 추론 특화 여부 미확인 [cccccccccccc].\n\n"
    "근거 공백: TurboQuant 운영 배포 사례 미확인"
)


def stakeholder_stub(state):
    """R3 노드 자리. 형식만 같은 합성 Assessment 를 돌려준다."""
    run_id = state["run_id"]
    return {
        "stakeholder": {
            "status": "partial",
            "claims": [{"claim_id": "claim-stakeholder", "text": "개발자 통합 논의 확인",
                        "technology": "TurboQuant", "kind": "fact",
                        "evidence_ids": ["e-web-1"]}],
            "evidence": [{"evidence_id": "e-web-1", "source_id": "web-1", "run_id": run_id,
                          "collection": "web", "quote": "합성 인터뷰", "location": "paragraph:1",
                          "allowed_uses": ["stakeholder"]}],
            "sources": [{"source_id": "web-1", "run_id": run_id, "collection": "web",
                         "allowed_uses": ["stakeholder"], "title": "합성 출처",
                         "url": "https://example.org/synthetic", "published_at": None,
                         "retrieved_at": "2026-09-22"}],
            "gaps": [{"role": "stakeholder", "technology": "ITME", "item": "investors",
                      "reason": "원문 본문 근거 미확보"}],
        },
        "trace": [{"node": "stakeholder", "status": "partial"}],
    }


class FakeSynthesis:
    """synthesis 의 구조화 출력 LLM 자리."""

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return Synthesis(
            agreements=["두 기술 모두 KV cache 메모리 병목을 완화한다."],
            conflicts=[{"perspective": "TRL", "why": "실증 환경의 범위가 다르다"}],
            gaps=["두 기술의 결합 실측 자료 미확인"],
            combination_hypothesis="개별 근거에서 도출한 추론이며 실측 결과가 아니다.",
        )


@pytest.fixture
def wired(monkeypatch, tmp_path):
    for module, chunks in ((research, PAPERS), (maturity, PAPERS), (domain, PAPERS),
                           (market, ECOSYSTEM)):
        monkeypatch.setattr(module, "bind_document_search",
                            lambda *a, _chunks=chunks, **k: search(_chunks))
    monkeypatch.setattr(market, "run_node", lambda *a, **k: MARKET_TEXT)
    monkeypatch.setattr(common, "draft",
                        lambda *a, schema=common.Draft, **k: schema.model_validate(PAPERS_DRAFT))

    from src.agents import report as report_module
    from src.agents import synthesis as synthesis_module

    monkeypatch.setattr(synthesis_module, "get_llm", lambda: FakeSynthesis())
    monkeypatch.setattr(report_module, "model_name", lambda: "test-model")
    monkeypatch.setattr("src.rag.index.build", lambda: {"manifest": [{"id": "turboquant-paper"}]})
    # 원문(data/raw/)은 저작권 문제로 커밋하지 않는다. 대신 tmp 원문과 해시가 맞는 매니페스트를
    # 넣어 setup → verify_corpus 연결을 실물로 탄다.
    raw = tmp_path / "raw.txt"
    raw.write_bytes(b"fixture source")
    manifest = {"sources": [{"id": "turboquant-paper", "local_path": str(raw),
                             "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}]}
    monkeypatch.setattr("src.rag.index.load_manifest", lambda: manifest)
    monkeypatch.setattr(pdf, "render_pdf",
                        lambda md, out: (out.parent.mkdir(parents=True, exist_ok=True),
                                         out.write_bytes(b"%PDF fixture")))

    def run(state=None, **overrides):
        initial = state or {
            "run_id": "e2e-test", "run_status": "running", "trace": [],
            "run_config": {"domain": "데이터센터/클라우드", "technologies": ["TurboQuant", "ITME"],
                           "runs_dir": str(tmp_path), "started_at": "2026-09-22T00:00:00+00:00"},
        }
        graph = build_graph(stakeholder=stakeholder_stub, **overrides)
        return invoke(graph, initial), tmp_path / initial["run_id"]

    return run


def test_setup_stops_on_corpus_hash_mismatch(wired, tmp_path):
    """원문이 매니페스트와 다르면 setup 에서 실행이 멈춘다(§3)."""
    (tmp_path / "raw.txt").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="매니페스트와 다르다"):
        wired()


def test_pipeline_stops_for_human_review(wired):
    """첫 실행은 검토 대기다. §8 — ID 대조만으로 자동 통과시키지 않는다."""
    final, directory = wired()

    assert final["run_status"] == "partial"
    assert final["review_status"] == "pending"
    assert "report" not in final
    assert (directory / "review.csv").exists()
    assert (directory / "draft.json").exists()


def test_every_assessment_merges_into_the_registries(wired):
    final, _ = wired()

    for node in ASSESSMENT_NODES:
        Assessment.model_validate(final[node])

    # 합류 단계가 다섯 Assessment 의 출처를 모두 모은다 (§6·§9)
    assert {"turboquant-paper", "itme-paper", "cxl-report", "quant-report",
            "web-1"} <= set(final["source_registry"])
    assert final["evidence_registry"] and final["gaps"]


def test_gaps_are_merged_sequentially(wired):
    """§7 — 합류 후 수집, 종합 후 순차 병합.

    fixture 의 종합 공백("두 기술의 결합 실측 자료 미확인")은 노드가 이미 보고한
    "결합 실측 자료 없음" 을 되풀이한 것이라 걸러진다. §6 은 종합에게 공백을 *정리* 하라고
    한다 — 같은 내용을 말만 바꿔 다시 싣지 않는다.
    """
    final, _ = wired()
    roles = {gap["role"] for gap in final["gaps"]}
    items = [gap["item"] for gap in final["gaps"]]

    assert roles & {"research", "maturity", "market", "stakeholder", "domain"}
    assert "결합 실측 자료 없음" in items
    assert "두 기술의 결합 실측 자료 미확인" not in items


def test_resume_after_review_produces_the_submission(wired, tmp_path):
    """검토 판정을 채우고 재개하면 보고서와 제출본까지 나온다 (부록 A).

    평가 보류 항목이 남아 있으므로 run_status 는 partial 이다. §7 — 부분 결과는 일부
    항목을 평가 보류로 남겼으나 포함된 주장의 검증은 통과한 경우다. 보고서는 낸다.
    """
    _, directory = wired()
    fill_review(directory / "review.csv", "확인")

    draft = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
    final, _ = wired(state=draft | {"trace": []}, start="review")

    assert final["run_status"] == "partial"     # 근거 공백이 남았다. 실패가 아니다.
    assert final["review_status"] == "passed"
    assert "## REFERENCE" in final["report"]
    assert final["report_paths"][0].endswith("report.md")
    assert (directory / "final").is_dir()
    assert json.loads((directory / "submission.json").read_text())["generated"] is True


def test_rejected_claim_never_reaches_the_report(wired, tmp_path):
    """§8 — 부결된 주장은 사실 서술에서 제외한다."""
    _, directory = wired()
    fill_review(directory / "review.csv", lambda row: "부결" if row["node"] == "market" else "확인")

    draft = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
    final, _ = wired(state=draft | {"trace": []}, start="review")

    rejected = final["validation"]["rejected_claims"]
    assert rejected

    body = final["report"][:final["report"].index("## 5. 시사점")]
    assert "발표 단계, 배포 확인 미확인" not in body      # 부결된 시장성 주장이 본문에 없다
    limits = final["report"][final["report"].index("## 6. 한계"):]
    assert all(f"검토 부결: {claim_id}" in limits for claim_id in rejected)
    assert "cxl-report" not in final["report"]           # 그 주장의 출처도 REFERENCE 에서 빠진다
