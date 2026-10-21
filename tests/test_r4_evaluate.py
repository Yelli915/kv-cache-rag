"""평가 노드 공통 조립 — `common.evaluate` (검증: 인터페이스 계약 ① · 설계서 §6 · 부록 A)

research · maturity · domain 이 같은 조립을 쓴다. 노드별 파일은 검색 방식만 보고,
인용 사슬·status 판정·수치 재작성은 여기서 한 번 본다. LLM(`common.draft`)과 색인은 가짜다.
"""

from types import SimpleNamespace

from src.agents import common
from src.agents.common import Draft, DomainDraft, evaluate
from src.schema import Assessment

TQ_ID = "aaaaaaaaaaaa"
ITME_ID = "bbbbbbbbbbbb"
UNSEEN_ID = "ffffffffffff"


def chunk(cid, *, applies_to, scope="direct", collection="papers_core", source_id="paper-1",
          page=7, text="본문 내용"):
    return {"chunk_id": cid, "source_id": source_id, "source": source_id, "collection": collection,
            "applies_to": list(applies_to), "scope": scope, "page": page, "text": text}


def make_bind(chunks_by_tech):
    """technology 로 고정 청크를 돌려주는 가짜 bind_document_search. query 는 무시한다."""
    def bind(role, collection=None):
        return SimpleNamespace(invoke=lambda p: list(chunks_by_tech.get(p["technology"], [])))
    return bind


def drafts(monkeypatch, *outputs):
    """`common.draft` 를 호출 순서대로 outputs 를 내는 가짜로 바꾸고 받은 피드백을 모은다."""
    calls = []

    def fake(instruction, domain, context, schema=Draft, feedback=""):
        calls.append(feedback)
        return schema.model_validate(outputs[min(len(calls), len(outputs)) - 1])

    monkeypatch.setattr(common, "draft", fake)
    return calls


def run(chunks, **overrides):
    kwargs = dict(node="research", role="research", run_id="r4-test", domain="d",
                  instruction="i", chunks=chunks, gap_reason="미확인")
    return evaluate(**(kwargs | overrides))


TQ, ITME = chunk(TQ_ID, applies_to=["TurboQuant"]), chunk(ITME_ID, applies_to=["ITME"])


def test_sections_become_one_claim_per_technology_and_chain_closes(monkeypatch):
    uncited = chunk("dddddddddddd", applies_to=["TurboQuant"])
    drafts(monkeypatch, {"turboquant": f"SW 계층에서 압축한다 [{TQ_ID}].",
                         "itme": f"HW 계층에서 가속한다 [{ITME_ID}].", "gaps": []})

    out = run([TQ, uncited, ITME])
    a = Assessment.model_validate(out["research"])

    assert a.status == "completed" and a.gaps == []
    assert [(c.technology, c.evidence_ids) for c in a.claims] == [
        ("TurboQuant", [TQ_ID]), ("ITME", [ITME_ID])]
    # 검색됐지만 본문이 인용하지 않은 청크는 등록하지 않는다
    assert {e.evidence_id for e in a.evidence} == {TQ_ID, ITME_ID}
    assert out["trace"][0]["attempt"] == 1


def test_evidence_quote_keeps_the_whole_chunk(monkeypatch):
    """quote 를 자르면 수치 검증과 사람의 검토가 원문 일부만 본다."""
    long = chunk(TQ_ID, applies_to=["TurboQuant"], text="x" * 1000 + " 3.5 bits")
    drafts(monkeypatch, {"turboquant": f"3.5비트 [{TQ_ID}].", "itme": f"가속 [{ITME_ID}].", "gaps": []})

    evidence = run([long, ITME])["research"]["evidence"]

    assert next(e for e in evidence if e["evidence_id"] == TQ_ID)["quote"].endswith("3.5 bits")


def test_structured_gaps_keep_their_technology(monkeypatch):
    drafts(monkeypatch, {"turboquant": f"압축 [{TQ_ID}].", "itme": f"가속 [{ITME_ID}].",
                         "gaps": [{"technology": "ITME", "item": "운영 환경 실증"}]})

    gaps = run([TQ, ITME])["research"]["gaps"]

    assert gaps == [{"role": "research", "technology": "ITME", "item": "운영 환경 실증",
                     "reason": "미확인"}]


def test_comparison_only_technology_is_partial_with_gap(monkeypatch):
    itme_cmp = chunk(ITME_ID, applies_to=["ITME"], scope="comparison")
    drafts(monkeypatch, {"turboquant": f"압축 [{TQ_ID}].", "itme": f"비교군뿐 [{ITME_ID}].", "gaps": []})

    a = run([TQ, itme_cmp])["research"]

    assert a["status"] == "partial"
    assert [g["item"] for g in a["gaps"]] == ["ITME 1차 원문 인용 미확보"]


def test_section_without_citation_is_a_gap_not_a_claim(monkeypatch):
    drafts(monkeypatch, {"turboquant": f"압축 [{TQ_ID}].", "itme": "근거 없이 쓴 문장.", "gaps": []})

    a = run([TQ, ITME])["research"]

    assert [c["technology"] for c in a["claims"]] == ["TurboQuant"]
    assert "ITME 절 인용 근거 미확보" in [g["item"] for g in a["gaps"]]


def test_unknown_citation_fails_with_no_claims(monkeypatch):
    drafts(monkeypatch, {"turboquant": f"압축 [{UNSEEN_ID}].", "itme": f"가속 [{ITME_ID}].", "gaps": []})

    a = run([TQ, ITME])["research"]

    assert a["status"] == "failed" and a["claims"] == []


def test_no_citations_fails_cleanly(monkeypatch):
    drafts(monkeypatch, {"turboquant": "근거 없음.", "itme": "근거 없음.", "gaps": []})

    a = run([])["research"]

    assert (a["status"], a["claims"], a["sources"], a["evidence"]) == ("failed", [], [], [])


def test_unsupported_number_gets_one_rewrite_with_the_offending_line(monkeypatch):
    """기준선 실행 — 지시문의 8배가 근거 없이 research 본문에 들어갔다."""
    tq = chunk(TQ_ID, applies_to=["TurboQuant"], text="TurboQuant quantizes to 3.5 bits.")
    calls = drafts(monkeypatch,
                   {"turboquant": f"H100 에서 8배 가속한다 [{TQ_ID}].", "itme": f"가속 [{ITME_ID}].", "gaps": []},
                   {"turboquant": f"3.5비트로 양자화한다 [{TQ_ID}].", "itme": f"가속 [{ITME_ID}].",
                    "gaps": [{"technology": "TurboQuant", "item": "H100 가속 배수"}]})

    out = run([tq, ITME])

    assert len(calls) == 2 and "8배" in calls[1]
    assert out["trace"][0]["attempt"] == 2 and out["trace"][0]["number_problems"] == 0
    assert "8배" not in out["research"]["claims"][0]["text"]


def test_rewrite_happens_at_most_once(monkeypatch):
    bad = {"turboquant": f"8배 [{TQ_ID}].", "itme": f"가속 [{ITME_ID}].", "gaps": []}
    calls = drafts(monkeypatch, bad, bad)

    trace = run([TQ, ITME])["trace"][0]

    assert len(calls) == 2 and trace["number_problems"] == 1   # 남은 오류는 review 가 막는다


def test_domain_hypothesis_is_a_separate_claim_with_premises(monkeypatch):
    drafts(monkeypatch, {"turboquant": f"배치를 키운다 [{TQ_ID}].", "itme": f"세션을 늘린다 [{ITME_ID}].",
                         "hypothesis": "결합하면 시너지가 있을 것으로 추정된다.", "gaps": []})

    a = Assessment.model_validate(run([TQ, ITME], node="domain_assessment", role="domain",
                                      schema=DomainDraft)["domain_assessment"])

    assert [c.kind for c in a.claims] == ["fact", "fact", "hypothesis"]
    hypothesis = a.claims[-1]
    assert set(hypothesis.evidence_ids) == {TQ_ID, ITME_ID} and hypothesis.explanation
    assert all(s.allowed_uses == ["domain"] for s in a.sources)



def test_uncited_hypothesis_number_uses_the_rewrite(monkeypatch):
    """가설은 인용이 없으면 본문 전체 인용을 전제로 Claim 이 된다 — 같은 근거로 미리 검사한다."""
    calls = drafts(monkeypatch,
                   {"turboquant": f"배치를 키운다 [{TQ_ID}].", "itme": f"세션을 늘린다 [{ITME_ID}].",
                    "hypothesis": "결합하면 비용이 9배 준다.", "gaps": []},
                   {"turboquant": f"배치를 키운다 [{TQ_ID}].", "itme": f"세션을 늘린다 [{ITME_ID}].",
                    "hypothesis": "결합 효과는 실측이 없다.", "gaps": []})

    out = run([TQ, ITME], node="domain_assessment", role="domain", schema=DomainDraft)

    assert len(calls) == 2 and "9배" in calls[1]
    assert out["trace"][0]["number_problems"] == 0


def test_structured_output_failure_retries_once_then_fails_cleanly(monkeypatch):
    """Gemini 가 JSON 을 잘라 보내도 노드가 죽지 않는다 — 1회 재시도 후 failed Assessment."""
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise ValueError("truncated JSON")

    monkeypatch.setattr(common, "draft", broken)
    out = run([TQ, ITME])

    a = Assessment.model_validate(out["research"])
    assert len(calls) == 2 and a.status == "failed" and a.claims == []
    assert {g.technology for g in a.gaps} == {"TurboQuant", "ITME"}
    assert "truncated JSON" in out["trace"][0]["error"]


def test_structured_output_failure_then_success_uses_the_retry(monkeypatch):
    outputs = iter([ValueError("truncated"),
                    {"turboquant": f"압축 [{TQ_ID}].", "itme": f"가속 [{ITME_ID}].", "gaps": []}])

    def flaky(instruction, domain, context, schema=Draft, feedback=""):
        nxt = next(outputs)
        if isinstance(nxt, Exception):
            raise nxt
        return schema.model_validate(nxt)

    monkeypatch.setattr(common, "draft", flaky)
    out = run([TQ, ITME])

    assert out["research"]["status"] == "completed" and out["trace"][0]["attempt"] == 2


def test_market_unsupported_number_gets_one_rewrite(monkeypatch):
    from src.agents import market

    eco = [chunk("cccccccccccc", applies_to=["ITME"], collection="ecosystem", scope="ecosystem",
                 text="CXL modules shipped."),
           chunk("dddddddddddd", applies_to=["TurboQuant"], collection="ecosystem", scope="ecosystem",
                 text="Frameworks support KV cache quantization.")]
    monkeypatch.setattr(market, "bind_document_search",
                        lambda *a, **k: SimpleNamespace(invoke=lambda p: eco))
    texts = iter(["## ① 시장 규모/성장성\n시장이 30% 성장 [cccccccccccc].\n## ② 상용화/채택 현황\n지원 [dddddddddddd].",
                  "## ① 시장 규모/성장성\n정량 리포트 미확인 [cccccccccccc].\n## ② 상용화/채택 현황\n지원 [dddddddddddd]."])
    instructions = []
    monkeypatch.setattr(market, "run_node", lambda instruction, d, c: instructions.append(instruction) or next(texts))

    out = market.market({"run_id": "r4-test", "run_config": {}})

    assert len(instructions) == 2 and "30%" in instructions[1]
    assert out["trace"][0]["attempt"] == 2 and out["trace"][0]["number_problems"] == 0
