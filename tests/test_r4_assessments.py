"""평가 노드 넷이 schema.Assessment 계약을 지키는지 본다 (설계서 §6).

검색과 LLM 은 fixture 로 대체한다. 확인하는 것은 *계약* 이다 — 자기 결과 키 하나와
trace 만 쓰는지, 반환한 dict 가 Assessment 로 검증되는지, 인용 사슬이 닫히는지.
"""

import pytest

from src.agents import common, domain, market, maturity, research
from src.schema import Assessment

NODES = [
    (research, "research", "research"),
    (maturity, "maturity", "maturity"),
    (market, "market", "market"),
    (domain, "domain_assessment", "domain"),
]

CHUNKS = [
    {"chunk_id": "aaaaaaaaaaaa", "source_id": "turboquant-paper", "collection": "papers_core",
     "page": 3, "text": "TurboQuant reports 3.5-bit KV cache quantization.",
     "applies_to": ["TurboQuant"], "scope": "direct", "cosine_score": 0.7},
    {"chunk_id": "bbbbbbbbbbbb", "source_id": "itme-paper", "collection": "papers_core",
     "page": 7, "text": "ITME reports 1.80x throughput over an NVMe-oF baseline.",
     "applies_to": ["ITME"], "scope": "direct", "cosine_score": 0.6},
]


class FakeSearch:
    def invoke(self, payload):
        return CHUNKS


def fake_text(instruction, domain_text, context):
    """market 의 세 절."""
    return (
        "## ① 시장 규모/성장성\n정량 리포트 미확인 [aaaaaaaaaaaa].\n\n"
        "## ② 상용화/채택 현황\n발표 단계, 배포 확인 미확인 [bbbbbbbbbbbb].\n\n"
        "## ③ 생태계 지지\n일반 CXL 근거이며 추론 특화 여부 미확인 [bbbbbbbbbbbb].\n\n"
        "근거 공백: TurboQuant 서빙 처리량 미확인 | 결합 실측 자료 없음"
    )


def fake_draft(instruction, domain_text, context, schema=common.Draft, feedback=""):
    return schema.model_validate({
        "turboquant": "TurboQuant 는 3.5비트 양자화를 보고한다 [aaaaaaaaaaaa].",
        "itme": "ITME 는 NVMe-oF 대비 1.80배 처리량을 보고한다 [bbbbbbbbbbbb].",
        "hypothesis": "두 기술의 결합은 실측 근거가 없다 [aaaaaaaaaaaa].",
        "gaps": [{"technology": "TurboQuant", "item": "서빙 처리량"}],
    })


@pytest.fixture
def stub(monkeypatch):
    for module, _, _ in NODES:
        monkeypatch.setattr(module, "bind_document_search", lambda *a, **k: FakeSearch())
    monkeypatch.setattr(market, "run_node", fake_text)
    monkeypatch.setattr(common, "draft", fake_draft)


STATE = {"run_id": "r4-test", "run_config": {"domain": "데이터센터/클라우드",
                                             "technologies": ["TurboQuant", "ITME"]}}


@pytest.mark.parametrize("module,key,_role", NODES, ids=[n[1] for n in NODES])
def test_node_returns_a_valid_assessment(stub, module, key, _role):
    update = getattr(module, key if key != "domain_assessment" else "domain_assessment")(STATE)

    assert set(update) == {key, "trace"}, "자기 결과 키와 trace 만 쓴다 (§7)"
    assessment = Assessment.model_validate(update[key])   # 인용 사슬까지 여기서 검증된다
    assert assessment.status in {"completed", "partial", "failed"}
    assert update["trace"][0]["node"] == key


@pytest.mark.parametrize("module,key,_role", NODES, ids=[n[1] for n in NODES])
def test_every_claim_carries_evidence(stub, module, key, _role):
    """§6 — 근거가 없는 항목은 Claim 이 아니라 Gap 으로 기록한다."""
    assessment = Assessment.model_validate(
        getattr(module, "domain_assessment" if key == "domain_assessment" else key)(STATE)[key])

    assert all(claim.evidence_ids for claim in assessment.claims)
    assert all(claim.explanation for claim in assessment.claims if claim.kind != "fact")


def test_market_without_evidence_fails_instead_of_inventing(monkeypatch):
    """§5 — 자료가 없으면 미확인. 근거 0건으로 시장성을 생성하지 않는다."""
    class Empty:
        def invoke(self, payload):
            return []

    monkeypatch.setattr(market, "bind_document_search", lambda *a, **k: Empty())
    assessment = Assessment.model_validate(market.market(STATE)["market"])

    assert assessment.status == "failed" and assessment.claims == []
    assert {gap.technology for gap in assessment.gaps} == {"TurboQuant", "ITME"}
