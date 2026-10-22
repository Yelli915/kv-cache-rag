"""평가 노드 넷이 schema.Assessment 계약을 지키는지 본다 (설계서 §6).

검색과 LLM 은 fixture 로 대체한다. 확인하는 것은 *계약* 이다 — 자기 결과 키 하나와
trace 만 쓰는지, 반환한 dict 가 Assessment 로 검증되는지, 인용 사슬이 닫히는지.
"""

import pytest

from src.agents import common, domain, market, maturity, research
from src.schema import Assessment
from tests.helpers import ITME_ID, PAPERS, TQ_ID, search

NODES = [(research.research, "research"), (maturity.maturity, "maturity"),
         (market.market, "market"), (domain.domain_assessment, "domain_assessment")]
STATE = {"run_id": "r4-test", "run_config": {"domain": "데이터센터/클라우드",
                                             "technologies": ["TurboQuant", "ITME"]}}
MARKET_DRAFT = {"size": f"정량 리포트 미확인 [{TQ_ID}].",
                "adoption": f"발표 단계, 배포 확인 미확인 [{ITME_ID}].",
                "ecosystem": f"일반 CXL 근거이며 추론 특화 여부 미확인 [{ITME_ID}].",
                "gaps": [{"technology": "TurboQuant", "item": "서빙 처리량 미확인"},
                         {"technology": "both", "item": "결합 실측 자료 없음"}]}
DRAFT = {"turboquant": f"TurboQuant 는 3.5비트 양자화를 보고한다 [{TQ_ID}].",
         "itme": f"ITME 는 NVMe-oF 대비 1.80배 처리량을 보고한다 [{ITME_ID}].",
         "hypothesis": f"두 기술의 결합은 실측 근거가 없다 [{TQ_ID}].",
         "gaps": [{"technology": "TurboQuant", "item": "서빙 처리량"}]}


@pytest.fixture
def stub(monkeypatch):
    for module in (research, maturity, market, domain):
        monkeypatch.setattr(module, "bind_document_search", lambda *a, **k: search(PAPERS))
    monkeypatch.setattr(common, "draft", lambda *a, schema=common.Draft, **k: schema.model_validate(
        MARKET_DRAFT if schema is market.MarketDraft else DRAFT))


@pytest.mark.parametrize("node,key", NODES, ids=[key for _, key in NODES])
def test_node_returns_a_valid_assessment_with_evidence_per_claim(stub, node, key):
    update = node(STATE)

    assert set(update) == {key, "trace"}, "자기 결과 키와 trace 만 쓴다 (§7)"
    assessment = Assessment.model_validate(update[key])   # 인용 사슬까지 여기서 검증된다
    assert update["trace"][0]["node"] == key
    # §6 — 근거가 없는 항목은 Claim 이 아니라 Gap 으로 기록한다.
    assert assessment.claims and all(claim.evidence_ids for claim in assessment.claims)
    assert all(claim.explanation for claim in assessment.claims if claim.kind != "fact")


def test_market_without_evidence_fails_instead_of_inventing(monkeypatch):
    """§5 — 자료가 없으면 미확인. 근거 0건으로 시장성을 생성하지 않는다."""
    monkeypatch.setattr(market, "bind_document_search", lambda *a, **k: search([]))
    assessment = Assessment.model_validate(market.market(STATE)["market"])

    assert assessment.status == "failed" and assessment.claims == []
    assert {gap.technology for gap in assessment.gaps} == {"TurboQuant", "ITME"}
