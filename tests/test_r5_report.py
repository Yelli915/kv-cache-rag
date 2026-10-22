"""R5 report.py 단위 테스트. 판단 LLM 을 부르지 않는 결정적 포맷터다."""

import pytest

from src.agents import report as report_module
from tests.helpers import assessment


@pytest.fixture(autouse=True)
def fixed_model(monkeypatch):
    monkeypatch.setattr(report_module, "model_name", lambda: "test-model")


def _state():
    return {
        "run_id": "r5-test",
        "run_config": {"domain": "데이터센터/클라우드", "started_at": "2026-09-22T00:00:00+00:00"},
        "research": assessment("research", "기술 조사 본문"),
        "maturity": assessment("maturity", "TRL 평가 본문"),
        "market": assessment("market", "시장성 평가 본문"),
        "stakeholder": assessment("stakeholder", "이해관계자 평가 본문"),
        "domain_assessment": assessment("domain", "도메인 평가 본문"),
        "gaps": [{"role": "market", "technology": "TurboQuant", "item": "서빙 처리량",
                  "reason": "공개 자료 미확인"}],
        "synthesis": {
            "agreements": ["두 기술 모두 KV cache 메모리 병목을 완화한다."],
            "conflicts": [],
            "gaps": ["두 기술의 결합 실측 자료 미확인"],
            "combination_hypothesis": "두 기술은 서로 다른 시스템 계층에서 병목을 완화할 가능성이 있다.",
        },
    }


def render(state=None):
    return report_module.report(state or _state())["report"]


def section6(markdown):
    return markdown[markdown.index("## 6. 한계"):]


def test_report_has_every_section_and_every_approved_claim():
    """conflicts 가 비어도 정상 생성된다 — 상충 0건은 오류가 아니다(§5)."""
    result = report_module.report(_state())
    markdown = result["report"]

    for heading in ("# KV cache 최적화 기술 다관점 평가", "## SUMMARY", "## 4. 관점별 평가",
                    "## 5. 시사점", "## 6. 한계", "## REFERENCE"):
        assert heading in markdown
    assert "확인된 관점 간 상충 없음" in markdown
    for body in ("기술 조사 본문", "TRL 평가 본문", "시장성 평가 본문", "이해관계자 평가 본문",
                 "도메인 평가 본문"):
        assert body in markdown
    assert result["trace"][0]["node"] == "report" and result["trace"][0]["deterministic"] is True


def test_rejected_claim_leaves_the_body_and_becomes_a_gap():
    """§8 — 부결된 주장은 사실 서술에서 제외하고 평가 보류 항목으로 옮긴다."""
    state = _state() | {"validation": {
        "rejected_claims": ["claim-market"],
        "claim_verdicts": {"claim-market": {"verdict": "부결", "reviewer": "권예리",
                                            "comment": "인접 CXL 제품 자료임"}},
    }}
    markdown = render(state)

    assert "시장성 평가 본문" not in markdown
    assert "검토 부결: claim-market" in markdown and "권예리" in markdown
    # §9 — 본문에서 실제 사용한 자료만 REFERENCE 에 남는다.
    assert "market 출처" not in markdown
    assert "maturity 출처" in markdown


def test_gaps_and_survey_date_are_in_the_limits_section():
    """§9 목차표 — 6 한계는 자료 공백·조사 시점·분석의 한계를 담는다."""
    markdown = render()
    limits = section6(markdown)

    assert "TurboQuant 서빙 처리량: 공개 자료 미확인" in limits
    assert "2026-09-22T00:00:00+00:00" in limits
    assert "### 5.3 결합 가설" in markdown        # 5 절에는 공백 소절이 없다


def test_only_inference_and_hypothesis_are_marked():
    """§6 — 사실과 추론을 구분한다. 사실이 기본이라 모든 문단에 라벨을 붙이지 않는다."""
    state = _state()
    state["domain_assessment"]["claims"].append({
        "claim_id": "claim-hypo", "text": "결합 효과 가설", "technology": "both",
        "kind": "hypothesis", "evidence_ids": ["e-domain"],
        "explanation": "개별 근거에서 도출한 추론이다",
    })
    markdown = render(state)

    assert "[사실" not in markdown and "[추론" not in markdown and "[가설" not in markdown
    assert "TRL 평가 본문" in markdown                     # 사실 주장은 본문만
    assert "> **가설** — 실측 결과가 아니다. 전제: 개별 근거에서 도출한 추론이다" in markdown


def test_carried_verdicts_are_disclosed():
    """§10 — 검토 범위를 명시적으로 기록한다. 이월은 이번 실행에서 다시 본 것이 아니다."""
    state = _state() | {"validation": {
        "carried_claims": ["claim-market"],
        "claim_verdicts": {"claim-market": {"verdict": "확인", "reviewer": "권예리",
                                            "carried_from": "20260922-122252-d94611"}},
    }}
    limits = section6(render(state))

    assert "내용 검토 이월: 주장 1건" in limits
    assert "20260922-122252-d94611" in limits
    assert "이번 실행에서 다시 검토하지 않았다" in limits


def test_summary_stays_within_the_half_page_budget():
    """§9 — SUMMARY 는 PDF 반 페이지 이내다. 전부 나열하면 제출본이 생성되지 않는다."""
    from src.output.pdf import check_summary

    state = _state()
    state["synthesis"] |= {
        "agreements": [f"공통 사실 {i}" for i in range(7)],
        "conflicts": [{"perspective": p, "why": f"{p} 근거 범위가 다르다"}
                      for p in ("TRL", "시장성", "이해관계자", "도메인")],
        "gaps": [f"근거 공백 {i}" for i in range(41)],
    }
    markdown = render(state)

    passed, message = check_summary(markdown)
    assert passed, message
    assert "외 38건은 §6 참고" in markdown      # 나머지는 절을 가리킨다
    assert "외 4건은 §5.1 참고" in markdown


def test_gap_line_does_not_repeat_the_technology():
    state = _state()
    state["gaps"] = [
        {"role": "research", "technology": "TurboQuant",
         "item": "TurboQuant의 E2E 처리량", "reason": "제공된 근거에서 확인하지 못함"},
        {"role": "stakeholder", "technology": "ITME",
         "item": "investors", "reason": "원문 본문 근거 미확보"},
    ]
    limits = section6(render(state))

    assert "TurboQuant TurboQuant의" not in limits
    assert "- TurboQuant의 E2E 처리량: 제공된 근거에서 확인하지 못함" in limits
    assert "- ITME investors: 원문 본문 근거 미확보" in limits      # 접두어가 필요한 쪽


def test_cached_model_responses_are_disclosed(monkeypatch):
    """§6 — 실제 비용은 실행 기록으로 확인한다. 캐시 적중은 새로 판단하지 않았다는 뜻이다."""
    monkeypatch.setattr(report_module, "llm_report",
                        lambda: {"cache_hits": 5, "cache_misses": 1})

    limits = section6(render())

    assert "모델 응답 재사용: 5건" in limits and "새로 생성 1건" in limits


def test_inline_citations_become_reference_numbers():
    """본문의 chunk_id 는 검증용 내부 값이다. 독자는 REFERENCE 번호를 읽어야 한다."""
    state = _state()
    state["run_config"]["sources"] = [
        {"id": "s-maturity", "title": "ITME 원문", "url": "https://arxiv.org/abs/2606.12556",
         "version": "arXiv:2606.12556v2", "publisher": "arXiv"},
    ]
    state["maturity"]["claims"][0]["text"] = "ITME 는 1.80배를 보고한다 [aaaaaaaaaaaa]."
    state["maturity"]["evidence"][0] |= {"evidence_id": "aaaaaaaaaaaa", "location": "p.7",
                                         "collection": "papers_core"}
    state["maturity"]["claims"][0]["evidence_ids"] = ["aaaaaaaaaaaa"]
    state["maturity"]["sources"][0]["collection"] = "papers_core"

    markdown = render(state)

    assert "[aaaaaaaaaaaa]" not in markdown          # 내부 ID 가 본문에 남지 않는다
    assert "ITME 는 1.80배를 보고한다 [1]." in markdown
    reference = markdown[markdown.index("## REFERENCE"):]
    assert "- [1] ITME 원문" in reference
    assert "arXiv:2606.12556v2" in reference and "인용 p.7" in reference   # §9 항목
    assert "https://arxiv.org/abs/2606.12556" in reference


def test_unknown_citation_id_is_left_alone():
    """번호를 못 찾으면 원래 ID 를 남긴다. 조용히 지우면 인용이 사라진 것처럼 보인다."""
    state = _state()
    state["maturity"]["claims"][0]["text"] = "근거 없는 인용 [ffffffffffff]."

    assert "[ffffffffffff]" in render(state)


def test_overview_section_has_no_verdict_labels():
    """§9 — 3 기술 개요는 각 기술의 접근 방식과 적용 조건을 적는 서술 절이다."""
    markdown = render()
    overview = markdown[markdown.index("## 3. 기술 개요"):markdown.index("## 4. 관점별 평가")]
    perspectives = markdown[markdown.index("## 4. 관점별 평가"):markdown.index("## 5. 시사점")]

    assert "**[사실" not in overview and "기술 조사 본문" in overview
    assert "TRL 평가 본문" in perspectives


def test_both_is_written_in_korean():
    state = _state()
    state["gaps"] = [{"role": "market", "technology": "both", "item": "시장 규모",
                      "reason": "미확인"}]
    markdown = render(state)

    assert "두 기술 시장 규모" in markdown and "both" not in markdown


def test_both_prefix_is_skipped_when_the_item_already_says_it():
    """LLM 이 "양 기술의 ..." 라고 쓰면 "두 기술 양 기술의 ..." 가 되면 안 된다."""
    state = _state()
    state["gaps"] = [
        {"role": "research", "technology": "both", "item": "양 기술의 하드웨어 스펙 비교",
         "reason": "제공된 근거에서 확인하지 못함"},
        {"role": "market", "technology": "both", "item": "시장 규모·성장률",
         "reason": "ecosystem 색인 내 근거 미확인"},
    ]
    limits = render(state)

    assert "- 양 기술의 하드웨어 스펙 비교:" in limits
    assert "두 기술 양 기술의" not in limits
    assert "- 두 기술 시장 규모·성장률:" in limits      # 기술을 안 말한 항목엔 붙인다


def test_premise_prefix_is_not_doubled():
    """모델이 "전제는 ..." 으로 시작해도 "전제: 전제는" 이 되지 않는다."""
    state = _state()
    state["market"]["claims"][0] |= {"kind": "inference",
                                     "explanation": "전제는 공식 코드가 없다는 점이다"}
    markdown = render(state)

    assert "전제: 공식 코드가 없다는 점이다" in markdown
    assert "전제: 전제는" not in markdown


def test_repeated_reference_numbers_collapse():
    """같은 출처의 다른 청크를 잇달아 인용해도 지면에는 번호가 한 번만 나온다."""
    state = _state()
    state["maturity"]["claims"][0]["text"] = "ITME 는 로컬 기준선 대비 개선을 보고한다 [aaaaaaaaaaaa][bbbbbbbbbbbb]."
    state["maturity"]["claims"][0]["evidence_ids"] = ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    state["maturity"]["evidence"] = [
        {"evidence_id": "aaaaaaaaaaaa", "source_id": "s-maturity", "run_id": "r5-test",
         "collection": "papers_core", "quote": "인용1", "location": "p.7",
         "allowed_uses": ["maturity"]},
        {"evidence_id": "bbbbbbbbbbbb", "source_id": "s-maturity", "run_id": "r5-test",
         "collection": "papers_core", "quote": "인용2", "location": "p.9",
         "allowed_uses": ["maturity"]},
    ]
    state["maturity"]["sources"][0]["collection"] = "papers_core"

    markdown = render(state)

    assert "개선을 보고한다 [1]." in markdown
    assert "[1][1]" not in markdown
    assert "인용 p.7, p.9" in markdown      # 두 위치는 REFERENCE 에 남는다
