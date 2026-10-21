"""종합 노드가 Assessment 를 읽고 gaps 를 순차 병합하는지 본다 (설계서 §5·§6·§7).

LLM 은 fixture 로 대체한다. 확인하는 것은 계약이다 — 어떤 State 필드를 읽고 쓰는지.
"""

import pytest

from src.agents import synthesis as node
from src.schema import Synthesis
from tests import helpers
from tests.helpers import FakeLLM


def assessment(role):
    """종합은 status·claims·gaps 만 읽는다. 공백이 있는 partial 로 둔다."""
    return helpers.assessment(role) | {"status": "partial", "gaps": [
        {"role": role, "technology": "TurboQuant", "item": f"{role} 공백", "reason": "근거 미확인"}]}


STATE = {
    "run_id": "r5-test",
    "research": assessment("research"),
    "maturity": assessment("maturity"),
    "market": assessment("market"),
    "stakeholder": assessment("stakeholder"),
    "domain_assessment": assessment("domain"),
    # collect_evidence 가 합류시킨 공백
    "gaps": [{"role": "maturity", "technology": "TurboQuant", "item": "maturity 공백",
              "reason": "근거 미확인"}],
}


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM(Synthesis(agreements=["공통 사실"], conflicts=[],
                             gaps=["결합 실측 자료 없음"],
                             combination_hypothesis="추론임을 명시한 결합 가설"))
    monkeypatch.setattr(node, "get_llm", lambda: fake)
    yield fake
    assert fake.schema is Synthesis, "공용 schema 를 써야 한다"


def test_writes_synthesis_and_merged_gaps(llm):
    update = node.synthesis(STATE)

    assert set(update) == {"synthesis", "gaps", "trace"}
    assert update["trace"][0]["node"] == "synthesis"


def test_gaps_merge_sequentially_without_dropping_earlier_ones(llm):
    """§7 — 합류 후 수집, 종합 후 순차 병합. 앞 단계 공백을 덮어쓰지 않는다."""
    gaps = node.synthesis(STATE)["gaps"]

    assert {gap["role"] for gap in gaps} == {"maturity", "synthesis"}
    assert any(gap["item"] == "maturity 공백" for gap in gaps)
    assert any(gap["role"] == "synthesis" and gap["item"] == "결합 실측 자료 없음"
               for gap in gaps)


def test_same_gap_is_not_added_twice(llm):
    """LLM 이 앞 단계와 같은 공백을 다시 말해도 한 번만 남는다."""
    llm.result = Synthesis(gaps=["maturity 공백"], combination_hypothesis="가설")
    gaps = node.synthesis(STATE)["gaps"]

    assert [gap["item"] for gap in gaps].count("maturity 공백") == 1


def test_reads_claims_not_the_removed_text_field(llm):
    """Assessment 에 text 필드는 없다. Claim 을 프롬프트에 넣어야 한다."""
    node.synthesis(STATE)

    assert "maturity 주장" in llm.prompt and "market 주장" in llm.prompt
    assert "research 주장" in llm.prompt      # 기술 조사도 종합 입력이다 (§6)


def test_broken_output_fails_with_a_reason_instead_of_raising(monkeypatch):
    """구조화 출력이 깨져도 그래프를 죽이지 않는다. 사유가 validation 오류로 올라가 fail 로 간다."""
    from src.graph import final_check
    from src.output import validate

    class Broken(FakeLLM):
        def invoke(self, prompt):
            raise ValueError("tool call 형식 오류")

    monkeypatch.setattr(node, "get_llm", lambda: Broken(None))
    update = node.synthesis(STATE)

    assert update["trace"][0]["status"] == "failed"
    assert "tool call 형식 오류" in update["synthesis"]["error"]
    assert update["gaps"] == STATE["gaps"]                  # 합류분 공백은 잃지 않는다
    errors = validate.check(STATE | update)["validation"]["errors"]
    assert [e["kind"] for e in errors if e["node"] == "synthesis"] == ["종합 생성 실패"]
    assert final_check(STATE | update | {"validation": {"errors": errors}})["run_status"] == "failed"


def test_no_conflict_is_allowed(llm):
    """§5 — 상충하는 의견을 억지로 만들지 않는다. conflicts 가 비어도 통과한다."""
    assert node.synthesis(STATE)["synthesis"]["conflicts"] == []


def test_restated_gap_is_not_added_again(llm):
    """§6 — 종합은 공백을 *정리* 한다. 말만 바꿔 다시 싣지 않는다."""
    llm.result = Synthesis(gaps=["maturity 공백에 대한 추가 확인 필요"],
                           combination_hypothesis="가설")
    items = [gap["item"] for gap in node.synthesis(STATE)["gaps"]]

    assert items == ["maturity 공백"]


def test_genuinely_new_gap_is_kept(llm):
    llm.result = Synthesis(gaps=["총소유비용 정량치"], combination_hypothesis="가설")
    gaps = node.synthesis(STATE)["gaps"]

    assert [g["item"] for g in gaps] == ["maturity 공백", "총소유비용 정량치"]
