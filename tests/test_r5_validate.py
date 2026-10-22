"""R5 validate.py 단위 테스트 — 결정적 형식 검사와 수치 검사."""

import pytest

from src.output import validate


def _base_state():
    """정상적인 Claim → Evidence → Source 연결."""
    return {
        "run_id": "run-test-001",
        "stakeholder": {
            "claims": [{"claim_id": "claim-001", "technology": "ITME", "kind": "fact",
                        "text": "테스트용 주장", "evidence_ids": ["evidence-001"]}],
            "evidence": [{"evidence_id": "evidence-001", "source_id": "source-001",
                          "run_id": "run-test-001", "quote": "테스트용 원문 근거",
                          "allowed_uses": ["stakeholder"]}],
            "sources": [{"source_id": "source-001", "run_id": "run-test-001",
                         "url": "https://example.com/test"}],
            "gaps": [],
            "status": "completed",
        },
    }


def kinds(state):
    return [error["kind"] for error in validate.check(state)["validation"]["errors"]]


def test_valid_structured_claim_passes():
    assert kinds(_base_state()) == []


@pytest.mark.parametrize("path,value,kind", [
    (("claims", "evidence_ids"), ["evidence-missing"], "Evidence 없음"),
    (("evidence", "run_id"), "run-other", "다른 run의 Evidence"),
    (("evidence", "allowed_uses"), ["market"], "허용되지 않은 Evidence 사용"),
])
def test_broken_link_fails(path, value, kind):
    state = _base_state()
    state["stakeholder"][path[0]][0][path[1]] = value

    assert kind in kinds(state)


def test_check_writes_the_validation_contract():
    """State 필드는 validation 하나다. validation_errors/validation_round 는 필드가 아니다."""
    result = validate.check(_base_state())

    assert set(result) == {"validation", "trace"}
    assert set(result["validation"]) == {"errors", "checked_nodes"}
    assert result["trace"][0]["node"] == "validate"


def test_errors_keep_claim_id_for_locating_the_body():
    """설계서 §8 — 검증 실패 시 본문 위치를 찾을 수 있도록 claim_id 를 유지한다."""
    state = _base_state()
    state["stakeholder"]["claims"][0]["evidence_ids"] = ["evidence-missing"]

    errors = validate.check(state)["validation"]["errors"]
    assert errors and all(error["claim_id"] for error in errors)


def _numbers_state(text, quote):
    state = _base_state() | {"run_config": {"numeric_check": True}}
    state["stakeholder"]["claims"][0]["text"] = text
    state["stakeholder"]["evidence"][0]["quote"] = quote
    return state


def _numeric_kinds(text, quote, technology="ITME"):
    state = _numbers_state(text, quote)
    state["stakeholder"]["claims"][0]["technology"] = technology
    errors = validate.check(state)["validation"]["errors"]
    return [(e["kind"], e["value"]) for e in errors]


def test_number_absent_from_the_cited_quote_fails():
    """기준선 실행 — 지시문의 수치가 근거 없이 본문에 들어간 채 ID 검사를 통과했다."""
    assert _numeric_kinds("어텐션 로짓 연산이 H100 에서 8배 빨라진다.",
                          "TurboQuant quantizes the KV cache to 3.5 bits.") == [("근거에 없는 수치", "8배")]


def test_korean_units_match_english_sources():
    quote = "Up to 8x faster attention logits on H100 GPUs, 3.5-bit cache, 35.7% after 128 GB."
    assert _numeric_kinds("H100 에서 8배, 3.5비트, 128GB 소진 후 35.7% 개선.", quote) == []


def test_table_numbers_count_except_for_multipliers():
    """논문 표는 단위를 열 제목에만 쓴다. 단 배수는 표의 아무 숫자로 통과시키지 않는다."""
    table = "Method KV Size Average\nTurboQuant(ours) 2.5 49.44\nLlama-3.1-8B-Instruct d=8"
    assert _numeric_kinds("2.5비트에서 평균 49.44 이다.", table) == []
    assert _numeric_kinds("8배 빠르다.", table) == [("근거에 없는 수치", "8배")]


def test_conditioned_number_without_its_condition_fails():
    """설계서 §2 — 8배는 H100 어텐션 로짓 연산 한정이다."""
    assert _numeric_kinds("어텐션이 8배 빨라진다.", "8x speedup on H100.",
                          "TurboQuant") == [("수치 조건 누락", "8배")]


def test_condition_rules_apply_only_to_their_technology():
    """8배·H100 은 TurboQuant 규칙이다. 다른 기술의 정상적인 8배 주장에 H100 을 요구하지 않는다."""
    assert _numeric_kinds("메모리를 8배 줄인다.", "reduces memory by 8x", "ITME") == []
    # 어느 기술인지 모르는 Claim(market 등)은 모든 규칙을 건다
    assert _numeric_kinds("메모리를 8배 줄인다.", "reduces memory by 8x", "both") == [("수치 조건 누락", "8배")]


def test_gap_list_line_is_not_a_claim():
    assert _numeric_kinds("확인한 사실 없음.\n근거 공백: H100 8배 가속 수치", "no numbers") == []


def test_condition_can_be_stated_the_way_the_source_states_it():
    """ITME 초록은 35.7% 의 조건을 128GB 가 아니라 호스트 메모리 한계로 적는다."""
    quote = "accommodate KV cache beyond host memory limits, achieving up to a 35.7% throughput improvement"
    assert _numeric_kinds("호스트 메모리 한계를 넘는 구간에서 35.7% 개선.", quote) == []


def test_number_is_checked_against_its_own_sentences_citation():
    """개선 후 실행 — 문단의 다른 문장이 인용한 원문에 같은 값이 있어 128GB 가 통과했다."""
    state = _numbers_state("구조는 이렇다 [aaaaaaaaaaaa]. 128GB 소진 후 35.7% 개선 [bbbbbbbbbbbb].", "")
    state["stakeholder"]["claims"][0]["evidence_ids"] = ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    state["stakeholder"]["evidence"] = [
        {**state["stakeholder"]["evidence"][0], "evidence_id": "aaaaaaaaaaaa", "quote": "pool of 128GB DRAM"},
        {**state["stakeholder"]["evidence"][0], "evidence_id": "bbbbbbbbbbbb",
         "quote": "35.7% throughput beyond host memory limits"},
    ]
    errors = validate.check(state)["validation"]["errors"]
    assert [(e["kind"], e["value"]) for e in errors] == [("근거에 없는 수치", "128GB")]


def test_a_number_with_another_unit_is_not_a_bare_table_number():
    """그림 축 `128 MB` 로 `128GB` 를 통과시키지 않는다."""
    assert _numeric_kinds("128GB 를 쓴다.", "Transfer Size 64 MB 128 MB 256 MB") == [("근거에 없는 수치", "128GB")]


def test_runs_before_full_quotes_skip_the_numeric_check():
    """`--resume <옛 run_id>` — 300자로 잘린 quote 로 맞는 수치를 오류로 만들지 않는다."""
    state = _numbers_state("35.7% 개선 (호스트 메모리 한계 이후).", "ITME improves throughput up to")
    del state["run_config"]
    result = validate.check(state)
    assert result["validation"]["errors"] == []
    assert result["trace"][0]["numeric_check"] is False


def test_dimension_and_hex_are_not_multipliers():
    assert validate.numbers("1x1 conv, 4×4 tile, 0x1f 주소") == set()
    assert validate.numbers("speedup of 8x. and 2× faster") == {(8.0, "배"), (2.0, "배")}
