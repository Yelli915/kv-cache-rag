"""평가 종합 — 담당: R5 | State 읽기 + gaps 순차 병합 (설계서 §6·§7)

원칙:
- 새로운 외부 근거를 검색하지 않는다. §7 — 종합 단계는 이미 병합된 근거를 재사용하며
  새 자료가 필요하면 그 항목을 공백으로 남긴다.
- 앞선 평가 노드가 확정한 Claim 만 재사용한다.
- 관점 간 차이를 억지로 만들지 않는다(§5).
- 결합 효과는 공개 실측 근거가 없으면 가설로만 기록한다(§2·§5).

`Synthesis`·`Conflict` 는 `src/schema.py` 가 소유한다. 여기서 다시 정의하지 않는다.
"""

import difflib

from src.llm import get_llm
from src.schema import Gap, Synthesis

PERSPECTIVES = [
    "maturity",
    "market",
    "stakeholder",
    "domain_assessment",
]


GAP_SIMILARITY = 0.6    # 실측: 종합이 다시 쓴 공백은 대개 0.6~0.9, 새 항목은 0.5 미만


def _restates(item: str, existing: list[str]) -> bool:
    """이미 있는 공백을 말만 바꿔 되풀이한 것인지 본다."""
    return any(difflib.SequenceMatcher(None, item, seen).ratio() >= GAP_SIMILARITY
               for seen in existing)


def _claims_text(assessment: dict) -> str:
    """Assessment 를 종합 입력용 텍스트로 편다. 본문(text) 필드는 더 이상 없다."""
    if not assessment:
        return "(없음)"
    lines = [f"status: {assessment.get('status', '미상')}"]
    for claim in assessment.get("claims") or []:
        lines.append(
            f"- [{claim.get('kind', '')}/{claim.get('technology', '')}] "
            f"{claim.get('text', '').strip()} "
            f"(근거 {len(claim.get('evidence_ids') or [])}건)"
        )
    for gap in assessment.get("gaps") or []:
        lines.append(f"- (공백) {gap.get('technology', '')} {gap.get('item', '')}: "
                     f"{gap.get('reason', '')}")
    return "\n".join(lines) if len(lines) > 1 else "status: " + str(assessment.get("status"))


PROMPT = """너는 평가 종합 담당이다.

TRL, 시장성, 이해관계자, 도메인 적용 평가 결과를 종합한다.

반드시 다음 원칙을 따른다.

1. 아래 입력에 포함된 내용만 사용한다.
2. 새로운 성능 수치, 시장 사례, 채택 사례, 실험 결과를 만들어내지 않는다.
3. 특정 기술을 종합 승자 또는 더 우수한 기술로 판정하지 않는다.
4. agreements에는 여러 관점에서 공통적으로 확인되는 내용을 적는다.
5. conflicts에는 실제로 평가가 달라지는 경우만 적는다.
   차이가 없다면 빈 목록으로 둔다.
6. 상충이나 차이를 보고서 목적을 위해 억지로 만들어내지 않는다.
7. 각 관점에서 근거가 없다고 판단한 내용은 gaps에 유지한다.
8. TurboQuant와 ITME를 동일 시스템에서 함께 측정한 공개 실험이
   확인되지 않았다면 결합 효과는 실측 결과처럼 쓰지 않는다.
9. combination_hypothesis는 개별 기술의 확인된 특성으로부터 도출한
   가설임을 명확히 표현한다.


# TRL 평가
{maturity}

# 시장성 평가
{market}

# 이해관계자 평가
{stakeholder}

# 도메인 적용 평가
{domain_assessment}

# 기술 조사
{research}

# 기존 근거 공백
{gaps}
"""


def synthesis(state) -> dict:
    """종합 결과와, 합류분에 자기 공백을 이어 붙인 gaps 를 반환한다(§7 순차 병합).

    구조화 출력이 깨지면 `synthesis.error` 에 사유를 남긴다. `validate.check` 가 이를 오류로
    올려 `final_check` 가 fail 로 보낸다 — 예외로 그래프를 죽이면 사유 파일이 남지 않는다.
    """
    merged_gaps = list(state.get("gaps") or [])          # collect_evidence 가 합류시킨 공백
    # §6 은 종합에게 공백을 "정리" 하라고 한다. 그런데 모델은 노드가 이미 보고한 공백을
    # 자기 말로 다시 쓴다. 문자열 완전 일치로는 못 걸러서 유사도로 본다.
    seen = [gap.get("item", "") for gap in merged_gaps]

    prompt = PROMPT.format(
        research=_claims_text(state.get("research") or {}),
        gaps="\n".join(f"- {gap.get('technology', '')} {gap.get('item', '')}: "
                        f"{gap.get('reason', '')}" for gap in merged_gaps) or "(없음)",
        **{perspective: _claims_text(state.get(perspective) or {})
           for perspective in PERSPECTIVES},
    )
    try:
        result = get_llm().with_structured_output(Synthesis).invoke(prompt)
    except Exception as exc:             # 파서·API 오류 모두 — common._try_draft 와 같은 처리
        result, error = None, f"{type(exc).__name__}: {exc}"[:300]
    else:
        error = "" if result is not None else "빈 구조화 출력"
    if result is None:
        return {"synthesis": {"error": error}, "gaps": merged_gaps,
                "trace": [{"node": "synthesis", "status": "failed", "attempt": 1, "error": error}]}

    # 종합이 새로 지목한 공백만 Gap 으로 덧붙인다. 앞 단계 공백은 그대로 유지한다.
    for item in result.gaps:
        gap = Gap(role="synthesis", technology="both", item=item,
                  reason="종합 단계에서 확인한 공백").model_dump()
        if not _restates(item, seen):
            seen.append(item)
            merged_gaps.append(gap)

    # 보고서 §5 가 읽는 목록에도 앞 단계 공백을 남긴다.
    result.gaps = sorted({*result.gaps, *(gap.get("item", "") for gap in merged_gaps)} - {""})

    return {
        "synthesis": result.model_dump(),
        "gaps": merged_gaps,
        "trace": [{
            "node": "synthesis",
            "status": "ok",
            "attempt": 1,
            "agreements": len(result.agreements),
            "conflicts": len(result.conflicts),
            "gaps": len(merged_gaps),
        }],
    }
