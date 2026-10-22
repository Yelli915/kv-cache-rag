"""보고서 생성 — 담당: R5 | 결정적 포맷터

추가 판단 LLM을 호출하지 않고, 이미 확정된 State 내용을
설계서 §9 목차에 맞춰 조립한다.

원칙:
- SUMMARY도 synthesis 결과만 사용해 결정적으로 생성한다.
- 결합 가설은 §4 본문이 아니라 §5 시사점에만 배치한다.
- conflicts는 실제로 존재할 때만 출력한다.
- 최종 REFERENCE는 reference.py가 실제 사용 근거를 기준으로 생성한다.
"""

import re

from src.agents.common import domain_of
from src.llm import llm_report, model_name
from src.output import reference
from src.output.validate import CITATION_RE
from src.state import ASSESSMENT_NODES


def _bullets(items) -> str:
    """문자열 목록을 Markdown bullet로 변환한다."""
    return "\n".join(f"- {item}" for item in items) if items else "- 해당 없음"


KIND_LABEL = {"fact": "사실", "inference": "추론", "hypothesis": "가설"}
TECH_LABEL = {"both": "두 기술"}      # 내부 enum 값을 그대로 지면에 내보내지 않는다
ROLE_LABEL = {"research": "기술 조사", "maturity": "TRL", "market": "시장성",
              "stakeholder": "이해관계자", "domain": "도메인 적용", "synthesis": "종합"}
# item 이 이미 대상 기술을 말하고 있으면 접두어를 붙이지 않는다. LLM 이 쓴 문장이라
# "양 기술" · "두 기술" · 기술명 나열 등 표현이 갈린다.
TECH_ALIAS = {"both": ("두 기술", "양 기술", "TurboQuant", "ITME"),
              "TurboQuant": ("TurboQuant",), "ITME": ("ITME",)}


def _number_citations(markdown: str, marks: dict) -> str:
    """본문의 검증용 ID 를 REFERENCE 번호로 바꾼다.

    `[cefcd0973374]` 는 chunk_id 다 — validate 가 대조하는 내부 값이라 파이프라인에는
    필요하지만 독자는 어느 출처인지 알 수 없다. REFERENCE 에도 그 ID 가 없다.
    번호를 못 찾으면 원래 ID 를 남긴다. 조용히 지우면 인용이 사라진 것처럼 보인다.
    """
    numbered = CITATION_RE.sub(
        lambda m: f"[{marks[m.group(1)]}]" if m.group(1) in marks else m.group(0), markdown)
    # 같은 출처의 서로 다른 청크를 잇달아 인용하면 [1][1] 이 된다. 지면에는 한 번이면 된다.
    return re.sub(r"(?:\[(\d+)\])(?:\[\1\])+", r"[\1]", numbered)


def _held(state: dict) -> str:
    """평가 보류를 남긴 관점이 있으면 머리말에 경고를 단다(§5 — 자료가 없는 항목은 평가 보류)."""
    held = [node for node in ASSESSMENT_NODES
            if (state.get(node) or {}).get("status") in {"partial", "failed"}]
    if not held:
        return ""
    return ("\n> ⚠️ 근거를 확보하지 못해 **평가 보류** 항목을 남긴 관점: "
            f"{', '.join(held)}\n")


def _approved(state: dict) -> dict:
    """내용 검토에서 부결된 Claim 을 뺀 Assessment 를 만든다.

    §8 — 근거를 확보하지 못한 주장은 평가 보류 항목으로 옮기고 사실 서술에서 제외한다.
    원본 State 를 고치지 않는다. Assessment 의 단독 쓰기 주체는 각 평가 노드다.
    """
    rejected = set((state.get("validation") or {}).get("rejected_claims") or [])
    approved = {}
    for node in ASSESSMENT_NODES:
        assessment = dict(state.get(node) or {})
        if assessment.get("claims"):
            assessment["claims"] = [claim for claim in assessment["claims"]
                                    if claim.get("claim_id") not in rejected]
        approved[node] = assessment
    return approved


def _claims(assessment: dict) -> str:
    """Claim 을 본문으로 편다.

    §6 — *공개 자료에서 확인한 사실과 그 사실에서 추론한 내용을 구분한다.* 사실이 기본이라
    따로 표시하지 않고, 추론·가설만 전제와 함께 표시한다. 모든 문단에 "[사실]" 을 붙이면
    구분이 아니라 소음이 된다.
    """
    claims = assessment.get("claims") or []
    if not claims:
        return "- 승인된 주장 없음. 근거 공백은 §6 을 참고한다."

    blocks = []
    for claim in claims:
        body = (claim.get("text") or "").strip()
        kind = claim.get("kind", "")
        if kind != "fact":
            note = KIND_LABEL.get(kind, kind)
            explanation = (claim.get("explanation") or "").strip()
            # 모델이 "전제는 ..." 으로 시작하는 경우가 잦다. "전제: 전제는" 이 되지 않게 한다.
            for prefix in ("전제는 ", "전제: ", "전제 "):
                if explanation.startswith(prefix):
                    explanation = explanation[len(prefix):].lstrip()
                    break
            body += f"\n\n> **{note}** — 실측 결과가 아니다."
            body += f" 전제: {explanation}" if explanation else ""
        blocks.append(body)
    return "\n\n".join(blocks)

def _carried(state: dict) -> str:
    """이월된 검토 판정을 밝힌다(§10 — 조사 수행 시점과 정보의 검토 범위를 명시적으로 기록).

    주장과 근거가 완전히 같을 때만 옮기지만, 사람이 **이번 실행에서** 다시 본 것은 아니다.
    """
    validation = state.get("validation") or {}
    carried = validation.get("carried_claims") or []
    if not carried:
        return ""
    verdicts = validation.get("claim_verdicts") or {}
    runs = sorted({(verdicts.get(cid) or {}).get("carried_from", "") for cid in carried} - {""})
    return (f"- 내용 검토 이월: 주장 {len(carried)}건은 이전 실행"
            f"{'(' + ', '.join(runs) + ')' if runs else ''}의 판정을 그대로 사용했다. "
            f"주장 문장과 인용 근거가 완전히 같은 경우에만 옮겼으며, 이번 실행에서 다시 검토하지 않았다.")


def _reused() -> str:
    """캐시로 재사용한 모델 응답을 밝힌다(§6 — 실제 비용은 실행 기록으로 확인한다).

    적중은 이번 실행에서 모델이 새로 판단하지 않았다는 뜻이다. 조사 시점 해석에 영향을 준다.
    """
    report = llm_report()
    hits = report.get("cache_hits") or 0
    if not hits:
        return ""
    return (f"\n- 모델 응답 재사용: {hits}건은 이전 실행과 프롬프트가 같아 캐시된 응답을 "
            f"그대로 사용했다(이번 실행에서 새로 생성 {report.get('cache_misses', 0)}건).")


def _gaps_section(state: dict) -> str:
    """§6 한계의 근거 공백. 평평하게 나열하면 수십 줄이 되므로 관점별로 묶는다.

    §9 목차표 — 6 한계는 자료 공백을 담는다. 항목 자체는 §5 의 *자료가 없는 항목은 평가
    보류로 남긴다* 를 이행한 기록이라 줄이지 않는다.
    """
    grouped: dict[str, list[str]] = {}
    for gap in state.get("gaps") or []:
        item, technology = gap.get("item", ""), gap.get("technology", "")
        named = any(alias in item for alias in TECH_ALIAS.get(technology, (technology,)) if alias)
        head = item if named else f"{TECH_LABEL.get(technology, technology)} {item}".strip()
        line = f"{head}: {gap.get('reason', '')}".strip(": ").strip()
        grouped.setdefault(gap.get("role", ""), []).append(line)

    validation = state.get("validation") or {}
    verdicts = validation.get("claim_verdicts") or {}
    for claim_id in validation.get("rejected_claims") or []:
        verdict = verdicts.get(claim_id) or {}
        grouped.setdefault("review", []).append(
            f"검토 부결: {claim_id} (검토자: {verdict.get('reviewer') or '검토자 미상'} — "
            f"{verdict.get('comment') or '사유 미기재'})")

    if not grouped:
        return "- 해당 없음"

    blocks = []
    for role in list(ROLE_LABEL) + ["review"]:
        items = grouped.get(role)
        if not items:
            continue
        label = "내용 검토" if role == "review" else ROLE_LABEL[role]
        blocks.append(f"*{label}* ({len(items)}건)\n" + _bullets(items))
    return "\n\n".join(blocks)

SUMMARY_TOP = 3     # §9 — SUMMARY 는 PDF 반 페이지 이내다. 전체 목록은 §5·§6 에 있다.


def _top(items, section) -> str:
    """앞의 몇 건만 싣고 나머지는 해당 절을 가리킨다."""
    shown = _bullets(items[:SUMMARY_TOP])
    rest = len(items) - SUMMARY_TOP
    return shown + (f"\n- 외 {rest}건은 {section} 참고" if rest > 0 else "")


def _summary(synthesis: dict) -> str:
    """LLM 없이 synthesis 의 확정 결과만 사용해 SUMMARY 를 만든다.

    §9 목차표 — SUMMARY 는 *평가 결과, 관점별 주요 차이, **중요한** 공백* 을 반 페이지
    이내로 담는다. 전부 나열하면 분량 점검에 걸려 제출본이 생성되지 않는다.
    """
    parts = []

    agreements = synthesis.get("agreements") or []
    conflicts = synthesis.get("conflicts") or []
    gaps = synthesis.get("gaps") or []

    if agreements:
        parts.append("네 관점에서 공통적으로 확인된 사항은 다음과 같다.\n"
                     + _top(agreements, "§5.1"))

    if conflicts:
        parts.append("관점별 평가가 실제로 달라진 지점은 다음과 같다.\n"
                     + _top(_conflict_lines(conflicts), "§5.2"))

    if gaps:
        parts.append("평가 과정에서 확인된 주요 근거 공백은 다음과 같다.\n"
                     + _top(gaps, "§6"))

    if not parts:
        return "확정된 종합 평가 결과가 없다."

    return "\n\n".join(parts)


def _conflict_lines(conflicts: list[dict]) -> list[str]:
    return [f"**{conflict['perspective']}** — {conflict['why']}" for conflict in conflicts]


def _conflicts(conflicts: list[dict]) -> str:
    """관점별 차이를 Markdown으로 변환한다."""
    return _bullets(_conflict_lines(conflicts)) if conflicts else "- 확인된 관점 간 상충 없음"


def report(state) -> dict:
    """검증된 State 를 §9 목차에 맞춰 Markdown 으로 조립한다. 판단 LLM 을 부르지 않는다(§6)."""
    synthesis = state["synthesis"]
    config = state.get("run_config") or {}
    domain = domain_of(state)
    approved = _approved(state)

    summary = _summary(synthesis)
    conflicts = _conflicts(synthesis.get("conflicts") or [])
    references, marks = reference.build(approved | {"run_config": config})

    md = f"""# KV cache 최적화 기술 다관점 평가

**대상 기술** TurboQuant (SW 압축) · ITME (HW 메모리 확장)
**적용 도메인** {domain}
**생성 모델** {model_name()}
{_held(state)}
## SUMMARY

{summary}

## 1. 분석 배경

KV cache는 재계산 낭비를 줄이는 장치이지만, 문맥이 길어질수록 Key·Value 텐서가
토큰 수에 비례해 증가해 가속기 HBM을 소진시킨다. 연산 병목이 메모리 병목으로
이동하는 구조다.

본 보고서는 KV cache 데이터 축소 접근인 TurboQuant와 메모리 계층 확장 접근인
ITME를 대상으로, TRL·시장성·이해관계자·도메인 네 관점에서 평가한다.
두 기술의 개별 근거를 우선 검토하며, 동일 시스템에서 두 기술을 결합했을 때의
효과는 공개 실측 자료가 확인되지 않는 한 가설로 분리한다.

## 2. 기술 선정

데이터센터/클라우드 도메인을 먼저 확정한 뒤, 동일한 KV cache 메모리 병목에
서로 다른 시스템 계층에서 접근하는 기술을 각각 선정했다.

| 접근 | 기술 | 선정 이유 |
|---|---|---|
| KV cache 데이터 축소 | TurboQuant | KV cache 양자화를 통해 저장량과 메모리 사용량을 줄이는 접근 |
| 메모리 계층 확장 | ITME | CXL-Hybrid 기반 계층적 메모리 확장으로 용량과 데이터 이동 문제에 접근 |

두 기술은 상호 배타적인 대안으로 가정하지 않는다. 결합 효과는 §5.3의 가설로만
다룬다.

## 3. 기술 개요

{_claims(approved["research"])}

## 4. 관점별 평가

### 4.1 TRL (기술성숙도)

{_claims(approved["maturity"])}

### 4.2 시장성

{_claims(approved["market"])}

### 4.3 이해관계자

{_claims(approved["stakeholder"])}

### 4.4 도메인 적용 ({domain})

{_claims(approved["domain_assessment"])}

## 5. 시사점

### 5.1 관점 간 일치

{_bullets(synthesis.get('agreements') or [])}

### 5.2 관점 간 차이 및 상충

{conflicts}

### 5.3 결합 가설 (추론 — 실측 근거 아님)

{synthesis.get('combination_hypothesis', '해당 없음')}

## 6. 한계

**확인된 근거 공백**

{_gaps_section(state)}

**조사 시점과 검토 범위**

- 조사 수행 시점: {config.get('started_at', '미상')}
- 근거 범위: 지정 Doc Pool 색인(papers_core·ecosystem·context)과 이번 실행에서 본문을
  확보한 웹 자료로 한정한다. 후보로만 조회한 자료는 근거로 쓰지 않는다.
{_carried(state)}{_reused()}
**분석의 한계**

- 본 평가는 공개된 논문·백서·사례 자료를 기반으로 하며 자체 실측 벤치마크가 아니다.
- TRL·시장성·이해관계자 평가는 공개 정보 기반 추정이므로 실제 최신 상용 배치 현황과 차이가 있을 수 있다.
- TurboQuant와 ITME의 결합 효과는 동일 시스템에서 함께 측정된 공개 자료가 확인되지 않는 한 실측 결과가 아닌 가설로 다룬다.
- GPU 벤치마크 수치는 원 논문의 보고값이며 본 프로젝트가 직접 측정한 결과가 아니다.
- 논문마다 평가 모델·하드웨어·부하 조건이 다르므로 성능 개선 배수만으로 기술 간 우열을 단정하지 않는다.
- 자동 검증은 인용 ID의 존재 여부, 실행 범위, 허용 자료 범위 등을 결정적으로 검사한다.
- Claim과 인용 근거의 내용적 적합성은 내용 검토 워크시트에서 사람이 확인한다.
- 확증 편향을 줄이기 위해 기술별 검색량을 균형 있게 유지하고, 성능 향상뿐 아니라 잔여 비용과 근거 공백도 함께 기록한다.

{references}
"""
    md = _number_citations(md, marks)

    return {
        "report": md,
        "trace": [{
            "node": "report",
            "status": "ok",
            "attempt": 1,
            "chars": len(md),
            "claims": sum(len(a.get("claims") or []) for a in approved.values()),
            "deterministic": True,
        }],
    }
