"""평가 노드 공통 헬퍼 — 담당: R1 (계약 소유)

평가 노드는 모두 `검색 → 근거 포맷 → LLM → Assessment` 로 같은 모양이라 여기 한 번만 쓴다.
각 노드는 질의와 지시문만 갖는다.

- `evaluate` — research · maturity · domain. 기술별 본문을 **구조화 출력**으로 받는다.
  자유 텍스트를 헤더 정규식으로 자르면 모델이 절 번호를 하나만 바꿔도 두 기술이 한 Claim 으로
  뭉친다(기준선 실행 `20261021-121122-d0a3d6` 의 research).
- `run_node` — market. 기술별이 아니라 `## ①②③` 항목별 절이라 텍스트로 받는다.
"""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.output.validate import CITATION_RE, describe, numeric_errors
from src.schema import Assessment, Claim, Evidence, Gap, Source
from src.tools.docs import format_chunks
from src.tools.web_store import utcnow

TECHS = ("TurboQuant", "ITME")

_RULES = """공통 규칙:
- 아래 <document> 근거에 있는 내용만 쓴다. 없는 내용을 추론으로 채우지 않는다.
- 인용은 대괄호 ID 로 문장 끝에 붙인다. 예: ... 이다 [a1b2c3d4e5f6]
- 특정 기술을 승자로 선정하거나 우열을 판정하지 않는다.
- 서로 다른 실험의 수치를 직접 비교하지 않는다. 비교 대상과 실험 조건을 함께 적는다.
- 성능 향상뿐 아니라 잔여 비용(정확도 손실, 전송 지연, 시스템 복잡도)도 함께 보고한다.
- 본문의 모든 수치는 그 문장이 인용한 근거에 있어야 한다. 지시문에 나온 수치를 근거 없이
  옮겨 적지 않는다."""

GROUND_RULES = _RULES + """
- 근거가 없는 항목은 "미확인" 이라고 쓰고, 그 항목을 마지막 "근거 공백" 목록에 적는다.

마지막 줄은 반드시 다음 형식으로 끝낸다:
근거 공백: 항목1 | 항목2      (없으면 "근거 공백: 없음")"""

STRUCTURED_RULES = _RULES + """
- 근거가 없어 확인하지 못한 항목은 본문에 쓰지 않고 gaps 에만 적는다.
- turboquant 에는 TurboQuant 만, itme 에는 ITME 만 쓴다."""

_TEMPLATE = PromptTemplate.from_template(
    "{instruction}\n\n{rules}\n\n#평가 도메인:\n{domain}\n\n#근거:\n{context}\n\n#작성:"
)


def domain_of(state) -> str:
    """평가 도메인. State 에는 domain 필드가 없고 run_config 에 있다(설계서 §7)."""
    return (state.get("run_config") or {}).get("domain") or "데이터센터/클라우드 추론 인프라"


def unique(hits) -> list[dict]:
    """검색 결과를 chunk_id 로 중복 제거한다. 처음 나온 순서를 지킨다."""
    seen: dict[str, dict] = {}
    for c in hits:
        seen.setdefault(c["chunk_id"], c)
    return list(seen.values())


def run_node(instruction: str, domain: str, context: str) -> str:
    chain = _TEMPLATE | get_llm() | StrOutputParser()
    # Gemini 3 은 content 를 블록 목록으로 준다. 파서가 str 하위 클래스로 풀어 주지만
    # State 직렬화에 순수 str 만 흐르게 한다.
    return str(chain.invoke({"instruction": instruction, "rules": GROUND_RULES,
                             "domain": domain, "context": context}))


class DraftGap(BaseModel):
    technology: Literal["TurboQuant", "ITME", "both"]
    item: str = Field(description="근거로 확인하지 못한 항목")


class Draft(BaseModel):
    turboquant: str = Field(description="TurboQuant 절 본문. 사실 문장 끝에 [12자리hex] 인용")
    itme: str = Field(description="ITME 절 본문. 사실 문장 끝에 [12자리hex] 인용")
    gaps: list[DraftGap] = Field(default_factory=list)


class DomainDraft(Draft):
    hypothesis: str = Field(default="", description=(
        "두 기술 결합 가설. 결합 실측 근거가 없는 추론임을 밝히고 전제 근거를 인용"))


def draft(instruction: str, domain: str, context: str, schema=Draft, feedback: str = ""):
    prompt = _TEMPLATE.format(instruction=instruction, rules=STRUCTURED_RULES + feedback,
                              domain=domain, context=context)
    return get_llm().with_structured_output(schema).invoke(prompt)


NUMBER_FEEDBACK = ("\n\n#이전 초안의 오류 — 아래 수치는 인용한 근거에 없거나 조건이 빠졌다. "
                   "근거에 있는 수치만 조건과 함께 쓰고, 없으면 그 문장을 빼고 근거 공백에 적는다:\n")


def number_problems(sections, by_id, premises=()) -> list[dict]:
    """각 절의 수치를 그 절이 인용한 청크 전문과 대조한다(validate 와 같은 검사).

    절은 (본문, 기술) 이다 — 조건어 규칙이 기술별이다. `premises` 는 인용이 없으면 본문 전체
    인용을 전제로 삼는 절(결합 가설)이다. 최종 Claim 이 그렇게 만들어지므로 같은 근거로 미리
    검사해야 재작성 기회를 쓸 수 있다.
    """
    everything = {c: by_id[c]["text"] for t, _ in (*sections, *premises)
                  for c in CITATION_RE.findall(t) if c in by_id}
    problems = []
    for text, technology in sections:
        quotes = {c: by_id[c]["text"] for c in CITATION_RE.findall(text) if c in by_id}
        problems += numeric_errors(text, quotes, technology) if quotes else []
    for text, technology in premises:
        quotes = {c: by_id[c]["text"] for c in CITATION_RE.findall(text) if c in by_id} or everything
        problems += numeric_errors(text, quotes, technology) if quotes else []
    return problems


def _number_problems(result, by_id) -> list[dict]:
    hypothesis = getattr(result, "hypothesis", "")
    return number_problems([(result.turboquant, "TurboQuant"), (result.itme, "ITME")], by_id,
                           premises=[(hypothesis, "both")] if hypothesis.strip() else [])


def _try_draft(*args, **kwargs):
    """구조화 출력이 잘리거나 형식이 틀리면 예외 대신 (None, 사유) 를 돌려준다."""
    try:
        result = draft(*args, **kwargs)
    except Exception as exc:             # 파서·API 오류 모두 — 노드가 죽으면 실행 전체가 죽는다
        return None, f"{type(exc).__name__}: {exc}"[:300]
    return (result, None) if result is not None else (None, "빈 구조화 출력")


def register(cited, by_id, run_id, role) -> tuple[dict, dict]:
    """본문이 실제로 인용한 청크만 Source·Evidence 로 올린다 — 인용 사슬을 닫는다.

    quote 는 청크 전문이다. 잘라 두면 수치 검증과 사람의 검토가 원문 일부만 본다.
    """
    sources, evidence = {}, {}
    for cid in cited:
        c = by_id[cid]
        sid = c.get("source_id") or c.get("source", "unknown_source")
        collection = c["collection"]
        sources.setdefault(sid, Source(source_id=sid, run_id=run_id, collection=collection,
                                       allowed_uses=[role], title=c.get("title", sid)))
        evidence[cid] = Evidence(evidence_id=cid, source_id=sid, run_id=run_id,
                                 collection=collection, allowed_uses=[role],
                                 quote=c.get("text", ""), location=f"p.{c.get('page', 1)}")
    return sources, evidence


def _failed(node, role, chunks, attempt, error, extra_trace) -> dict:
    """생성이 두 번 다 실패했다. 근거 없이 Claim 을 만들지 않고 failed 로 남긴다(계약 ①)."""
    assessment = Assessment(status="failed", claims=[], sources=[], evidence=[], gaps=[
        Gap(role=role, technology=t, item=f"{t} 평가 생성 실패", reason=error) for t in TECHS])
    return {node: assessment.model_dump(), "trace": [{
        "node": node, "status": "failed", "attempt": attempt, "error": error,
        "timestamp": utcnow(), "chunks": len(chunks),
        **(extra_trace or {})}]}


def evaluate(*, node: str, role: str, run_id: str, domain: str, instruction: str,
             chunks: list[dict], gap_reason: str, schema=Draft, extra_trace=None) -> dict:
    """근거 청크로 초안을 받아 Assessment 와 trace 를 반환한다.

    근거에 없는 수치가 나오면 문제 문장을 짚어 **한 번만** 다시 쓰게 한다(부록 A — 재시도는
    노드 내부의 최대 1회). 그래도 남으면 그대로 내보내고 `review` 의 검증이 실행을 막는다.
    """
    by_id = {c["chunk_id"]: c for c in chunks}
    context = format_chunks(chunks)
    result, error = _try_draft(instruction, domain, context, schema)
    problems, attempt = (_number_problems(result, by_id) if result else []), 1
    if result is None or problems:
        # 재시도는 한 번뿐이다 — 생성 실패와 수치 오류가 같은 기회를 쓴다(부록 A)
        feedback = NUMBER_FEEDBACK + describe(problems) if problems else ""
        retried, error = _try_draft(instruction, domain, context, schema, feedback=feedback)
        result = retried or result
        problems, attempt = (_number_problems(result, by_id) if result else []), 2
    if result is None:
        return _failed(node, role, chunks, attempt, error, extra_trace)

    parts = [("tq", "TurboQuant", "fact", result.turboquant),
             ("itme", "ITME", "fact", result.itme)]
    if getattr(result, "hypothesis", "").strip():
        parts.append(("hypo", "both", "hypothesis", result.hypothesis))

    found = [c for *_, text in parts for c in CITATION_RE.findall(text)]
    valid = [c for c in found if c in by_id]
    invalid = sorted({c for c in found if c not in by_id})
    cited = sorted(set(valid))

    sources, evidence = register(cited, by_id, run_id, role)

    # 기술이 확보됐다 = 그 기술의 1차 원문(scope=direct)을 인용했다. 비교군 논문은 세지 않는다.
    covered = {t for cid in cited for t in by_id[cid].get("applies_to", [])
               if by_id[cid].get("scope") == "direct"}
    if valid and not invalid:
        status = "completed" if covered >= set(TECHS) else "partial"
    else:
        status = "failed"

    gaps = [Gap(role=role, technology=g.technology, item=g.item, reason=gap_reason)
            for g in result.gaps]
    claims = []
    if status != "failed":                       # failed 면 claims=[] (계약 ①)
        for key, technology, kind, text in parts:
            if not text.strip():
                continue
            ids = list(dict.fromkeys(c for c in CITATION_RE.findall(text) if c in by_id))
            if kind == "hypothesis" and not ids:
                ids = list(cited)    # 가설의 전제는 두 절의 근거 전체다(§6)
            if not ids:
                # §6 — 근거가 없는 항목은 Claim 이 아니라 Gap 이다.
                gaps.append(Gap(role=role, technology=technology, item=f"{technology} 절 인용 근거 미확보",
                                reason="본문 해당 절에 유효한 인용 ID 가 없다"))
                continue
            claims.append(Claim(
                claim_id=f"claim_{role}_{key}_{run_id[:8]}", text=text.strip(),
                technology=technology, kind=kind, evidence_ids=ids,
                explanation=("두 기술의 개별 근거에서 도출한 결합 가설이며 결합 실측 근거가 아니다."
                             if kind == "hypothesis" else "")))
    for tech in TECHS:
        if tech not in covered:
            gaps.append(Gap(role=role, technology=tech, item=f"{tech} 1차 원문 인용 미확보",
                            reason="scope=direct 청크 인용 부재"))

    assessment = Assessment(status=status, claims=claims, sources=list(sources.values()),
                            evidence=list(evidence.values()), gaps=gaps)
    return {node: assessment.model_dump(), "trace": [{
        "node": node, "status": status, "attempt": attempt,
        "timestamp": utcnow(), "chunks": len(chunks),
        "citations": len(valid), "invalid_citations": len(invalid),
        "number_problems": len(problems), "covered_techs": sorted(covered),
        **(extra_trace or {})}]}
