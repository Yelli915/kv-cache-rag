"""시장성 평가 — 담당: R4 | ecosystem RAG (설계서 §5·§6)

시장 근거는 논문 본문에 없다. 그래서 근거를 ecosystem 컬렉션으로 색인해서 RAG 로 찾는다.
papers_core 를 조회하면 의미적으로 가까운 기술 서술이 시장 근거 자리에 놓이므로
`bind_document_search("market")` 로 컬렉션을 코드에서 고정한다(§3).

§6 대로 Claim 단위로 문장·대상 기술·종류·evidence_ids 를 묶어 Assessment 를 반환한다.
근거가 없는 항목은 Claim 으로 만들지 않고 Gap 으로 기록한다.
"""

import re
from datetime import datetime, timezone

from src.agents.common import NUMBER_FEEDBACK, number_problems, register, run_node
from src.output.validate import CITATION_RE, describe
from src.schema import Assessment, Claim, Gap
from src.tools.docs import format_chunks
from src.tools.docs import bind_document_search

INSTRUCTION = """너는 시장성 평가 담당이다. 아래 세 절의 제목을 **그대로** 쓰고 각각 채운다.

## ① 시장 규모/성장성
정량 리포트가 있으면 그대로 인용, 없으면 "미확인".

## ② 상용화/채택 현황
**"발표됨" 과 "배포 확인됨" 을 반드시 구분한다.**
발표 자료만 있으면 "발표 단계, 배포 확인 미확인" 으로 적는다.

## ③ 생태계 지지
지원 프레임워크, 표준화 동향.
**CXL 제품·PoC 근거와 ITME 채택 근거를 분리한다.** 삼성·SK하이닉스의 CXL 제품이
존재한다는 사실은 ITME 아키텍처가 채택되었다는 근거가 아니다. 해당 근거가 LLM 추론
워크로드 특화인지 개별 확인해 적고, 불명이면 "일반 CXL 근거이며 추론 특화 여부 미확인".

논문 성능 수치와 시장 매출을 섞지 않는다."""

TECHS = ["TurboQuant", "ITME"]
QUERIES = [
    ("시장 규모 성장성 추론 메모리 비용", "both"),
    ("상용화 제품 출시 발표 실제 배포 채택", "both"),
    ("서빙 프레임워크 지원 현황 표준화 동향", "both"),
]
SECTION_RE = re.compile(r"^##\s*([①②③])\s*(.*)$", re.MULTILINE)
SECTION_LABELS = {"①": "시장 규모/성장성", "②": "상용화/채택 현황", "③": "생태계 지지"}


def _split_sections(text: str) -> dict[str, str]:
    """`## ① ...` 절 단위로 나눈다. 제목을 못 찾으면 전체를 한 덩어리로 둔다."""
    marks = list(SECTION_RE.finditer(text))
    if not marks:
        return {}
    bounds = [m.start() for m in marks] + [len(text)]
    return {m.group(1): text[bounds[i]:bounds[i + 1]].strip()
            for i, m in enumerate(marks)}


def _gaps_from_text(text: str, reason: str) -> list[Gap]:
    """본문 마지막 줄의 '근거 공백: ...' 을 Gap 으로 바꾼다."""
    match = re.search(r"근거\s*공백\s*:\s*(.+)\s*$", text, re.MULTILINE)
    if not match or match.group(1).strip() in ("", "없음", "-", "None"):
        return []
    gaps = []
    for item in (g.strip() for g in match.group(1).split("|") if g.strip()):
        lower = item.lower()
        technology = "both"
        if "turboquant" in lower and "itme" not in lower:
            technology = "TurboQuant"
        elif "itme" in lower and "turboquant" not in lower:
            technology = "ITME"
        gaps.append(Gap(role="market", technology=technology, item=item, reason=reason))
    return gaps


def market(state) -> dict:
    run_id = state.get("run_id", "default_run")
    domain = (state.get("run_config") or {}).get("domain", "데이터센터/클라우드")
    search = bind_document_search("market")      # ecosystem 고정 (§3)

    chunks, by_id = [], {}
    for query, technology in QUERIES:
        for chunk in search.invoke({"query": query, "technology": technology, "top_k": 5}):
            cid = chunk.get("chunk_id")
            if cid and cid not in by_id:
                by_id[cid] = chunk
                chunks.append(chunk)

    if not chunks:
        # §5 "자료가 없으면 미확인" — 근거 0건으로 시장성을 생성하지 않는다.
        assessment = Assessment(status="failed", claims=[], sources=[], evidence=[], gaps=[
            Gap(role="market", technology=technology, item="시장 근거 미확보",
                reason="ecosystem 컬렉션 검색 결과 0건") for technology in TECHS])
        return {"market": assessment.model_dump(),
                "trace": [{"node": "market", "status": "failed", "attempt": 1,
                           "timestamp": datetime.now(timezone.utc).isoformat(), "chunks": 0}]}

    context = format_chunks(chunks)
    text = run_node(INSTRUCTION, domain, context)
    # 다른 평가 노드와 같은 수치 검사·1회 재작성(부록 A). 없으면 review 검증에서 바로 실패한다.
    problems, attempt = number_problems([(text, "both")], by_id), 1
    if problems:
        text = run_node(INSTRUCTION + NUMBER_FEEDBACK + describe(problems), domain, context)
        problems, attempt = number_problems([(text, "both")], by_id), 2

    found = CITATION_RE.findall(text)
    valid = [cid for cid in found if cid in by_id]
    invalid = [cid for cid in found if cid not in by_id]
    cited = set(valid)

    sources, evidence = register(cited, by_id, run_id, "market")

    covered = {tech for cid in cited for tech in by_id[cid].get("applies_to", [])}
    if valid and not invalid and len(covered) == len(TECHS):
        status = "completed"
    elif valid and not invalid:
        status = "partial"
    else:
        status = "failed"

    claims, gaps = [], _gaps_from_text(text, "ecosystem 색인 내 근거 미확인")
    if status != "failed":
        sections = _split_sections(text) or {"①": text}
        for mark, body in sections.items():
            body_cited = list(dict.fromkeys(c for c in CITATION_RE.findall(body) if c in cited))
            label = SECTION_LABELS.get(mark, "시장성")
            if not body_cited:
                # §6 — 근거가 없는 항목은 Claim 이 아니라 Gap 이다.
                gaps.append(Gap(role="market", technology="both", item=label,
                                reason="본문에 인용된 ecosystem 근거 없음"))
                continue
            claims.append(Claim(
                claim_id=f"claim_market_{mark}_{run_id[:8]}", text=body,
                technology="both", kind="fact", evidence_ids=body_cited))

    for tech in TECHS:
        if tech not in covered:
            gaps.append(Gap(role="market", technology=tech,
                            item=f"{tech} 시장 근거 인용 미확보",
                            reason="ecosystem 색인 내 해당 기술 청크 인용 부재"))

    assessment = Assessment(status=status, claims=claims, sources=list(sources.values()),
                            evidence=list(evidence.values()), gaps=gaps)
    return {
        "market": assessment.model_dump(),
        "trace": [{"node": "market", "status": status, "attempt": attempt,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "chunks": len(chunks), "citations": len(valid),
                   "number_problems": len(problems),
                   "invalid_citations": len(invalid), "covered_techs": sorted(covered)}],
    }
