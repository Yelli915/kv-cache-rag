"""REFERENCE 포맷터 — 담당: R5

원칙:
- 실제 Claim에 사용된 Evidence를 따라가 최종 출처를 추출한다(§8).
- 후보로 검색됐지만 Claim에 사용되지 않은 자료는 최종 REFERENCE에서 제외한다(§9).
- 출처마다 번호를 매긴다. 본문의 `[chunk_id]` 는 검증용 내부 ID 라 독자가 읽을 수 없다.
  report 가 이 번호로 바꿔 넣는다.
- 색인 출처의 서지정보는 `run_config.sources`(매니페스트)에서 채운다. 청크 메타데이터에는
  제목·URL·버전이 없어 source_id 만 남는다 — §9 가 요구하는 저자·연도·게재처·인용 페이지·
  URL 를 그것만으로는 채울 수 없다.
"""

from src.state import ASSESSMENT_NODES

GROUPS = [
    ("[A] Doc Pool 논문 (RAG 색인)", ["papers_core"]),
    ("[B] 풀 밖 색인 자료 (ecosystem · context)", ["ecosystem", "context"]),
]

def _used(state: dict) -> tuple[list[dict], dict]:
    """실제 Claim 이 쓴 Source(중복 제거)와 source_id -> 인용 위치(§9 — 인용 페이지를 적는다).

    후보로 검색됐지만 Claim 에 쓰이지 않은 자료는 빠진다.
    """
    sources, pages = {}, {}
    for node_name in ASSESSMENT_NODES:
        assessment = state.get(node_name) or {}
        evidence = {e["evidence_id"]: e for e in assessment.get("evidence") or [] if e.get("evidence_id")}
        used = {evidence[eid].get("source_id") for claim in assessment.get("claims") or []
                for eid in claim.get("evidence_ids") or [] if eid in evidence} - {None, ""}
        for source in assessment.get("sources") or []:
            if source.get("source_id") in used:
                sources[source["source_id"]] = source
        for item in evidence.values():
            if item.get("source_id") in used and item.get("location"):
                pages.setdefault(item["source_id"], set()).add(item["location"])
    return list(sources.values()), pages


def _manifest_by_id(state: dict) -> dict:
    """setup 이 run_config.sources 에 넣어 둔 매니페스트를 source_id 로 색인한다."""
    sources = (state.get("run_config") or {}).get("sources") or []
    return {entry.get("id"): entry for entry in sources if entry.get("id")}


def _date(value) -> str:
    """ISO 타임스탬프를 날짜까지만 남긴다. §9 는 게시일·조회일을 요구하며 초 단위는 노이즈다."""
    text = str(value or "").strip()
    return text[:10] if len(text) >= 10 and text[4] == "-" and text[7] == "-" else text


def _page_key(location: str):
    """p.5 가 p.15 뒤로 가지 않게 숫자로 정렬한다."""
    digits = "".join(c for c in str(location) if c.isdigit())
    return (int(digits) if digits else 0, str(location))


def _format_structured_sources(sources: list[dict], numbers: dict, pages: dict) -> str:
    """신형 Source 구조를 Markdown REFERENCE 로 변환한다. 번호는 본문 인용과 맞춘다."""
    lines = ["## REFERENCE", ""]

    indexed = {"papers_core", "ecosystem", "context"}
    rag_sources = [s for s in sources if s.get("collection") in indexed]
    web_sources = [s for s in sources if s.get("collection") not in indexed]

    def entry(source: dict) -> str:
        number = numbers.get(source.get("source_id"), "?")
        title = source.get("title") or source.get("source_id") or "제목 미확인"
        version = str(source.get("version") or "")
        # 논문은 게시일 대신 arXiv 버전이 연도를 담는다(§9 — "연도 ... 또는 arXiv 버전").
        # 버전을 날짜 자리에 쓰면 뒤에서 다시 적지 않는다.
        published = _date(source.get("published") or source.get("published_at"))
        head = published or version or "날짜 미확인"
        bits = [b for b in (source.get("venue") or source.get("publisher") or "",
                            version if published else "") if b]
        cited = sorted(pages.get(source.get("source_id"), []), key=_page_key)
        if cited:
            bits.append("인용 " + ", ".join(cited))
        url = source.get("url") or ""
        return (f"- [{number}] {title} ({head})"
                + (f" · {' · '.join(bits)}" if bits else "")
                + (f"  \n  {url}" if url else ""))

    for heading, collections in GROUPS:
        rows = [s for s in rag_sources if s.get("collection") in collections]
        lines += [f"### {heading}", ""]
        if not rows:
            lines += ["- 해당 없음", ""]
            continue
        lines += [entry(s) for s in sorted(rows, key=lambda i: numbers.get(i["source_id"], 0))]
        lines.append("")

    lines += ["### [C] 웹 조회 자료", ""]
    if not web_sources:
        lines.append("- 해당 없음")
    else:
        for source in sorted(web_sources, key=lambda i: numbers.get(i["source_id"], 0)):
            number = numbers.get(source.get("source_id"), "?")
            institution = (source.get("institution") or source.get("organization")
                           or source.get("source_type") or "출처 미확인")
            lines.append(
                f"- [{number}] {institution} ({_date(source.get('published_at')) or '게시일 미확인'}). "
                f"{source.get('title') or '제목 미확인'}. "
                f"조회 {_date(source.get('retrieved_at')) or '조회일 미확인'}"
                + (f"  \n  {source.get('url')}" if source.get("url") else ""))

    return "\n".join(lines)


def build(state: dict) -> tuple[str, dict]:
    """최종 REFERENCE 와 `evidence_id -> 번호` 지도를 함께 돌려준다.

    §8 — 참고문헌은 실제 포함된 Claim 의 근거에서 역으로 만든다.
    §9 — 본문에서 실제 사용한 자료만 적고 후보로만 조회한 자료는 제외한다.
    """
    manifest = _manifest_by_id(state)
    # 청크에는 제목·URL·버전이 없어 R4 가 source_id 를 title 자리에 넣는다. 서지 항목만
    # 매니페스트 값으로 덮는다(§9). run_id·collection·allowed_uses 는 Source 것을 지킨다.
    biblio = ("title", "url", "version", "publisher", "published_at", "landing")
    sources, pages = _used(state)
    used = []
    for source in sources:
        entry = manifest.get(source.get("source_id"), {})
        used.append(source | {k: entry[k] for k in biblio if entry.get(k)})

    order = sorted(used, key=lambda s: ({"papers_core": 0, "ecosystem": 1, "context": 1}
                                        .get(s.get("collection"), 2),
                                        s.get("title") or s.get("source_id") or ""))
    numbers = {s["source_id"]: i for i, s in enumerate(order, 1)}

    marks = {}
    for node_name in ASSESSMENT_NODES:
        assessment = state.get(node_name) or {}
        for evidence in assessment.get("evidence") or []:
            number = numbers.get(evidence.get("source_id"))
            if number:
                marks[evidence["evidence_id"]] = number

    return _format_structured_sources(order, numbers, pages), marks
