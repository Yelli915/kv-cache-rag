"""R5 reference.py 단위 테스트 — 실제 Claim 이 쓴 근거에서 역으로 만든 REFERENCE."""

from src.output import reference


def _source(name, published):
    return {"source_id": f"source-{name}", "source_type": "official", "title": f"{name.title()} Source",
            "published_at": published, "retrieved_at": "2026-09-22",
            "url": f"https://example.com/{name}"}


def _state():
    return {"stakeholder": {
        "claims": [{"claim_id": "claim-001", "technology": "ITME", "kind": "fact",
                    "text": "실제로 사용된 주장", "evidence_ids": ["evidence-used"]}],
        "evidence": [{"evidence_id": "evidence-used", "source_id": "source-used", "quote": "인용된 원문"},
                     {"evidence_id": "evidence-unused", "source_id": "source-unused", "quote": "검색만 된 원문"}],
        "sources": [_source("used", "2026-09-01"), _source("unused", "2026-09-02")],
        "gaps": [], "status": "completed",
    }}


def test_only_used_source_is_in_the_web_section():
    """§9 — 후보로 검색됐지만 Claim 에 쓰이지 않은 자료는 REFERENCE 에서 뺀다."""
    output, _ = reference.build(_state())

    assert "## REFERENCE" in output and "### [C] 웹 조회 자료" in output
    assert "Used Source" in output and "https://example.com/used" in output
    assert "Unused Source" not in output and "https://example.com/unused" not in output


def test_duplicate_source_is_removed():
    """여러 Claim 이 같은 Source 를 써도 REFERENCE 에는 한 번만 나온다."""
    state = _state()
    state["stakeholder"]["claims"].append({
        "claim_id": "claim-002", "technology": "ITME", "kind": "inference",
        "text": "같은 출처를 사용하는 두 번째 주장", "evidence_ids": ["evidence-used"]})

    output, _ = reference.build(state)

    assert output.count("https://example.com/used") == 1


def _paper_state():
    return {
        "run_config": {"sources": [
            {"id": "itme-paper", "title": "ITME: Inference Tiered Memory Expansion",
             "url": "https://arxiv.org/pdf/2606.12556",
             "version": "arXiv:2606.12556v2 (2026-06)"},
        ]},
        "maturity": {
            "claims": [{"claim_id": "c1", "technology": "ITME", "kind": "fact",
                        "text": "본문", "evidence_ids": ["e1", "e2"]}],
            "evidence": [
                {"evidence_id": "e1", "source_id": "itme-paper", "location": "p.15"},
                {"evidence_id": "e2", "source_id": "itme-paper", "location": "p.5"},
            ],
            "sources": [{"source_id": "itme-paper", "collection": "papers_core",
                         "title": "itme-paper"}],
            "gaps": [], "status": "completed",
        },
    }


def test_paper_title_and_version_come_from_the_manifest():
    """§9 — 논문은 제목·arXiv 버전·인용 페이지·URL 을 적는다. 청크에는 이 정보가 없다."""
    output, marks = reference.build(_paper_state())

    assert "ITME: Inference Tiered Memory Expansion" in output
    assert "itme-paper (날짜 미확인)" not in output      # source_id 가 제목 자리에 오면 안 된다
    assert "(arXiv:2606.12556v2 (2026-06))" in output   # 버전이 연도를 대신한다
    assert "https://arxiv.org/pdf/2606.12556" in output
    assert marks == {"e1": 1, "e2": 1}


def test_cited_pages_sort_numerically():
    """p.5 가 p.15 뒤로 가면 안 된다."""
    assert "인용 p.5, p.15" in reference.build(_paper_state())[0]


def test_timestamps_are_trimmed_to_dates():
    state = _state()
    state["stakeholder"]["sources"][0] |= {
        "collection": "web", "published_at": "2026-04-07T15:52:54Z",
        "retrieved_at": "2026-09-22T04:55:31.105781+00:00"}

    output, _ = reference.build(state)
    assert "(2026-04-07)" in output and "조회 2026-09-22  " in output
    assert "T15:52:54Z" not in output
