"""그래프 배선 검증용 mock 노드 — 담당: R1 (`tests/test_r1_graph.py`, `scripts/r1_smoke.py`)

형식은 `src/schema.py` 계약 그대로다. LLM·검색을 타지 않으므로 실행 경로 검증에만 쓴다.
실물 노드가 맞물리는지는 `tests/test_integration_graph.py` 가 본다.
여기서 나오는 값은 합성 데이터이며 평가 결과가 아니다.
"""


def assessment(node, run_id, *, status="completed", source_id="web-mock-1", quote="합성 인용"):
    source = {"source_id": source_id, "run_id": run_id, "collection": "web",
              "allowed_uses": ["stakeholder"], "title": f"{source_id} 합성 출처",
              "url": f"https://example.org/{source_id}"}
    evidence = {"evidence_id": f"e-{source_id}", "source_id": source_id, "run_id": run_id,
                "collection": "web", "quote": quote, "location": "paragraph:1",
                "allowed_uses": ["stakeholder"]}
    claim = {"claim_id": f"claim-{node}", "text": f"{node} 합성 주장", "technology": "ITME",
             "kind": "fact", "evidence_ids": [evidence["evidence_id"]]}
    gaps = [] if status == "completed" else [
        {"role": "market" if node == "market" else "stakeholder", "technology": "TurboQuant",
         "item": "합성 공백", "reason": "합성 데이터라 근거가 없다"}]
    return {"claims": [] if status == "failed" else [claim],
            "sources": [source], "evidence": [evidence], "gaps": gaps, "status": status}


def node(name, **kw):
    """자기 결과 키 하나와 trace 만 쓰는 평가 노드."""
    def fn(state):
        return {name: assessment(name, state["run_id"], **kw),
                "trace": [{"node": name, "status": "ok"}]}
    return fn


def setup(state):
    from src.graph import run_dir
    run_dir(state).mkdir(parents=True, exist_ok=True)   # 실물 setup 과 같은 책임
    return {"run_config": state["run_config"] | {"sources": [{"id": "mock-paper"}]},
            "trace": [{"node": "setup", "status": "ok"}]}


def synthesis(state):
    """종합. gaps 는 합류분에 자기 공백을 순차 병합한다(설계서 §7)."""
    return {"synthesis": {"agreements": ["합성 일치"],
                          "conflicts": [{"perspective": "TRL", "why": "합성"}],
                          "gaps": [g["item"] for g in state["gaps"]],
                          "combination_hypothesis": "합성 가설"},
            "gaps": state["gaps"] + [{"role": "synthesis", "technology": "ITME",
                                      "item": "결합 실측", "reason": "공개된 결합 실험 없음"}],
            "trace": [{"node": "synthesis", "status": "ok"}]}


def review(status="passed", errors=()):
    """내용 검토. validation 과 review_status 를 쓴다."""
    def fn(state):
        return {"validation": {"errors": list(errors), "reviewer": "합성"},
                "review_status": status,
                "trace": [{"node": "review", "status": "ok"}]}
    return fn


def report(state):
    return {"report": f"# 합성 보고서\n\n인용 {len(state['evidence_registry'])}건",
            "trace": [{"node": "report", "status": "ok"}]}


def publish(state):
    from src.graph import run_dir
    path = run_dir(state) / "report.md"
    path.write_text(state["report"], encoding="utf-8")
    return {"report_paths": [str(path)], "trace": [{"node": "publish", "status": "ok"}]}


def all_nodes(*, statuses=None, review_status="passed", review_errors=(), **node_kw):
    """다섯 평가 노드 + 나머지를 한꺼번에 mock 으로 만든다."""
    statuses = statuses or {}
    nodes = {"setup": setup, "synthesis": synthesis, "report": report, "publish": publish,
             "review": review(review_status, review_errors),
             "research": node("research", status=statuses.get("research", "completed"))}
    for name in ("maturity", "market", "stakeholder", "domain_assessment"):
        nodes[name] = node(name, status=statuses.get(name, "completed"), **node_kw.get(name, {}))
    return nodes
