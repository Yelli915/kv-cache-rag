"""그래프 배선 — 담당: R1 (설계서 §8, 부록 A 그래프 소스 그대로)

    setup → research → [maturity ‖ market ‖ stakeholder ‖ domain_assessment]
          → collect_evidence → synthesis → review → final_check
                                             ↑            ├ 검토 대기  → save_draft ─┐
                                             └────────────────────────────────────────┘
                                                          ├ 미해결 오류 → fail  → END
                                                          └ 통과       → report → publish → END

**재시도는 노드 내부의 최대 1회 처리다**(부록 A). 그래서 조건부 엣지는 `final_check`
하나뿐이다. `final_check` 는 인용 오류와 사람의 내용 검토 완료 여부를 함께 확인한다.

검토 대기는 초안을 저장하고 **내용 검토(`review`)로 되돌아간다.** 사람이 검토 결과를
반영해야 진행되므로 `save_draft` 뒤에서 실행을 멈춘다(`interrupt_after`). 재개는 고친
초안을 들고 `build_graph(start="review")` 로 다시 들어온다.

병합 오류는 분기가 아니라 `collect_evidence` 의 예외다. 같은 ID 에 다른 내용이 들어오면
어느 원문을 가리키는지 고를 근거가 없다(설계서 §7). 어긋난 근거로 종합·검토에 LLM 을
태우지 않고 실행을 끝낸다.
"""

import hashlib
import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.agents.domain import domain_assessment
from src.agents.market import market
from src.agents.maturity import maturity
from src.agents.report import report
from src.agents.research import research
from src.agents.stakeholder import stakeholder
from src.agents.synthesis import synthesis
from src.output.pdf import publish
from src.output.review import review
from src.schema import Assessment
from src.state import ReportState, run_dir
from src.tools.web_store import save_json, utcnow

FANOUT = ["maturity", "market", "stakeholder", "domain_assessment"]
# 합류 단계에서 모든 자료를 모은다(§6). 참고문헌을 실제 인용에서 역으로 만들려면
# 기술 조사 단계의 근거도 레지스트리에 있어야 한다(§9).
MERGED = ["research"] + FANOUT


class MergeConflict(Exception):
    """같은 ID 에 다른 내용이 들어왔다. 인용이 어느 원문을 가리키는지 알 수 없어 실행을 끝낸다."""


def event(node, status="ok", **fields) -> dict:
    return dict(node=node, status=status, attempt=1, timestamp=utcnow(), **fields)


# ── R1 소유 노드 ────────────────────────────────────────────────────────────

def verify_corpus(manifest: dict) -> list[str]:
    """적재 전에 원문 SHA-256 을 다시 확인한다(설계서 §3).

    원문이 바뀌면 청크가 바뀌고 chunk_id 가 전부 달라진다. 조용히 넘어가면 이전 보고서의
    인용이 가리키던 원문이 사라지고, 두 실행이 왜 다른지 알 수 없게 된다.
    """
    problems = []
    for entry in manifest.get("sources", []):
        local, expected = entry.get("local_path"), entry.get("sha256")
        if not local or not expected:
            continue
        path = Path(local)
        if not path.exists():
            problems.append(f"{entry['id']}: 원문이 없다 ({local})")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"{entry['id']}: SHA-256 불일치 ({local})\n"
                            f"    매니페스트 {expected}\n    실제       {actual}")
    return problems


def setup(state) -> dict:
    """설정 확인과 논문 적재. 적재한 자료 버전을 run_config 에 남긴다."""
    if not state.get("run_id"):
        raise ValueError("run_id 가 없다. app.py 가 실행마다 발급한다")
    config = state.get("run_config") or {}
    for key in ("domain", "technologies"):
        if not config.get(key):
            raise ValueError(f"run_config.{key} 가 없다")
    run_dir(state).mkdir(parents=True, exist_ok=True)

    from src.rag.index import build, load_manifest   # 임베딩을 끌고 오므로 호출 시점에 import

    problems = verify_corpus(load_manifest())
    if problems:
        for line in problems:
            print(f"[setup] ⚠ {line}")
        raise ValueError(
            f"원문 {len(problems)}건이 매니페스트와 다르다. 의도한 갱신이면 "
            "`uv run python scripts/prepare_sources.py --refresh` 로 매니페스트를 다시 만들고 "
            "goldenset 재라벨링을 검토할 것")

    manifest = build()["manifest"]
    return {"run_config": config | {"sources": manifest},
            "trace": [event("setup", sources=len(manifest), run_id=state["run_id"])]}


def _merge_one(store: dict, item: dict, id_field: str, node: str) -> list[str]:
    """ID 하나를 레지스트리에 합친다. 같은 ID 에 다른 내용이 오면 병합 오류다(§7).

    `allowed_uses` 만 예외로 합집합을 취한다. §7 은 ID 를 원문 URL·버전 또는 본문
    해시·위치로 만들라고 한다 — 즉 같은 ID 면 같은 원문이다. 반면 허용 용도는 원문의
    속성이 아니라 **사용 권한**이라, 같은 청크를 TRL 노드와 시장성 노드가 각각 인용하면
    자기 역할만 적어 온다. 이걸 내용 불일치로 보면 정상 실행이 전부 병합 오류가 된다.
    노드별 권한 검사는 각 Assessment 안에서 이미 끝났다(§8).
    """
    ident = item[id_field]
    kept = store.get(ident)
    if kept is None:
        store[ident] = dict(item)
        return []

    without_uses = {k: v for k, v in item.items() if k != "allowed_uses"}
    if {k: v for k, v in kept.items() if k != "allowed_uses"} != without_uses:
        return [f"{node}: {ident} 가 기존 내용과 다르다"]

    kept["allowed_uses"] = sorted({*(kept.get("allowed_uses") or []),
                                   *(item.get("allowed_uses") or [])})
    return []


def collect_evidence(state) -> dict:
    """다섯 Assessment 의 출처와 근거를 한 번에 병합한다. **검증은 여기서 한 번만.**

    같은 ID 에 다른 내용이 오면 병합 오류다(§7). 어느 쪽이 맞는지 고를 근거가 없고,
    조용히 덮어쓰면 내용 검토가 엉뚱한 원문을 대조하게 된다.
    """
    sources, evidence, gaps, errors = {}, {}, [], []
    for name in MERGED:
        raw = state.get(name)
        if not raw:
            errors.append(f"{name}: Assessment 가 없다")
            continue
        try:
            assessment = Assessment.model_validate(raw).model_dump()
        except ValidationError as exc:
            errors.append(f"{name}: 구조 오류 — {exc.error_count()}건")
            continue
        for key, store, id_field in (("sources", sources, "source_id"),
                                     ("evidence", evidence, "evidence_id")):
            for item in assessment[key]:
                errors.extend(_merge_one(store, item, id_field, name))
        gaps.extend(assessment["gaps"])

    if errors:
        save_json(run_dir(state) / "merge-errors.json", {"errors": errors})
        raise MergeConflict("; ".join(errors))

    return {"source_registry": sources, "evidence_registry": evidence, "gaps": gaps,
            "trace": [event("collect_evidence", sources=len(sources),
                            evidence=len(evidence), gaps=len(gaps))]}


def final_check(state) -> dict:
    """인용 오류와 사람의 내용 검토 완료 여부를 함께 확인하고 run_status 를 판정한다(부록 A)."""
    statuses = {state.get(name, {}).get("status") for name in MERGED}
    errors = (state.get("validation") or {}).get("errors")
    if "failed" in statuses or errors:
        status = "failed"          # 구조 오류나 미해결 인용 오류가 남은 경우(§7)
    elif state.get("review_status") == "pending" or "partial" in statuses:
        status = "partial"         # 일부 항목을 평가 보류로 남겼거나 내용 검토가 끝나지 않았다
    else:
        status = "completed"
    return {"run_status": status,
            "trace": [event("final_check", status, review_status=state.get("review_status", ""))]}


def route_after_final(state) -> str:
    if state["run_status"] == "failed":
        return "fail"
    if state.get("review_status") == "pending":
        return "save_draft"
    return "report"


def save_draft(state) -> dict:
    """검토용 초안만 저장한다(§7). 제출본과 다른 경로에 둔다(§8).

    다음 엣지는 `review` 로 돌아가지만 사람의 검토 결과가 있어야 진행되므로
    여기서 실행이 멈춘다(`interrupt_after`). 재개는 `app.py --resume <run_id>`.
    """
    draft = {k: v for k, v in state.items() if k != "trace"}
    save_json(run_dir(state) / "draft.json", draft)
    return {"trace": [event("save_draft", "partial", path=str(run_dir(state) / "draft.json"))]}


def fail(state) -> dict:
    """미해결 오류. 오류와 로그를 저장하고 제출용 출력을 막는다(§8).

    `report` 를 쓰지 않는다 — 실패한 실행이 보고서 자리를 차지하면 안 된다.
    """
    validation = state.get("validation") or {}
    save_json(run_dir(state) / "validation-errors.json", validation)
    return {"trace": [event("fail", "failed", errors=validation.get("errors", []))]}


DEFAULT_NODES = {
    "setup": setup,                      # R1
    "research": research,                # R4
    "maturity": maturity,                # R4
    "market": market,                    # R4
    "stakeholder": stakeholder,          # R3
    "domain_assessment": domain_assessment,   # R4
    "collect_evidence": collect_evidence,     # R1
    "synthesis": synthesis,              # R5
    "review": review,                    # R5 — 형식 검사 + 사람의 내용 검토
    "final_check": final_check,          # R1
    "save_draft": save_draft,            # R1
    "fail": fail,                        # R1
    "report": report,                    # R5
    "publish": publish,                  # R5 — 레이아웃 확인 후 제출본 저장
}


def build_graph(start="setup", **overrides):
    """노드 함수를 이름으로 갈아끼운다. R4·R5 이행 전에는 mock 을 주입해 전 경로를 돌린다.

    `start="review"` 는 재개용이다(부록 A 의 `검토 결과 반영 후 재개`). 사람이 draft.json 의
    검토 결과를 반영해 두었으므로 평가·종합을 다시 돌리지 않고 내용 검토부터 이어 간다.
    """
    nodes = DEFAULT_NODES | overrides
    b = StateGraph(ReportState)
    for name, fn in nodes.items():
        b.add_node(name, fn)

    b.add_edge(START, start)
    b.add_edge("setup", "research")
    for name in FANOUT:
        b.add_edge("research", name)          # fan-out
    b.add_edge(FANOUT, "collect_evidence")    # 넷이 모두 끝나야 병합·종합을 시작한다(§8)
    b.add_edge("collect_evidence", "synthesis")
    b.add_edge("synthesis", "review")
    b.add_edge("review", "final_check")
    b.add_conditional_edges("final_check", route_after_final,
                            ["report", "save_draft", "fail"])
    b.add_edge("save_draft", "review")        # 검토 결과 반영 후 재개(부록 A)
    b.add_edge("report", "publish")
    b.add_edge("publish", END)
    b.add_edge("fail", END)
    # save_draft 뒤에서 멈춘다. 사람의 검토 없이 review 로 돌아가면 무한히 돈다.
    return b.compile(checkpointer=InMemorySaver(), interrupt_after=["save_draft"])


def invoke(graph, state):
    """checkpointer 를 붙였으므로 thread_id 가 필요하다. 실행 하나가 스레드 하나다."""
    thread = state.get("run_id") or "unknown"
    return graph.invoke(state, config={"configurable": {"thread_id": thread}})


def resume_state(run_id, runs_dir="runs") -> dict:
    """save_draft 가 남긴 초안을 초기 State 로 되돌린다.

    사람은 이 draft.json 의 `validation` · `review_status` 를 직접 고쳐 검토 결과를 넣는다.
    그게 재개의 입력이다. `review_status` 가 pending 이면 다시 초안 저장에서 멈춘다.
    """
    draft = json.loads((Path(runs_dir) / run_id / "draft.json").read_text(encoding="utf-8"))
    return draft | {"trace": []}
