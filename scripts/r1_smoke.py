"""R1 실행 증빙 — 네 경로를 전 노드 mock 으로 돌린다.

    uv run python -m scripts.r1_smoke --root /tmp/kv-r1-smoke

그래프 배선·병합·run_status 판정·저장(`app.finish`)은 실물이다. 평가 내용만 합성이므로
여기 남는 보고서·주장은 실제 기술 평가가 아니다. LLM·웹·색인은 타지 않는다.
mock 노드는 tests/mock_nodes.py 와 같은 것을 쓴다 — 테스트와 증빙이 갈라지지 않게.
"""

import argparse
import sys
from pathlib import Path

# 'python scripts/r1_smoke.py' 로 직접 실행해도 src·app·tests 를 찾게 한다.
# app 을 import 하면 작업 디렉토리가 저장소 루트로 옮겨지므로 --root 상대경로는 루트 기준이다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import finish, start_run
from src.graph import MergeConflict, build_graph, invoke, resume_state
from src.tools.web_store import save_json
from tests import mock_nodes


def initial(root, run_id):
    return {"run_id": run_id, "run_status": "running", "trace": [],
            "run_config": {"domain": "데이터센터/클라우드 (합성 데이터)", "runs_dir": str(root),
                           "technologies": ["TurboQuant", "ITME"]}}


def execute(state, *, start="setup", **mock_kw):
    start_run(state)
    final = invoke(build_graph(start=start, **mock_nodes.all_nodes(**mock_kw)), state)
    return final, finish(final, final["run_status"])


def run(root):
    root = Path(root)      # resolve 하지 않는다 — 증빙에 로컬 절대경로를 남기지 않기 위해
    if root.exists():
        raise ValueError("새 root 를 쓸 것 — 이전 증빙을 덮어쓰지 않는다")
    root.mkdir(parents=True)
    cases = []

    final, status = execute(initial(root, "r1-completed"))
    cases.append({"case": "completed", "passed": status == "completed",
                  "run_status": status, "report_paths": final.get("report_paths", []),
                  "sources": len(final["source_registry"]), "gaps": len(final["gaps"])})

    final, status = execute(initial(root, "r1-review-pending"), review_status="pending")
    draft = root / "r1-review-pending" / "draft.json"
    cases.append({"case": "review_pending", "passed": status == "partial" and draft.exists(),
                  "run_status": status, "draft": draft.exists(), "report": bool(final.get("report"))})

    # 사람이 초안을 검토하고 review 를 고친 뒤 재개하는 경로.
    resumed = resume_state("r1-review-pending", root)
    resumed |= {"validation": {"errors": [], "reviewer": "합성 검토"}, "review_status": "passed"}
    final, status = execute(resumed, start="review")
    cases.append({"case": "resume_after_review", "passed": status == "completed",
                  "run_status": status, "nodes": [t["node"] for t in final["trace"]]})

    final, status = execute(initial(root, "r1-review-errors"), review_errors=["없는 인용 ID"])
    cases.append({"case": "unresolved_errors", "passed": status == "failed",
                  "run_status": status, "report": bool(final.get("report"))})

    state = initial(root, "r1-merge-conflict")
    try:
        execute(state, market={"quote": "같은 ID 에 다른 인용문"})
        cases.append({"case": "merge_conflict", "passed": False, "run_status": "?"})
    except MergeConflict as exc:
        status = finish(state, "failed", [str(exc)])
        cases.append({"case": "merge_conflict", "passed": status == "failed",
                      "run_status": status, "errors": str(exc).split("; ")})

    save_json(root / "checks.json", {"synthetic_only": True, "cases": cases})
    for case in cases:
        print(("  ok " if case["passed"] else "FAIL ") + case["case"] + f" · {case['run_status']}")
    if not all(c["passed"] for c in cases):
        raise SystemExit("smoke 실패")
    return cases


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)
