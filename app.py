"""실행 진입점 — 담당: R1

    uv run python app.py
    uv run python app.py --domain "데이터센터/클라우드 (대규모 동시성, 비용 민감)"
    uv run python app.py --resume 20260922-141233-a1b2c3   # 검토 대기 초안 재개

실행 하나가 `runs/<run_id>/` 하나를 소유한다. R3 의 웹 스냅샷도 그 아래에 쌓이므로
run_id 를 발급하는 것이 이 파일의 첫 번째 책임이다.
"""

import argparse
import json
import os
import secrets
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# 설정·매니페스트·출력 경로가 전부 저장소 루트 기준 상대경로라 작업 디렉토리를 옮긴다.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)


from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback

from src.graph import MergeConflict, build_graph, invoke, resume_state, run_dir
from src.llm import enable_cache, llm_report
from src.settings import settings
from src.tools.web_store import save_json, utcnow


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def start_run(state) -> Path:
    """실행 디렉토리를 만들고 running 을 먼저 기록한다. 중간에 죽어도 흔적이 남는다."""
    directory = run_dir(state)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "run.json", {
        "run_id": state["run_id"], "run_config": state["run_config"],
        "run_status": "running", "started_at": utcnow(),
    })
    return directory


GENERATION_NODES = ("research", "maturity", "market", "stakeholder",
                    "domain_assessment", "synthesis")


def _accounting(state, usage) -> dict:
    """설계서 §6 — 토큰 사용량과 호출 집계. 실제 비용은 이 기록으로 확인한다.

    토큰은 콜백이 센 실제 API 응답이라 정확하다(제공자 무관). 캐시로 재사용한 응답은
    호출이 없으므로 0 으로 잡히고, 그게 실제 비용이다. 실제 호출 수는 `cache_misses`다.
    나머지는 trace 에서 센 값이다 — 콜백은 어느 노드가 불렀는지 모른다.
    """
    events = state.get("trace") or []
    node_events = Counter(e.get("node") for e in events
                          if e.get("node") and not e.get("tool"))
    record = {
        "node_events": {n: node_events[n] for n in GENERATION_NODES if node_events[n]},
        "tool_calls": sum(1 for e in events if e.get("tool")),
        "retries": sum(1 for e in events if (e.get("attempt") or 1) > 1),
    }
    if usage is not None:
        # {model: {input_tokens, output_tokens, total_tokens, ...}} — 모델별로 합산한다
        per_model = usage.usage_metadata
        record |= {key: sum(m.get(key, 0) for m in per_model.values())
                   for key in ("input_tokens", "output_tokens", "total_tokens")}
    return record


def finish(state, status, errors=(), usage=None) -> str:
    """실행 결과를 runs/<run_id>/ 에 남긴다. 실패해도 남긴다 — 그게 증빙이다."""
    directory = run_dir(state)
    started = directory / "run.json"
    save_json(started, json.loads(started.read_text(encoding="utf-8")) | {
        "run_id": state["run_id"], "run_config": state.get("run_config", {}),
        "run_status": status, "ended_at": utcnow(), "errors": list(errors),
        # run_config.model 은 설정값이고 이쪽이 실제로 적용된 값이다(§6).
        "llm": llm_report() | _accounting(state, usage),
    })
    save_json(directory / "state.json", {k: v for k, v in state.items() if k != "trace"})
    with (directory / "trace.jsonl").open("a", encoding="utf-8") as f:   # 재개 시 이어 쓴다
        for row in state.get("trace", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="데이터센터/클라우드 (대규모 동시성, 비용 민감)")
    p.add_argument("--resume", metavar="RUN_ID", help="검토 대기로 멈춘 실행의 draft.json 을 이어서 돌린다")
    p.add_argument("--run-id", help="run_id 를 직접 지정 (재현용)")
    p.add_argument("--no-carry-review", action="store_true",
                   help="검토 판정 원장(reviews/verdicts.json)을 무시하고 전부 새로 검토한다")
    args = p.parse_args()

    load_dotenv(override=True)
    enable_cache()          # 같은 프롬프트면 같은 응답 — 재현성을 모델에 기대지 않는다
    config = settings()["run"]

    if args.resume:
        # 부록 A 의 `검토 결과 반영 후 재개` — 내용 검토부터 이어 간다. 평가·종합은 다시 돌리지 않는다.
        state = resume_state(args.resume, config["runs_dir"])
        start = "review"
    else:
        state = {"run_id": args.run_id or new_run_id(), "run_status": "running", "trace": [],
                 "run_config": {"domain": args.domain, "model": settings()["llm"],
                                "limits": settings()["limits"], "started_at": utcnow(),
                                "no_carry_review": args.no_carry_review,
                                # Evidence.quote 가 청크 전문이라 수치 검사를 할 수 있는 실행이다
                                "numeric_check": True, **config}}
        start = "setup"

    directory = start_run(state)
    print(f"[run] {state['run_id']} · {directory}")

    with get_usage_metadata_callback() as usage:
        try:
            final = invoke(build_graph(start=start), state)
        except MergeConflict as exc:
            # 근거가 서로 어긋났다. merge-errors.json 은 collect_evidence 가 이미 남겼다.
            print(f"실패: 근거 병합 오류 — {exc}")
            return finish(state, "failed", [str(exc)], usage=usage)
        except Exception as exc:
            finish(state, "failed", [repr(exc)], usage=usage)
            raise                   # 스택 트레이스를 삼키지 않는다

    status = finish(final, final.get("run_status", "failed"), usage=usage)
    pending = (final.get("validation") or {}).get("pending_claims") or []

    carried = (final.get("validation") or {}).get("carried_claims") or []
    if carried:
        print(f"검토 판정 원장에서 가져온 Claim {len(carried)}건 "
              f"(주장·근거가 완전히 같은 것만)")

    if status == "failed":
        print(f"실패: {directory/'run.json'} 확인")
    elif final.get("review_status") == "pending":
        # §8 — ID 대조만으로 자동 통과시키지 않는다. 사람이 주장과 근거를 나란히 본다.
        print(f"검토 대기 — Claim {len(pending)}건. {directory/'review.csv'} 의 "
              f"review_result(확인|부결)·reviewer 를 채운 뒤\n"
              f"  uv run python app.py --resume {state['run_id']}")
    else:
        # run_status 는 §7 정의를 따라 partial 로 남는다. 사람에게는 "보고서가 나왔고
        # 평가 보류가 몇 건 있다" 가 읽을 값이라 그렇게 적는다.
        print("완료")
        for path in final.get("report_paths") or [directory]:
            print(f"  {path}")
        gaps = final.get("gaps") or []
        if gaps:
            print(f"  평가 보류 {len(gaps)}건 — 보고서 §6 한계 참고 (run_status={status})")
        if not any(str(p).endswith(".pdf") for p in final.get("report_paths") or []):
            print(f"  제출본 PDF 미생성 — 사유는 {directory/'submission.json'}")
    return status


if __name__ == "__main__":
    main()
