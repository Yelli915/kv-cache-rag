"""검토용 worksheet 를 읽기 좋게 펼친다 — 담당: R5

    uv run python -m scripts.review_reader 20260922-122252-d94611
    uv run python -m scripts.review_reader <run_id> --pending   # 미판정만
    uv run python -m scripts.review_reader <run_id> --claim claim_market_②_20260922

**읽기 전용이다.** 판정은 사람이 review.csv 에 직접 넣는다. 설계서 §8 — 이 과정은
ID 대조만으로 자동 통과시키지 않으며, 팀원이 주장과 근거를 나란히 보고 확인한다.
"""

import argparse
import csv
import os
import shutil
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.output.review import read_verdicts
from src.settings import settings

CHECKLIST = (
    "성능 수치의 단위·비교 기준선·실험 조건 / TRL 단계의 근거 / "
    "직접 채택과 인접 생태계 자료의 구분 (§8)"
)


def _wrap(text, width, indent=""):
    body = "\n\n".join(
        textwrap.fill(" ".join(block.split()), width, initial_indent=indent,
                      subsequent_indent=indent)
        for block in (text or "").split("\n\n") if block.strip()
    )
    return body or indent + "(내용 없음)"


def _claims(rows):
    """행을 claim_id 순서대로 묶는다. 한 Claim 이 근거 수만큼 행을 갖는다."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["claim_id"], []).append(row)
    return grouped


def show(run_id, *, runs_dir=None, only_pending=False, claim=None, width=None):
    runs_dir = Path(runs_dir or settings()["run"]["runs_dir"])
    path = runs_dir / run_id / "review.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    verdicts = read_verdicts(path)
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 100)

    grouped = _claims(rows)
    shown = 0
    for claim_id, claim_rows in grouped.items():
        if claim and claim != claim_id:
            continue
        verdict = verdicts.get(claim_id)
        if only_pending and verdict:
            continue
        shown += 1
        head = claim_rows[0]
        mark = f"{verdict['verdict']} · {verdict['reviewer'] or '검토자 미상'}" if verdict else "미판정"

        print("═" * width)
        print(f"[{shown}/{len(grouped)}] {claim_id}")
        print(f"  {head['node']} · {head['technology']} · {head['claim_kind']} · {mark}")
        print("─" * width)
        print(_wrap(head["claim_text"], width))
        for index, row in enumerate(claim_rows, 1):
            if not row["evidence_id"]:
                print("\n  근거 없음 — §6 에 따라 Gap 이어야 한다")
                continue
            print(f"\n  근거 {index}/{len(claim_rows)} · {row['source_id']} "
                  f"· {row['location'] or '위치 미상'} · {row['evidence_id']}")
            print(_wrap(row["evidence_quote"], width - 4, "    "))
        print()

    if not shown:
        print("표시할 Claim 이 없다." + (" 미판정 항목이 없다." if only_pending else ""))
        return 0

    print("═" * width)
    print(f"확인할 것: {CHECKLIST}")
    print(f"판정은 {path} 의 review_result(확인|부결)·reviewer 에 직접 넣는다. "
          f"한 Claim 당 한 행이면 된다.")
    return shown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--runs-dir")
    p.add_argument("--pending", action="store_true", help="아직 판정하지 않은 Claim 만")
    p.add_argument("--claim", help="claim_id 하나만")
    args = p.parse_args()
    show(args.run_id, runs_dir=args.runs_dir, only_pending=args.pending, claim=args.claim)


if __name__ == "__main__":
    main()
