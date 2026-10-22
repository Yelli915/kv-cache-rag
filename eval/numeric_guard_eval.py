"""수치 가드레일 탐지율 — 일부러 틀린 Claim 을 만들어 `validate.numeric_errors` 가 잡는지 잰다.

실행 기록의 Claim 에서 **근거로 뒷받침되는 수치**가 있는 줄을 골라 세 가지로 변조한다.

- 변조: 수치를 1.5배로 바꾼다 (원문에 없는 값)
- 조건 누락: 설계서 §2 조건어(H100·NVMe·TTFT·128GB)를 지운다
- 섞기: 다른 Claim 근거에만 있는 같은 단위 수치로 바꾼다 (다른 실험 수치 혼입)

같은 줄의 원본은 오탐 측정에 쓴다 — 원본에서 오류가 나면 오탐이다. API 키가 필요 없다.
바꾼 값이 인용 원문에 우연히 있으면 원문 대조로는 원리적으로 못 잡는다. 이 경우는 "원문에
있는 값" 으로 따로 세어 보여 준다(탐지율에서 빼서 숨기지 않는다).

    uv run python -m eval.numeric_guard_eval runs/<run_id>/state.json
    uv run python -m eval.numeric_guard_eval <state.json> --index   # quote 가 300자로 잘린 옛 기록

`--index` 는 잘린 quote 를 색인의 청크 전문으로 되돌린다(기준선 실행이 그렇다).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.output.validate import (CITATION_RE, UNITS, _NUMBER, bare_numbers, numbers,
                                 numeric_errors, qualifiers_for, segments)

NODES = ("research", "maturity", "market", "stakeholder", "domain_assessment")


def spans(text):
    """(시작, 끝, 값, 단위) — numbers() 와 같은 규칙으로 수치 표기 위치를 찾는다."""
    for unit, pattern in UNITS.items():
        for m in re.finditer(rf"{_NUMBER}\s*{pattern}", text, re.I):
            yield m.start(1), m.end(1), float(m.group(1)), unit


def supported(value, unit, quotes):
    return (value, unit) in set().union(*map(numbers, quotes)) or (
        unit != "배" and value in set().union(*map(bare_numbers, quotes)))


def claims(state, chunks):
    for node in NODES:
        assessment = state.get(node) or {}
        evidence = {e["evidence_id"]: e for e in assessment.get("evidence") or []}
        for claim in assessment.get("claims") or []:
            quotes = {i: chunks.get(i) or evidence[i].get("quote") or ""
                      for i in claim["evidence_ids"] if i in evidence}
            yield node, claim, quotes


def mutate(state, chunks):
    """(종류, 원본 줄, 변조 줄, 검사) 를 낸다. 변조 줄이 None 이면 원문에 있는 값이라 건너뛴 것.

    검사는 그 Claim 의 근거·기술로 묶은 `numeric_errors` 다 — 조건어 규칙이 기술별이다.
    """
    pool = [(v, u, cid) for _, c, q in claims(state, chunks)
            for v, u in set().union(*map(numbers, q.values())) for cid in [c["claim_id"]]]
    for _, claim, quotes in claims(state, chunks):
        technology = claim.get("technology") or "both"
        check = lambda text, q=quotes, t=technology: numeric_errors(text, q, t)
        for line in claim["text"].splitlines():
            if line.strip().startswith("근거 공백"):      # Gap 목록이지 주장이 아니다
                continue
            # 검사기와 같은 인용 단위로 자른다. 각 수치는 그 조각이 인용한 원문과 대조된다
            offset = 0
            pieces = []
            for part, ids in segments(line, quotes):
                at_ = line.index(part, offset)
                pieces.append((at_, at_ + len(part), [quotes[i] for i in ids]))
                offset = at_ + len(part)
            for start, end, value, unit in spans(line):
                cited = next((c for s, e, c in pieces if s <= start < e), list(quotes.values()))
                if not supported(value, unit, cited) or check(line):
                    continue                     # 원본이 이미 틀린 줄은 변조 대상이 아니다
                original = line[start:end]
                at = lambda new: line[:start] + new + line[end:]     # 그 위치의 수치만 바꾼다
                decimals = len(original.split(".")[1]) if "." in original else 0
                changed = f"{value * 1.5:.{decimals}f}"
                yield "변조", line, (None if supported(float(changed), unit, cited)
                                     else at(changed)), check
                qualifiers = qualifiers_for(value, unit, technology)
                if any(q.lower() in line.lower() for q in qualifiers):
                    dropped = line                # 같은 뜻의 표기를 모두 지워야 조건이 빠진다
                    for qualifier in qualifiers:
                        dropped = re.sub(re.escape(qualifier), "", dropped, flags=re.I)
                    yield "조건 누락", line, dropped, check
                others = [v for v, u, cid in pool if u == unit and cid != claim["claim_id"] and v != value]
                if others:
                    other = next((v for v in others if not supported(v, unit, cited)), None)
                    yield "섞기", line, (None if other is None
                                       else at(f"{other:g}")), check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", type=Path)
    ap.add_argument("--index", action="store_true", help="quote 를 색인 청크 전문으로 되돌린다")
    args = ap.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    chunks = {}
    if args.index:
        from src.rag.index import build
        chunks = {k: d.page_content for k, d in build()["chunks"].items()}

    lines = [(line, quotes, c.get("technology") or "both") for _, c, quotes in claims(state, chunks)
             for line in c["text"].splitlines() if numbers(CITATION_RE.sub("", line))]
    flagged = [line for line, quotes, tech in lines if numeric_errors(line, quotes, tech)]

    total, caught, present = Counter(), Counter(), Counter()
    seen = set()
    for kind, original, mutated, check in mutate(state, chunks):
        if mutated is None:
            present[kind] += 1
            continue
        if (kind, mutated) in seen:
            continue
        seen.add((kind, mutated))
        total[kind] += 1
        caught[kind] += bool(check(mutated))

    print(f"수치가 있는 줄 {len(lines)}개 중 원본에서 오류 {len(flagged)}개")
    for line in flagged:
        print(f"  - {line.strip()[:110]}")
    print("\n변조 종류      탐지 / 생성   (원문에 있는 값이라 못 만든 변조)")
    for kind in ("변조", "조건 누락", "섞기"):
        rate = f"({caught[kind] / total[kind]:.0%})" if total[kind] else "(표본 없음)"
        print(f"  {kind:<10} {caught[kind]:>3} / {total[kind]:<3} {rate:<7} {present[kind]}")
    n = sum(total.values())
    print(f"  {'합계':<10} {sum(caught.values()):>3} / {n:<3}" + (f" ({sum(caught.values()) / n:.0%})" if n else ""))


if __name__ == "__main__":
    main()
