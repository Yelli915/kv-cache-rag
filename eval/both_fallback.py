"""기술별 절 분리 안정성 — 같은 입력을 캐시 없이 N번 돌려 both fallback 을 센다.

research · maturity · domain 을 개선 전(헤더 정규식, git ref) 코드와 지금(구조화 출력) 코드로
각각 N번 실행한다. 검색·색인은 같고 LLM 응답만 매번 새로 받는다.

- both fallback — 기술별로 나뉘어야 할 fact Claim 이 `technology=both` 로 나왔다.
  개선 전 domain 은 설계상 fact 를 나누지 않았으므로 세지 않는다("설계상 both").
- 섞임 — TurboQuant Claim 본문에 ITME 가, ITME Claim 에 TurboQuant 가 나온다. 구조화 출력은
  both 가 구조적으로 나올 수 없으니, 나뉘었어도 내용이 섞였는지를 따로 본다.

**응답 캐시를 켜면 측정이 무의미하다** — 같은 프롬프트에 같은 응답이 나와 N번이 1번이 된다.
그래서 `enable_cache()` 를 부르지 않고, 켜져 있으면 멈춘다. 실제 API 를 호출한다.

    uv run python -m eval.both_fallback --runs 5
    uv run python -m eval.both_fallback --runs 5 --legacy-ref 891c6e6
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NODES = (("research", "research", "research"),
         ("maturity", "maturity", "maturity"),
         ("domain", "domain_assessment", "domain_assessment"))   # (모듈, 함수, State 키)


def legacy_package(ref: str) -> types.ModuleType:
    """git ref 시점의 common·research·maturity·domain 을 `legacy` 패키지로 불러온다."""
    root = Path(tempfile.mkdtemp()) / "legacy"
    root.mkdir()
    for name in ("common", "research", "maturity", "domain"):
        source = subprocess.check_output(["git", "show", f"{ref}:src/agents/{name}.py"], cwd=ROOT, text=True)
        (root / f"{name}.py").write_text(source.replace("src.agents.common", "legacy.common"), encoding="utf-8")
    package = types.ModuleType("legacy")
    package.__path__ = [str(root)]
    sys.modules["legacy"] = package
    for name in ("common", "research", "maturity", "domain"):
        spec = importlib.util.spec_from_file_location(f"legacy.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"legacy.{name}"] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
    return package


def classify(module: str, legacy: bool, assessment: dict) -> dict:
    facts = [c for c in assessment.get("claims") or [] if c["kind"] == "fact"]
    by_design = legacy and module == "domain"
    split = {c["technology"] for c in facts} >= {"TurboQuant", "ITME"}
    mixed = sum(1 for c in facts if
                (c["technology"] == "TurboQuant" and "ITME" in c["text"]) or
                (c["technology"] == "ITME" and "TurboQuant" in c["text"]))
    return {"status": assessment.get("status"),
            "fallback": not by_design and any(c["technology"] == "both" for c in facts),
            "by_design": by_design, "split": split, "mixed": mixed,
            # 섞임이 본문 탓인지 뒤에 붙은 요약·근거 공백 줄 탓인지 사람이 확인할 수 있게 남긴다
            "claims": [{"technology": c["technology"], "text": c["text"]} for c in facts]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--legacy-ref", default="891c6e6", help="헤더 정규식 시절 커밋")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "both_fallback_results.json")
    args = ap.parse_args()

    import os
    os.chdir(ROOT)
    from dotenv import load_dotenv
    from langchain_core.globals import get_llm_cache
    load_dotenv(override=True)
    assert get_llm_cache() is None, "응답 캐시가 켜져 있다 — 같은 응답이 반복돼 측정이 무의미하다"

    from src.rag.index import build
    build()                                   # 색인을 먼저 한 번 만들어 스레드가 공유한다

    import src.agents.domain, src.agents.maturity, src.agents.research  # noqa: E401,F401 — 하위 모듈을 패키지 속성으로 올린다
    current = sys.modules["src.agents"]
    legacy = legacy_package(args.legacy_ref)

    # 회차 단위로 개선 전·후를 번갈아 넣는다. 도중에 API 한도에 걸려도 두 쪽이 같은 수만큼
    # 남는다(2026-10-21 첫 측정은 개선 전부터 몰아 돌려 개선 후 20회가 비었다).
    jobs = [(label, package, legacy_flag, module, func, key, i)
            for i in range(args.runs)
            for label, package, legacy_flag in (("개선 전", legacy, True), ("개선 후", current, False))
            for module, func, key in NODES]

    def run(job):
        label, package, legacy_flag, module, func, key, i = job
        node = getattr(getattr(package, module), func)
        try:
            out = node({"run_id": f"both-{module}-{i}", "run_config": {}})
            return label, module, classify(module, legacy_flag, out[key])
        except Exception as exc:                # 한 번의 실패가 측정 전체를 죽이지 않게
            return label, module, {"error": f"{type(exc).__name__}: {exc}"[:200]}

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, jobs))

    table = {}
    for label, module, result in results:
        row = table.setdefault((label, module), Counter())
        row["runs"] += 1
        for k in ("fallback", "by_design", "split", "error"):
            row[k] += bool(result.get(k))
        row["mixed"] += result.get("mixed", 0)

    print(f"캐시 없음 · 노드당 {args.runs}회 · 개선 전 = {args.legacy_ref}\n")
    print(f"{'':<8}{'노드':<10}{'both fallback':>14}{'기술별 분리':>10}{'섞인 Claim':>10}{'오류':>6}")
    print("(both fallback·기술별 분리는 오류 없이 끝난 실행 수 기준)")
    for (label, module), row in table.items():
        done = row["runs"] - row["error"]
        fallback = "설계상 both" if row["by_design"] else f"{row['fallback']} / {done}"
        print(f"{label:<8}{module:<10}{fallback:>14}{row['split']:>7} / {done}"
              f"{row['mixed']:>10}{row['error']:>6}")
    args.out.write_text(json.dumps([{"version": l, "node": m, **r} for l, m, r in results],
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n원자료: {args.out}")


if __name__ == "__main__":
    main()
