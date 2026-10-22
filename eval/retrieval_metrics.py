"""검색 품질 실측 — 담당: R4 (설계서 §4)

설계서 §4의 "측정 예정(구현 단계)" 칸을 채우기 위한 벤치마크 평가 스크립트입니다.
언어별(ko/en) × 모드별(dense/rrf)로 분리 측정하고, 같은 사실(fact)의 한/영 질의가
모두 top_k 내에 회수된 비율인 '완전 회수율(Full Recovery Rate)'을 산출합니다.

정답은 문항의 `gold_chunk_ids` 로만 판정합니다(§4). `gold_source`·`gold_page` 대체 매칭은
답이 없는 같은 페이지 청크까지 적중으로 세어 지표를 부풀리므로 쓰지 않습니다.

dense 는 운영 검색(`src.rag.retrieve.search`) 그대로다. bm25·rrf 는 **이 스크립트 안에만** 있다 —
운영 검색은 dense 단일이고(interface ②), RRF 는 측정 근거가 생기면 도입한다. 그 근거를 만드는
비교다. bm25 는 한국어 질의를 `query_kw.to_keywords` 로 영어 키워드로 바꿔 쓴다(원래 설계).

실행 방법:
    uv run python -m eval.retrieval_metrics            # 측정
    uv run python -m eval.retrieval_metrics --label    # anchor 로 gold_chunk_ids 다시 라벨링
"""

import argparse
import functools
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------
# 저장소 루트 기준 경로 고정 (실행 위치 무관화)
# ----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.rag.query_kw import to_keywords
from src.rag.retrieve import permitted, search

GOLDEN = Path("eval/goldenset.json")
OUTPUT_JSON = Path("eval/retrieval_results.json")
KS = (1, 3, 5)
MIN_N = 12  # 서브셋당 표본 수가 이보다 적으면 경향성 경고 표시
MODES = ("dense", "bm25", "rrf")
RRF_K = 60  # Cormack et al. 2009 기본값. 튜닝하지 않는다 — 28문항으로 튜닝하면 과적합이다
COLLECTION = "papers_core"


def _squash(text: str) -> str:
    """공백을 지운다. ITME PDF 추출본은 띄어쓰기가 대부분 사라져 있어 anchor 를 이렇게 맞춘다."""
    return re.sub(r"\s+", "", text)


def label(items: List[Dict[str, Any]]) -> None:
    """anchor 문구를 담은 gold_source 청크를 모두 정답으로 적는다(설계서 §4)."""
    from src.rag.index import build
    chunks = build()["chunks"]
    for item in items:
        anchor = _squash(item["anchor"])
        item["gold_chunk_ids"] = sorted(
            cid for cid, doc in chunks.items()
            if doc.metadata["source_id"] == item["gold_source"] and anchor in _squash(doc.page_content))
        if not item["gold_chunk_ids"]:
            raise SystemExit(f"❌ {item['id']}: anchor 를 담은 청크가 없다 — {item['anchor']!r}")


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


def _candidates(technology: str) -> List[Any]:
    """dense 검색과 같은 범위 — 컬렉션·기술 필터를 통과한 청크."""
    from src.rag.index import build
    return [d for d in build()["chunks"].values()
            if d.metadata["collection"] == COLLECTION
            and permitted({**d.metadata, "chunk_id": d.metadata["chunk_id"]}, COLLECTION, technology)]


@functools.lru_cache(maxsize=None)
def _bm25_index(technology: str):
    from rank_bm25 import BM25Okapi
    docs = _candidates(technology)
    return BM25Okapi([_tokens(d.page_content) for d in docs]), [d.metadata["chunk_id"] for d in docs]


@functools.lru_cache(maxsize=None)   # rrf 가 dense·bm25 순위를 다시 쓴다
def bm25_ranking(question: str, technology: str) -> tuple:
    bm25, ids = _bm25_index(technology)
    scores = bm25.get_scores(_tokens(to_keywords(question)))
    return tuple(cid for _, cid in sorted(zip(scores, ids), key=lambda x: (-x[0], x[1])))


@functools.lru_cache(maxsize=None)
def dense_ranking(question: str, technology: str) -> tuple:
    hits = search(question, collection=COLLECTION, technology=technology, top_k=100)
    return tuple(h["chunk_id"] for h in hits)


def ranking(item: Dict[str, Any], mode: str) -> List[str]:
    if mode == "dense":
        return dense_ranking(item["question"], item["technology"])
    if mode == "bm25":
        return bm25_ranking(item["question"], item["technology"])
    fused = defaultdict(float)
    for ranked in (dense_ranking(item["question"], item["technology"]),
                   bm25_ranking(item["question"], item["technology"])):
        for r, cid in enumerate(ranked, start=1):
            fused[cid] += 1.0 / (RRF_K + r)
    return [cid for cid, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))]


def gold_chunks(item: Dict[str, Any]) -> set:
    """정답 청크 ID 집합. 설계서 §4 가 문항마다 지정하라고 한 값이다.

    §4 — *각 문항에는 fact_id, 언어, 대상 기술, 정답 근거 문장과 청크 ID를 지정한다.
    겹침 때문에 같은 근거를 담은 청크가 여러 개면 모두 정답 집합에 넣는다.
    같은 페이지라도 답이 없는 청크는 정답으로 세지 않는다.*

    (source, page) 로 대신 맞추면 §4 가 명시적으로 배제한 "답이 없는 같은 페이지 청크" 가
    적중으로 잡혀 지표가 실제보다 높게 나온다.
    """
    ids = item.get("gold_chunk_ids")
    if not ids:
        raise SystemExit(
            f"❌ {item.get('id')}: gold_chunk_ids 가 없습니다. 설계서 §4 는 문항마다 정답 청크 ID 를\n"
            f"   지정하라고 합니다. (source, page) 대체 매칭은 답이 없는 같은 페이지 청크까지\n"
            f"   적중으로 세어 지표를 부풀립니다. eval/goldenset.json 에 라벨링이 필요합니다.")
    return set(ids)


def rank_of(item: Dict[str, Any], mode: str) -> Optional[int]:
    """정답 청크가 처음 등장한 순위(1-based). 없으면 None."""
    targets = gold_chunks(item)
    for idx, chunk_id in enumerate(ranking(item, mode)[:max(KS)], start=1):
        if chunk_id in targets:
            return idx
    return None


def calculate_metrics(ranks: List[Optional[int]]) -> Dict[str, float]:
    """Hit@k 및 MRR@k 산출."""
    n = len(ranks) or 1
    metrics = {}
    for k in KS:
        metrics[f"Hit@{k}"] = sum(r is not None and r <= k for r in ranks) / n
        metrics[f"MRR@{k}"] = sum(1.0 / r for r in ranks if r is not None and r <= k) / n
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true", help="anchor 로 gold_chunk_ids 를 다시 라벨링한다")
    args = ap.parse_args()

    if args.label:
        data = json.loads(GOLDEN.read_text(encoding="utf-8"))
        label(data["items"])
        GOLDEN.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts = [len(i["gold_chunk_ids"]) for i in data["items"]]
        print(f"✅ {len(counts)}문항 라벨링 — 정답 청크 {min(counts)}~{max(counts)}개/문항")
        return

    if not GOLDEN.exists():
        sys.exit(f"❌ 오류: {GOLDEN} 파일이 존재하지 않습니다.")

    try:
        data = json.loads(GOLDEN.read_text(encoding="utf-8"))
        raw_items = data.get("items", [])
    except Exception as e:
        sys.exit(f"❌ 오류: {GOLDEN} 파싱 실패 - {e}")

    items = [i for i in raw_items if not str(i.get("question", "")).startswith("TODO")]
    if not items:
        sys.exit(f"⚠️ {GOLDEN}의 items가 아직 TODO 상태입니다. 정답 페이지와 질의를 먼저 라벨링하십시오.")

    print(
        f"\n========================================================================\n"
        f"📊 RAG 검색 품질 실측 시작 (총 {len(items)}문항)\n"
        f" - 언어: ko {sum(i.get('lang')=='ko' for i in items)}개 / en {sum(i.get('lang')=='en' for i in items)}개\n"
        f" - 기술: TurboQuant {sum(i.get('technology')=='TurboQuant' for i in items)}개 / "
        f"ITME {sum(i.get('technology')=='ITME' for i in items)}개\n"
        f"========================================================================"
    )

    report_data = {}
    markdown_lines = []
    markdown_lines.append("\n### [설계서 §4 기입용 실측 요약 표]")
    markdown_lines.append("| Mode | Lang | Sample (n) | Hit@1 | Hit@3 | Hit@5 | MRR@3 | MRR@5 |")
    markdown_lines.append("|---|---|---|---|---|---|---|---|")

    for mode in MODES:
        print(f"\n▶ 검색 모드: [{mode.upper()}]")
        ranks = {item["id"]: rank_of(item, mode) for item in items}
        report_data[mode] = {"details": ranks, "metrics": {}}

        # 언어별 메트릭 산출
        cells = ["전체", "ko", "en", "TurboQuant", "ITME",
                 "ko·TurboQuant", "en·TurboQuant", "ko·ITME", "en·ITME"]
        for lang in cells:
            want = set(lang.split("·"))
            subset_ranks = [
                ranks[x["id"]]
                for x in items
                if lang == "전체" or want <= {x.get("lang"), x.get("technology")}
            ]
            m = calculate_metrics(subset_ranks)
            report_data[mode]["metrics"][lang] = m

            flag = " ⚠️ [표본 부족: 경향성만 참조]" if len(subset_ranks) < MIN_N else ""
            metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in m.items())
            print(f"  - {lang:14s} (n={len(subset_ranks):2d}): {metrics_str}{flag}")

            markdown_lines.append(
                f"| {mode} | {lang} | {len(subset_ranks)} | "
                f"{m['Hit@1']:.3f} | {m['Hit@3']:.3f} | {m['Hit@5']:.3f} | "
                f"{m['MRR@3']:.3f} | {m['MRR@5']:.3f} |"
            )

        # 설계서 §10 — 실패한 질의 패턴 목록
        missed = [x for x in items if ranks[x["id"]] is None]
        print(f"\n  [top-{max(KS)} 실패 {len(missed)}건]")
        for x in missed:
            print(f"    - {x['id']}: {x['question']}")
        report_data[mode]["missed"] = [x["id"] for x in missed]

        # 완전 회수율: 동일 fact_id의 ko 및 en 질의가 모두 top_k 안에 들어온 비율
        by_fact = defaultdict(list)
        for x in items:
            fact_id = x.get("fact_id")
            if fact_id:
                by_fact[fact_id].append(ranks[x["id"]])

        print(f"\n  [한/영 완전 회수율 (Full Recovery Rate) - 총 {len(by_fact)}개 Fact]")
        report_data[mode]["full_recovery"] = {}
        for k in KS:
            full_recovered = sum(
                len(rs) >= 2 and all(r is not None and r <= k for r in rs)
                for rs in by_fact.values()
            )
            ratio = full_recovered / len(by_fact) if by_fact else 0.0
            report_data[mode]["full_recovery"][f"FullRecovery@{k}"] = ratio
            print(f"    - 완전 회수율@{k}: {ratio:.3f} ({full_recovered}/{len(by_fact)} facts)")

    # 설계서 §4 복사 붙여넣기용 마크다운 표 출력
    print("\n" + "\n".join(markdown_lines))

    # 실행 결과 JSON 덤프
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 실측 상세 데이터가 '{OUTPUT_JSON}'에 저장되었습니다.\n")


if __name__ == "__main__":
    main()