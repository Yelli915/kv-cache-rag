"""컬렉션별 인덱스 구축 — 담당: R2 (설계서 §3 §4)

수집·해시·쪽수는 **scripts/prepare_sources.py** 가 끝낸다.
여기서는 data/manifest.json 을 읽어 청킹·임베딩·인덱싱만 한다.

    uv run python scripts/prepare_sources.py   # 원문 수집 + 매니페스트
    uv run python -m scripts.ingest            # 색인 점검
"""

import json
from pathlib import Path

import pdfplumber
from langchain_community.vectorstores import FAISS

from src.rag import chunk as chunker
from src.rag.embed import get_embeddings

MANIFEST = Path("data/manifest.json")
COLLECTIONS = ("papers_core", "ecosystem", "context")

_cache: dict = {}


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "data/manifest.json 없음. 먼저 원문을 수집할 것:\n"
            "    uv run python scripts/prepare_sources.py"
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# 기본값 3 은 글자 간격이 좁은 2단 학회 논문에서 단어 사이 공백을 지운다
# ("ITMEachievesuptoa35.7%"). 1.5 면 공백이 돌아오고 1단 논문은 그대로다.
X_TOLERANCE = 1.5


def _page_text(page) -> str:
    """2단 논문은 단마다 따로 읽는다 — 통째로 읽으면 왼쪽·오른쪽 단의 줄이 번갈아 섞인다.

    페이지 가운데 선을 가로지르는 단어가 거의 없으면 2단으로 본다. 제목·그림처럼 전폭인 요소는
    좌우로 갈리지만 본문 문장이 섞이는 것보다 낫다. ponytail: 가로지르는 단어 비율 하나로
    판정한다. 3단·혼합 레이아웃이 들어오면 pdfplumber 의 단 검출로 바꾼다.

    검색 실측(eval/retrieval_metrics.py, 28문항): dense Hit@5 0.43 → 0.68, ITME 0.29 → 0.71.
    """
    words = page.extract_words(x_tolerance=X_TOLERANCE)
    # 좌표는 페이지 bbox 기준이다. CropBox 가 0 에서 시작하지 않는 PDF 도 있다.
    x0, top, x1, bottom = page.bbox
    mid = (x0 + x1) / 2
    if words and sum(w["x0"] < mid < w["x1"] for w in words) / len(words) < 0.02:
        left = page.crop((x0, top, mid, bottom)).extract_text(x_tolerance=X_TOLERANCE) or ""
        right = page.crop((mid, top, x1, bottom)).extract_text(x_tolerance=X_TOLERANCE) or ""
        return left + "\n" + right
    return page.extract_text(x_tolerance=X_TOLERANCE) or ""


def _pages(entry: dict) -> list[tuple[int, str, list]]:
    """매니페스트 항목을 [(페이지, 본문, 표)] 로 읽는다."""
    if entry["type"] == "pdf":
        with pdfplumber.open(entry["local_path"]) as pdf:
            pages = [(i + 1, _page_text(p), p.extract_tables())
                     for i, p in enumerate(pdf.pages)]
        # excerpt_only 문서는 sources.json 의 excerpt_pages: [시작, 끝] 로 범위를 좁힌다.
        lo, hi = (entry.get("excerpt_pages") or [0, 10**9])
        return [p for p in pages if lo <= p[0] <= hi]
    # 웹 문서는 prepare_sources 가 본문만 뽑아 .txt 로 저장해 둔다.
    return [(1, Path(entry["text_path"]).read_text(encoding="utf-8"), [])]


def build() -> dict:
    if _cache:
        return _cache

    mf = load_manifest()
    if not mf.get("within_budget", True):
        raise ValueError(
            f"색인 {mf['total_pages']}쪽 > 한도 {mf['page_budget']}쪽. "
            "sources.json 을 줄이거나 excerpt 범위를 지정할 것."
        )

    docs_by_col, notes = {c: [] for c in COLLECTIONS}, []
    for e in mf["sources"]:
        meta = {
            "collection": e["collection"],
            "source": e["id"],
            "source_id": e["id"],  # retrieve.py 가 source_id 를 우선 읽는다 (source 는 하위 호환).
            # applies_to 는 목록이다(한 문서가 두 기술에 걸릴 수 있음).
            "applies_to": e.get("applies_to", []),
            "perspectives": e.get("perspectives", []),
            "scope": e.get("scope", ""),
            "excerpt_only": e.get("excerpt_only", False),
        }
        chunks = chunker.split(_pages(e), meta)
        docs_by_col[e["collection"]] += chunks
        if e.get("excerpt_only") and not e.get("excerpt_pages"):
            notes.append(
                f"{e['id']}: excerpt_only 인데 excerpt_pages 가 없어 전문({e.get('pages_counted')}쪽)을 "
                "색인했다. sources.json 에 excerpt_pages: [시작, 끝] 을 지정할 것 (R2)."
            )
        print(f"[index] {e['collection']}/{e['id']}: "
              f"{e.get('pages_counted','?')}쪽 / 청크 {len(chunks)}개")

    embeddings = get_embeddings()
    stores = {}
    for collection, docs in docs_by_col.items():
        if not docs:
            continue
        # R3 contract: dense only until retrieval failure analysis justifies BM25.
        stores[collection] = {"dense": FAISS.from_documents(docs, embeddings)}

    for n in notes:
        print(f"[index] ⚠ {n}")
    print(f"[index] 합계 {mf['total_pages']}쪽 / 한도 {mf['page_budget']}쪽 · "
          f"청크 {sum(len(d) for d in docs_by_col.values())}개")

    _cache.update(
        stores=stores,
        manifest=mf["sources"],
        budget=mf,
        notes=notes,
        chunks={d.metadata["chunk_id"]: d for docs in docs_by_col.values() for d in docs},
    )
    return _cache
