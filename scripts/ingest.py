"""인덱스 점검 — 담당: R2

LLM 을 호출하지 않으므로 API 키 없이 돌아간다.
수집·200쪽 가드·청킹·컬렉션 분리를 한 번에 검증한다.

    uv run python -m scripts.ingest

('python scripts/ingest.py' 로 직접 실행하면 sys.path[0] 이 scripts/ 가 되어
 src 를 찾지 못한다. -m 은 현재 디렉토리를 경로에 넣는다.)
"""

import os
import sys
from pathlib import Path

# 실행 위치에 상관없이 동작하게 한다. 설정·매니페스트·출력 경로가 전부
# 저장소 루트 기준 상대경로라, sys.path 추가와 함께 작업 디렉토리도 루트로 옮긴다.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)


from src.rag.index import build
from src.tools.docs import search_source_documents


def main():
    idx = build()

    b = idx["budget"]
    print("\n=== 매니페스트 ===")
    for m in idx["manifest"]:
        print(f"  {m['collection']:12s} {m['id']:26s} "
              f"{m.get('pages_counted', 0):6}쪽 {m.get('scope',''):11s} "
              f"sha256={m['sha256'][:12]}…")
    print(f"  합계 {b['total_pages']}쪽 / 한도 {b['page_budget']}쪽 "
          f"({'OK' if b['within_budget'] else '초과'}) · 청크 {len(idx['chunks'])}개")

    print("\n=== 컬렉션별 검색 ===")
    for col, q in [("papers_core", "KV 캐시 메모리 병목"),
                   ("ecosystem", "시장 규모 채택 현황"),
                   ("context", "데이터센터 추론 비용")]:
        hits = search_source_documents.invoke({"query": q, "collection": col, "top_k": 3})
        if not hits:
            print(f"  {col:12s} ⚠ 근거 0건 — sources.yaml 미확정")
            continue
        assert all(h["collection"] == col for h in hits), f"{col} 컬렉션 누수"
        print(f"  {col:12s} {len(hits)}건  예: {hits[0]['source']} p{hits[0]['page']} "
              f"[{hits[0]['chunk_id']}]")

    print("\n=== 기술별 필터 ===")
    for tech in ("TurboQuant", "ITME"):
        hits = search_source_documents.invoke(
            {"query": "처리량 정확도", "collection": "papers_core", "technology": tech, "top_k": 3})
        assert hits, f"{tech} 검색 결과 없음"
        print(f"  {tech}: {len(hits)}건")

    for n in idx.get("notes", []):
        print(f"\n⚠ {n}")
    print("\nOK")


if __name__ == "__main__":
    main()
