#!/usr/bin/env python3
"""
RAG 코퍼스 준비 스크립트.

sources.json 에 적힌 출처를 내려받아
  - SHA-256 해시
  - PDF 실제 페이지 수 / 웹 문서 A4 환산 쪽수
  - 수집일
를 기록한 data/manifest.json 을 만든다.

설계서 3절의 "원문 URL, 문서 ID, 버전, 해시, PDF 페이지를 보존한다"와
"고유 페이지가 없는 웹 문서는 A4 기준 환산 쪽수를 산출한다"를 구현한다.

사용법:
    uv run python scripts/prepare_sources.py            # 없는 원문만 받고 해시 검증
    uv run python scripts/prepare_sources.py --refresh  # 원문 재수집 + 매니페스트 갱신

실패(다운로드·해시 불일치·PDF 쪽수)가 하나라도 있으면 매니페스트를 덮어쓰지 않는다 —
일부만 남은 매니페스트로 색인·분량 판정이 돌지 않게.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"
UA = "Mozilla/5.0 (compatible; KVCacheEval/1.0; academic coursework)"


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url: str, dest: Path) -> None:
    """임시 파일에 받은 뒤 교체한다. 실패해도 기존 원문은 그대로 남는다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def pdf_pages(path: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def html_to_text(path: Path) -> str:
    """본문만 남긴다. 탐색·스크립트·스타일 요소는 제거."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="원문을 다시 내려받고 매니페스트를 갱신한다. 원문이 바뀌면 chunk_id 가 "
                         "전부 달라져 goldenset 재라벨링이 필요하다")
    args = ap.parse_args()

    spec = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    cpp = spec["chars_per_page_a4_95pt"]
    budget = spec["page_budget"]
    prev = ({e["id"]: e for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"]}
            if MANIFEST.exists() else {})

    entries, failures = [], []
    for s in spec["sources"]:
        ext = "pdf" if s["type"] == "pdf" else "html"
        raw_path = RAW / f"{s['id']}.{ext}"

        old = prev.get(s["id"], {})
        if args.refresh or not raw_path.exists():
            try:
                print(f"[get ] {s['id']:<26} {s['url']}")
                download(s["url"], raw_path)
            except Exception as e:                       # noqa: BLE001
                print(f"[FAIL] {s['id']}: {e}", file=sys.stderr)
                failures.append({"id": s["id"], "url": s["url"], "error": str(e)})
                continue

        entry = {k: s[k] for k in
                 ("id", "collection", "title", "url", "type", "language",
                  "applies_to", "perspectives", "scope") if k in s}
        digest = sha256(raw_path)
        # data/raw 는 커밋하지 않으므로 새 클론의 기본 실행도 원문을 받는다 — 그때도
        # 커밋된 매니페스트 해시와 맞아야 한다. 원문 변경은 --refresh 일 때만 받아들인다.
        if old.get("sha256") and old["sha256"] != digest:
            if not args.refresh:
                print(f"[FAIL] {s['id']}: 해시 불일치 (매니페스트 {old['sha256'][:12]} ≠ "
                      f"원문 {digest[:12]})", file=sys.stderr)
                failures.append({"id": s["id"], "url": s["url"], "error": "sha256 mismatch"})
                continue
            print(f"[chg ] {s['id']}: 원문이 바뀌었다 — chunk_id 가 달라져 goldenset 재라벨링 필요")
        entry.update({
            "local_path": str(raw_path.relative_to(ROOT)),
            "sha256": digest,
            "bytes": raw_path.stat().st_size,
            # --refresh 가 아니면 같은 원문(해시 일치)이므로 기존 수집일을 유지한다.
            "retrieved_at": (date.today().isoformat() if args.refresh
                             else old.get("retrieved_at", date.today().isoformat())),
        })
        for k in ("version", "publisher", "published_at", "landing", "note", "excerpt_only",
                  "excerpt_pages"):
            if k in s:
                entry[k] = s[k]

        if s["type"] == "pdf":
            try:
                entry["pdf_pages"] = pdf_pages(raw_path)
                entry["pages_counted"] = entry["pdf_pages"]
                entry["page_basis"] = "pdf_actual"
            except Exception as e:                       # noqa: BLE001
                print(f"[FAIL] {s['id']} 페이지 수 실패: {e}", file=sys.stderr)
                failures.append({"id": s["id"], "url": s["url"], "error": f"pdf pages: {e}"})
                continue
        else:
            text = html_to_text(raw_path)
            txt_path = RAW / f"{s['id']}.txt"
            txt_path.write_text(text, encoding="utf-8")
            entry["text_path"] = str(txt_path.relative_to(ROOT))
            entry["chars"] = len(text)
            entry["pages_counted"] = round(len(text) / cpp, 1)
            entry["page_basis"] = f"a4_9.5pt_equiv@{cpp}chars"

        entries.append(entry)
        print(f"       -> {entry.get('pages_counted','?')}쪽  sha={entry['sha256'][:12]}")

    total = round(sum(e.get("pages_counted", 0) for e in entries), 1)
    by_coll: dict[str, float] = {}
    for e in entries:
        by_coll[e["collection"]] = round(
            by_coll.get(e["collection"], 0) + e.get("pages_counted", 0), 1)

    if failures:
        for f in failures:
            print(f"  [FAIL] {f['id']}: {f['error']}", file=sys.stderr)
        print(f"실패 {len(failures)}건 — 매니페스트를 갱신하지 않았다 ({MANIFEST.relative_to(ROOT)})",
              file=sys.stderr)
        return 1

    manifest = {
        "generated_at": date.today().isoformat(),
        "chars_per_page_a4_95pt": cpp,
        "page_budget": budget,
        "total_pages": total,
        "within_budget": total <= budget,
        "pages_by_collection": by_coll,
        "source_count": len(entries),
        "sources": entries,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 58)
    for c, p in sorted(by_coll.items()):
        print(f"  {c:<14} {p:>7} 쪽")
    print(f"  {'합계':<13} {total:>7} 쪽 / 한도 {budget}쪽  "
          f"{'OK' if total <= budget else '초과!'}")
    print("=" * 58)
    print(f"매니페스트: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
