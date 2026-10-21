"""청킹 — 담당: R2 (설계서 §3)

토큰 400 / overlap 80, 임베딩 입력 한도 512 상한.
표는 행 단위로 쪼개되 매 조각에 헤더 행을 반복해 붙인다 — 표를 통째로 자르면
행이 어느 열에 속하는지 잃어버려 벤치마크 수치 인용이 망가진다.
"""

import hashlib
import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.rag.embed import EMBED_MODEL, EMBED_REVISION
from src.settings import settings

ROWS_PER_TABLE_CHUNK = 12
# PDF 추출이 깨지면 흔히 남는 대체문자/전용영역 문자 (표·수식이 글리프로만 존재해
# 텍스트 추출에 실패한 경우).
_BROKEN_CHARS = {"\ufffd"}


def _looks_broken(text: str) -> bool:
    """표·수식이 PDF 추출 과정에서 깨졌는지 휴리스틱으로 판단한다.

    수식이 깨지면 대체문자가 섞이거나, 공백 없는 특수기호 나열(깨진 수식 기호)이
    비정상적으로 많이 남는 경우가 흔하다. 확정 판정이 아니라 검토 대상 표시용이다.
    """
    if not text.strip():
        return False
    bad = sum(1 for ch in text if ch in _BROKEN_CHARS or 0xE000 <= ord(ch) <= 0xF8FF)
    if bad / max(len(text), 1) > 0.02:
        return True
    symbol_runs = re.findall(r"[^\w\s]{6,}", text)
    return len(symbol_runs) >= 2


def _table_is_malformed(rows: list[list[str]]) -> bool:
    """열 수가 행마다 들쭉날쭉하면 pdfplumber 가 셀 경계를 잘못 잡은 것이다."""
    lengths = [len(r) for r in rows]
    if not lengths:
        return True
    mode = max(set(lengths), key=lengths.count)
    mismatched = sum(1 for n in lengths if n != mode)
    return mismatched / len(lengths) > 0.3


def chunk_id(source: str, page: int, text: str) -> str:
    """(출처, 페이지, 본문 해시). 재실행해도 인용 ID 가 변하지 않아야 한다."""
    return hashlib.sha1(f"{source}|{page}|{text}".encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def _splitter():
    """원문마다 토크나이저를 새로 불러오면 색인 생성이 원문 수 × 약 3초 늘어난다(실측 29초).

    토크나이저도 임베딩 모델과 같은 revision 으로 고정한다 — 청크 경계가 chunk_id 를 정한다.
    """
    lim = settings()["limits"]
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION),
        chunk_size=lim["chunk_tokens"],
        chunk_overlap=lim["chunk_overlap"],
    )


def table_chunks(tables: list, page: int) -> list[tuple[str, bool]]:
    """표를 헤더 반복 + 행 분할로 Markdown 조각 목록으로 만든다.

    반환값은 (조각, 검토 필요 여부) 쌍의 목록이다. 열 수가 들쭉날쭉하거나 조각
    본문이 깨진 것으로 보이면 검토 필요로 표시한다.
    """
    out = []
    for tbl in tables or []:
        rows = [[(c or "").strip() for c in r] for r in tbl if any(r)]
        if len(rows) < 2:
            continue
        malformed = _table_is_malformed(rows)
        header = "| " + " | ".join(rows[0]) + " |"
        sep = "|" + "---|" * len(rows[0])
        for i in range(1, len(rows), ROWS_PER_TABLE_CHUNK):
            body = "\n".join("| " + " | ".join(r) + " |" for r in rows[i : i + ROWS_PER_TABLE_CHUNK])
            piece = f"(p{page} 표)\n{header}\n{sep}\n{body}"
            out.append((piece, malformed or _looks_broken(body)))
    return out


def split(texts_by_page: list[tuple[int, str, list]], meta: dict) -> list[Document]:
    """[(페이지, 본문, 표)] 를 청크 Document 목록으로 만든다.

    각 조각에는 needs_review 메타데이터를 붙인다 — 표 추출이 들쭉날쭉하거나
    수식이 깨진 것으로 보이는 조각은 True 로 표시해 검토 대상임을 남긴다.
    """
    splitter, docs = _splitter(), []

    for page, body, tables in texts_by_page:
        pieces = [(d.page_content, _looks_broken(d.page_content)) for d in splitter.split_documents(
            [Document(page_content=body, metadata={})]
        )] if body.strip() else []
        pieces += table_chunks(tables, page)

        for text, needs_review in pieces:
            docs.append(Document(
                page_content=text,
                metadata={
                    **meta, "page": page,
                    "chunk_id": chunk_id(meta["source"], page, text),
                    "needs_review": needs_review,
                },
            ))
    return docs
