"""청킹·색인 — 담당: R2 (`src/rag/chunk.py`, `src/rag/index.py`)

PDF·토크나이저·임베딩 모델은 타지 않는다. 표 분할·chunk_id·추출/벡터 캐시 계약만 본다.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag import chunk, index


def test_table_chunks_repeat_the_header_in_every_piece(monkeypatch):
    """표를 통째로 자르면 행이 어느 열인지 잃는다 — 조각마다 헤더를 붙인다."""
    monkeypatch.setattr(chunk, "ROWS_PER_TABLE_CHUNK", 2)
    table = [["Method", "Bits"], ["KIVI", "2"], ["TurboQuant", "3.5"], ["FP16", "16"]]

    pieces = chunk.table_chunks([table], page=7)

    assert len(pieces) == 2
    assert all(text.startswith("(p7 표)\n| Method | Bits |") for text, _ in pieces)
    assert "| FP16 | 16 |" in pieces[1][0] and not any(review for _, review in pieces)


def test_ragged_or_garbled_tables_are_marked_for_review():
    ragged = [["a", "b", "c"], ["1"], ["2"], ["3", "4", "5"]]
    garbled = [["h"], ["\ufffd\ufffd 깨진 셀"]]

    assert chunk.table_chunks([ragged], 1)[0][1] is True
    assert chunk.table_chunks([garbled], 1)[0][1] is True
    assert chunk.table_chunks([[["only header"]]], 1) == []   # 본문 행이 없으면 조각도 없다


def test_chunk_id_is_stable_and_content_addressed():
    """재실행해도 인용 ID 가 같아야 하고, 본문이 바뀌면 달라져야 한다."""
    same = chunk.chunk_id("paper", 3, "본문")
    assert same == chunk.chunk_id("paper", 3, "본문") and len(same) == 12
    assert same != chunk.chunk_id("paper", 3, "본문 수정")


def test_two_column_split_uses_the_page_bbox():
    """CropBox 가 0 에서 시작하지 않는 PDF — 좌표를 페이지 bbox 기준으로 자른다."""
    crops = []

    class Page:
        bbox, width = (100, 0, 700, 800), 600

        def extract_words(self, **_):          # 가운데(400)를 가로지르는 단어가 없다 → 2단
            return [{"x0": 120, "x1": 300}, {"x0": 450, "x1": 650}]

        def crop(self, box):
            crops.append(box)
            return type("Half", (), {"extract_text": lambda self, **_: "half"})()

    assert index._page_text(Page()) == "half\nhalf"
    assert crops == [(100, 0, 400.0, 800), (400.0, 0, 700, 800)]


def test_page_cache_reuses_extraction_until_the_source_changes(tmp_path, monkeypatch):
    """PDF 추출 캐시 — 같은 원문이면 다시 파싱하지 않고, 해시가 바뀌면 다시 파싱한다."""
    calls = []
    monkeypatch.setattr(index, "PAGE_CACHE", tmp_path)
    monkeypatch.setattr(index, "_extract_pdf", lambda path: calls.append(path) or [(1, "본문", [["a", "b"]])])
    entry = {"id": "paper", "sha256": "aaa", "local_path": "paper.pdf"}

    assert index._cached_pdf(entry) == [(1, "본문", [["a", "b"]])]
    assert index._cached_pdf(entry) == [(1, "본문", [["a", "b"]])] and len(calls) == 1
    index._cached_pdf(entry | {"sha256": "bbb"})
    assert len(calls) == 2


def test_vector_cache_reuses_embeddings_until_a_chunk_changes(tmp_path, monkeypatch):
    """청크 벡터 캐시 — chunk_id(본문 해시)가 하나라도 바뀌면 다시 임베딩한다."""
    class Fake(Embeddings):
        calls = 0

        def embed_documents(self, texts):
            Fake.calls += 1
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    monkeypatch.setattr(index, "VECTOR_CACHE", tmp_path)
    docs = [Document(page_content="a", metadata={"chunk_id": "c1"})]

    index._faiss(docs, Fake())
    store = index._faiss(docs, Fake())
    assert Fake.calls == 1 and store.index.ntotal == 1
    index._faiss([Document(page_content="b", metadata={"chunk_id": "c2"})], Fake())
    assert Fake.calls == 2
