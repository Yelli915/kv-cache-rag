"""R3 run-bound web tools: candidate search and verified body collection."""
from contextlib import contextmanager
from contextvars import ContextVar
from html import escape
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from src.tools.web_store import WebEvidenceStore, FetchedPage  # noqa: F401 — 호출자가 여기서 가져간다

_CURRENT: ContextVar[WebEvidenceStore | None] = ContextVar('r3_web_store', default=None)


@contextmanager
def web_run(store):
    token = _CURRENT.set(store)
    try:
        yield store
    finally:
        _CURRENT.reset(token)


def _store():
    store = _CURRENT.get()
    if store is None:
        raise RuntimeError('web tools require an explicit run context; use bind_web_tools or web_run')
    return store


@tool
def search_market_signals(query: str, technology: str = 'both', top_k: int = 5) -> list[dict]:
    """Return stakeholder research candidates. Snippets are not factual evidence."""
    return _store().search(query, technology, top_k)


@tool
def fetch_web_evidence(url: str) -> dict:
    """Fetch original body and register evidence only after successful extraction."""
    return _store().fetch(url)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1)
    technology: Literal['TurboQuant', 'ITME', 'both'] = 'both'
    top_k: int = Field(default=5, ge=1, le=20)


class FetchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str


def bind_web_tools(store: WebEvidenceStore):
    return (
        StructuredTool.from_function(store.search, name='search_market_signals',
            description='Return research candidates, never verified evidence.', args_schema=SearchInput),
        StructuredTool.from_function(store.fetch, name='fetch_web_evidence',
            description='Fetch original body and provenance; failure cannot be cited.', args_schema=FetchInput),
    )


def format_signals(signals):
    """Only fetched bodies may be formatted as model evidence (legacy name)."""
    documents = []
    for s in signals:
        if s.get('status') != 'ok' or not s.get('body') or not s.get('evidence'):
            raise ValueError('unfetched search candidate cannot be model evidence')
        source = s['source']
        for e in s['evidence']:
            fields = dict(evidence_id=e['evidence_id'], source_id=e['source_id'],
                url=source['url'], title=source['title'], institution=source['institution'],
                published_at=source['published_at'] or '미상', location=e['location'], quote=e['quote'])
            documents.append('<document>' + ''.join(f'<{k}>{escape(str(v))}</{k}>' for k, v in fields.items()) + '</document>')
    return '\n'.join(documents)
