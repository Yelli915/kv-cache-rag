"""R3 document search boundary. Bind roles in Python before exposing tools to LLMs."""
from html import escape
from typing import Literal
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from src.rag.retrieve import search, validate_hits, ROLE_COLLECTIONS


@tool
def search_source_documents(query: str, collection: str, technology: str = 'both',
                            top_k: int = 5, perspective: str = '') -> list[dict]:
    """Search a collection using dense cosine. Trusted Python callers only.

    LLM integrations must use bind_document_search so collection and role cannot
    be selected by generated arguments.
    """
    hits = search(query, collection=collection, technology=technology,
                  top_k=top_k, perspective=perspective or None)
    return validate_hits(hits, collection, technology, perspective or None)


class BoundSearchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1)
    technology: Literal['TurboQuant', 'ITME', 'both'] = 'both'
    top_k: int = Field(default=5, ge=1, le=100)


def bind_document_search(role: str, collection: str | None = None):
    """Return a tool whose collection and perspective cannot be supplied by an LLM.

    domain defaults to papers_core; bind a second tool with collection='context'
    for background evidence. Role/collection choices belong to application code.
    """
    if role not in ROLE_COLLECTIONS:
        raise ValueError(f'unknown RAG role: {role}')
    collection = collection or ROLE_COLLECTIONS[role][0]
    if collection not in ROLE_COLLECTIONS[role]:
        raise ValueError(f'collection {collection} forbidden for {role}')

    def bound(query: str, technology: str = 'both', top_k: int = 5):
        hits = search(query, collection=collection, technology=technology,
                      top_k=top_k, perspective=role)
        return validate_hits(hits, collection, technology, role)

    return StructuredTool.from_function(bound, name=f'search_{role}_{collection}',
        description=f'Search {collection} evidence permitted for {role}.',
        args_schema=BoundSearchInput)


def format_chunks(chunks: list[dict]) -> str:
    """Escape untrusted source text while preserving the legacy prompt format."""
    output = []
    for c in chunks:
        fields = {'id': c['chunk_id'], 'collection': c['collection'],
                  'source': c.get('source_id') or c['source'], 'page': c['page'],
                  'applies_to': ','.join(c['applies_to']), 'scope': c['scope'], 'content': c['text']}
        output.append('<document>' + ''.join(f'<{k}>{escape(str(v))}</{k}>' for k, v in fields.items()) + '</document>')
    return '\n'.join(output)
