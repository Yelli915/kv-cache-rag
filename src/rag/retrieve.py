"""R3: exact dense cosine retrieval with explicit collection/role boundaries.

The small (<200-page) corpus is searched in full before metadata filtering;
this avoids losing a technology when the nearest neighbours belong to another.
"""
import math
import numpy as np

COLLECTIONS = frozenset({'papers_core', 'ecosystem', 'context'})
TECHNOLOGIES = frozenset({'TurboQuant', 'ITME'})
ROLE_COLLECTIONS = {
    'research': ('papers_core',), 'maturity': ('papers_core',),
    'market': ('ecosystem',), 'domain': ('papers_core', 'context'),
}
ROLE_SCOPES = {
    'research': {'direct'}, 'maturity': {'direct', 'comparison'},
    'market': {'direct', 'ecosystem'},
    'domain': {'direct', 'comparison', 'secondary'},
}


def build():
    # Import lazily: mock-index users need neither PDF parsing nor a downloaded E5.
    from src.rag.index import build as build_index
    return build_index()


def validate_request(query, collection, technology, top_k, perspective=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be nonempty')
    if collection not in COLLECTIONS:
        raise ValueError(f'unknown collection: {collection}')
    if technology not in TECHNOLOGIES | {'both'}:
        raise ValueError(f'unknown technology: {technology}')
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('top_k must be an integer in 1..100')
    if perspective is not None:
        if perspective not in ROLE_COLLECTIONS or collection not in ROLE_COLLECTIONS[perspective]:
            raise ValueError(f'collection {collection} forbidden for {perspective}')


def permitted(metadata, collection, technology, perspective=None):
    """Validate metadata, then filter a well-formed but irrelevant record."""
    if metadata.get('collection') != collection:
        raise ValueError('returned collection differs from requested collection')
    for key in ('applies_to', 'perspectives'):
        value = metadata.get(key)
        if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
            raise ValueError(f'missing/invalid metadata: {key}')
    if not set(metadata['applies_to']) <= TECHNOLOGIES:
        raise ValueError('invalid technology metadata')
    if not metadata.get('source_id', metadata.get('source')) or not metadata.get('chunk_id'):
        raise ValueError('missing source/chunk metadata')
    if metadata.get('scope') not in {'direct', 'comparison', 'ecosystem', 'secondary'}:
        raise ValueError('missing/invalid metadata: scope')
    if technology != 'both' and technology not in metadata['applies_to']:
        return False
    if perspective and perspective not in metadata['perspectives']:
        return False
    if perspective and metadata['scope'] not in ROLE_SCOPES[perspective]:
        return False
    if collection == 'context' and metadata['scope'] not in {'comparison', 'secondary'}:
        return False
    return True


def validate_hits(hits, collection, technology, perspective=None):
    """Second boundary check for tool consumers, including replaced backends."""
    for hit in hits:
        if not permitted(hit, collection, technology, perspective):
            raise ValueError('returned hit violates role/technology scope')
        if not isinstance(hit.get('text'), str) or not hit['text'].strip():
            raise ValueError('empty returned text')
        score = hit.get('cosine_score')
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not -1.00001 <= score <= 1.00001:
            raise ValueError('invalid cosine_score')
    return hits


def search(query: str, collection: str, technology: str = 'both',
           top_k: int | None = None, mode: str = 'dense',
           perspective: str | None = None) -> list[dict]:
    """Return dense cosine hits. RRF requires new measured evidence and is disabled."""
    top_k = 5 if top_k is None else top_k
    validate_request(query, collection, technology, top_k, perspective)
    if mode != 'dense':
        raise ValueError('only mode=dense is supported; RRF/BM25 need evaluation evidence')
    store = build()['stores'].get(collection)
    if store is None:
        return []
    dense = store['dense']
    n = dense.index.ntotal
    if not n:
        return []
    query_vector = np.asarray(dense.embeddings.embed_query(query), dtype=np.float32)
    if not np.isfinite(query_vector).all() or not np.isclose(np.linalg.norm(query_vector), 1, atol=1e-4):
        raise ValueError('R2 query embeddings must be normalized for cosine search')
    # Validate actual stored vectors, not just a config flag. Never label raw L2 cosine.
    vectors = dense.index.reconstruct_n(0, n)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('R2 index embeddings must be normalized for cosine search')
    import faiss
    metric = dense.index.metric_type
    if metric not in {faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT}:
        raise ValueError('unsupported FAISS metric')
    ranked = dense.similarity_search_with_score_by_vector(query_vector.tolist(), k=n)
    hits = []
    for doc, raw in ranked:
        meta = doc.metadata
        if not permitted(meta, collection, technology, perspective):
            continue
        source = meta.get('source_id') or meta['source']
        score = float(raw) if metric == faiss.METRIC_INNER_PRODUCT else 1.0 - float(raw) / 2.0
        hits.append({
            'chunk_id': meta['chunk_id'], 'source_id': source, 'source': source,
            'collection': collection, 'page': meta['page'], 'text': doc.page_content,
            'cosine_score': max(-1.0, min(1.0, score)),
            'applies_to': list(meta['applies_to']), 'perspectives': list(meta['perspectives']),
            'scope': meta['scope'],
        })
    hits.sort(key=lambda h: (-h['cosine_score'], h['chunk_id']))
    return validate_hits(hits[:top_k], collection, technology, perspective)
