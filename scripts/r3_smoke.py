"""Offline R3 handoff evidence. All web/LLM content is explicitly synthetic.

    uv run python -m scripts.r3_smoke --root /tmp/kv-r3-smoke

FAISS retrieval, HTML extraction, hash checks, role guards, node validation and
file persistence are real. Only embedding vectors and network/LLM I/O are fixtures.
An existing root is rejected rather than deleting or mixing earlier evidence.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

# 'python scripts/r3_smoke.py' 로 직접 실행해도 src·tests 를 찾게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.rag.index import dense_store
from src.rag import retrieve
from src.tools.docs import bind_document_search
from src.tools.web_fetch import FetchedPage
from src.tools.web_search import WebEvidenceStore, bind_web_tools
from src.agents.stakeholder import make_stakeholder
from src.tools.web_store import save_json, utcnow


class FixtureEmbeddings(Embeddings):
    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[0.6, 0.8] for _ in texts]


def run(root):
    root = Path(root)      # resolve 하지 않는다 — 증빙에 로컬 절대경로를 남기지 않기 위해
    if root.exists():
        raise ValueError('use a new smoke root; previous evidence is not overwritten')
    root.mkdir(parents=True)
    html = (Path(__file__).resolve().parents[1] / 'tests/fixtures/r3-web.html').read_bytes()
    rows = []
    doc = Document(page_content='Synthetic ITME retrieval fixture, not a paper result.', metadata={
        'chunk_id': '0123456789ab', 'source': 'fixture-paper', 'collection': 'papers_core',
        'applies_to': ['ITME'], 'perspectives': ['research', 'maturity', 'domain'],
        'scope': 'direct', 'page': 1})
    index = {'stores': {'papers_core': {'dense': dense_store([doc], FixtureEmbeddings())}},
             'chunks': {'0123456789ab': doc}}
    with patch.object(retrieve, 'build', return_value=index):
        hits = bind_document_search('research').invoke({'query': 'ITME', 'technology': 'ITME'})
        assert abs(hits[0]['cosine_score'] - 0.6) < 1e-5
        save_json(root / 'document-tool-output.json', hits)
        rows.append({'case': 'dense_cosine', 'passed': True, 'score': hits[0]['cosine_score']})
        try:
            retrieve.search('adoption', 'papers_core', perspective='market')
        except ValueError as exc:
            rows.append({'case': 'collection_violation', 'passed': True, 'error': str(exc)})
        else:
            raise AssertionError('collection violation was not blocked')

    def transport(url, **kwargs):
        return FetchedPage(url, 403 if url.endswith('/blocked') else 200, {'content-type': 'text/html'}, html)

    def searcher(query, top_k):
        return [{'url': 'https://example.org/synthetic', 'title': 'Synthetic candidate',
                 'content': 'This candidate snippet must not enter factual evidence.'}]

    store = WebEvidenceStore('r3-smoke', root=root / 'runs', transport=transport, searcher=searcher)
    search_tool, fetch_tool = bind_web_tools(store)
    candidates = search_tool.invoke({'query': 'developer statement', 'technology': 'TurboQuant'})
    save_json(root / 'search-tool-output.json', candidates)
    blocked = fetch_tool.invoke({'url': 'https://example.org/blocked'})
    assert blocked['status'] == 'failed' and 'evidence' not in blocked
    rows.append({'case': 'body_unavailable', 'passed': True, 'result': blocked})
    fetched = fetch_tool.invoke({'url': candidates[0]['url']})
    assert fetched['status'] == 'ok'
    save_json(root / 'fetch-tool-output.json', fetched)

    def writer(**kwargs):
        assert 'candidate snippet' not in kwargs['context']
        eid = next(e['evidence_id'] for e in fetched['evidence'] if 'Developer Kim' in e['quote'])
        return {'claims': [{'text': '합성 인터뷰에서 Kim은 TurboQuant 커널 검토를 언급했다.',
            'technology': 'TurboQuant', 'kind': 'fact', 'evidence_ids': [eid],
            'explanation': '', 'stakeholder_group': 'developers', 'actor': 'Kim (synthetic fixture)',
            'statement_date': None, 'context': '실제 평가에 사용하지 않는 테스트 인터뷰'}], 'gaps': []}

    output = make_stakeholder(store, writer=writer)({'run_id': 'r3-smoke'})
    assert output['stakeholder']['status'] == 'partial'
    assert set(output) == {'stakeholder', 'trace'}
    rows.append({'case': 'linked_assessment', 'passed': True, 'status': 'partial',
                 'claims': len(output['stakeholder']['claims']), 'gaps': len(output['stakeholder']['gaps'])})

    bad_store = WebEvidenceStore('r3-smoke-invalid', root=root / 'runs', transport=transport, searcher=searcher)
    bad = make_stakeholder(bad_store, writer=lambda **kw: {'claims': [{'text': 'bad'}], 'gaps': []})({'run_id': 'r3-smoke-invalid'})
    assert bad['stakeholder']['status'] == 'failed'
    rows.append({'case': 'unresolved_draft_error', 'passed': True, 'status': 'failed'})
    save_json(root / 'valid-assessment-output.json', output)
    save_json(root / 'failed-assessment-output.json', bad)
    result = {'fixture_only': True, 'timestamp': utcnow(), 'root': str(root), 'cases': rows}
    save_json(root / 'checks.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    run(parser.parse_args().root)
