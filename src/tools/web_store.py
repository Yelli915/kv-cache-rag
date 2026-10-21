"""Run-scoped evidence store. Search candidates cannot enter the evidence registry."""
from datetime import datetime, timezone
import hashlib
import json
import re
import threading
from pathlib import Path
from copy import deepcopy
from src.tools.web_fetch import FetchedPage, fetch_page, extract_body, public_url  # noqa: F401 — web_search 가 다시 내보낸다


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


ID_CHARS = 12


def short_id(prefix, value):
    """모델이 그대로 옮겨 적어야 하는 ID 다. 짧을수록 전사 오류가 준다.

    sha256 64자를 그대로 쓰면 LLM 이 끝 글자를 흘려 인용 검증이 실패한다. 실제로
    그 이유로 실행이 failed 로 끝난 적이 있다. 색인 쪽 chunk_id(계약 ③)도 12자다.
    해시 원문은 sha256 필드에 그대로 남으므로 무결성 검증은 영향받지 않는다.
    """
    return f'{prefix}{digest(value)[:ID_CHARS]}'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def _search_tavily(query, top_k):
    from langchain_tavily import TavilySearch
    response = TavilySearch(max_results=top_k).invoke({'query': query})
    if not isinstance(response, dict) or not isinstance(response.get('results'), list):
        raise ValueError('search provider did not return results')
    return response['results']


def classify(url, title):
    text = (url + ' ' + title).lower()
    for keys, label in [
        (('arxiv.org', 'usenix.org', 'acm.org', 'ieee'), '논문'),
        (('cxlconsortium', 'jedec', 'standard'), '표준'),
        (('github.com', 'discussions'), '개발자논의'),
        (('newsroom', '/press', '/blog'), '제품발표'),
        (('gartner', 'idc.com', 'report', 'outlook'), '리포트')]:
        if any(k in text for k in keys):
            return label
    return '미확인'


class WebEvidenceStore:
    """One store per active run; persisted budgets survive resume.

    transport/searcher are dependency injection boundaries, not evidence bypasses:
    extraction, size limits, hashing and registration always execute here.
    A run must have one process owner (R1); an RLock serializes local tool calls.
    """
    def __init__(self, run_id, root='runs', *, searcher=None, transport=None,
                 max_sources=24, max_fetches=40, max_searches=16,
                 max_body_chars=50000, max_total_chars=300000,
                 max_response_bytes=2_000_000, min_body_chars=120, timeout=15):
        if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
            raise ValueError('invalid run_id')
        self.run_id = run_id
        self.directory = Path(root).resolve() / run_id / 'web'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.searcher = searcher or _search_tavily
        self.transport = transport or fetch_page
        self.timeout = timeout
        self._lock = threading.RLock()
        self.limits = dict(max_sources=max_sources, max_fetches=max_fetches,
            max_searches=max_searches, max_body_chars=max_body_chars,
            max_total_chars=max_total_chars, max_response_bytes=max_response_bytes,
            min_body_chars=min_body_chars)
        if any(type(v) is not int or v <= 0 for v in self.limits.values()) or timeout <= 0:
            raise ValueError('web limits must be positive')
        path = self.directory / 'manifest.json'
        if path.exists():
            self.manifest = json.loads(path.read_text(encoding='utf-8'))
            if self.manifest['run_id'] != run_id or self.manifest['limits'] != self.limits:
                raise ValueError('resumed run must preserve run_id and web limits')
        else:
            self.manifest = {'schema_version': 1, 'run_id': run_id, 'limits': self.limits,
                'sources': {}, 'evidence': {}, 'fetches': {}, 'searches': {}, 'failures': [],
                'usage': {'search_calls': 0, 'fetch_calls': 0, 'body_chars': 0}}
            self._save()

    def _save(self):
        save_json(self.directory / 'manifest.json', self.manifest)

    def _event(self, tool, status, **fields):
        event = dict(tool=tool, node='stakeholder', attempt=1, timestamp=utcnow(), status=status, **fields)
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')

    def search(self, query, technology='both', top_k=5):
        if not isinstance(query, str) or not query.strip() or technology not in {'TurboQuant', 'ITME', 'both'}:
            raise ValueError('invalid search query/technology')
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('web top_k must be 1..20')
        q = query if technology == 'both' else f'{technology} {query}'
        key = digest(f'{q}|{top_k}')
        with self._lock:
            if key in self.manifest['searches']:
                return deepcopy(self.manifest['searches'][key])
            if self.manifest['usage']['search_calls'] >= self.limits['max_searches']:
                self._event('search_market_signals', 'blocked', reason='search budget exceeded')
                raise ValueError('search budget exceeded')
            self.manifest['usage']['search_calls'] += 1
            self._save()
            try:
                raw = self.searcher(q, top_k)
                if not isinstance(raw, list):
                    raise ValueError('search results must be a list')
                out, seen = [], set()
                for row in raw[:top_k]:
                    try:
                        url = public_url(row.get('url', ''))
                    except ValueError:
                        continue
                    if url in seen:
                        continue
                    seen.add(url)
                    title = str(row.get('title') or '')[:1000]
                    out.append(dict(url=url, title=title, published_at=row.get('published_date') or None,
                        retrieved_at=utcnow(), snippet=str(row.get('content') or row.get('snippet') or '')[:4000],
                        source_type=classify(url, title), technology=technology, status='candidate'))
                self.manifest['searches'][key] = out
                save_json(self.directory / 'search' / f'{key}.json', dict(query=q, results=out))
                self._event('search_market_signals', 'ok', query=q, results=len(out))
                self._save()
                return deepcopy(out)
            except Exception as exc:
                self._event('search_market_signals', 'failed', error=type(exc).__name__)
                self._save()
                # Avoid exposing provider credentials embedded in transport exception strings.
                raise RuntimeError(f'web search failed ({type(exc).__name__})') from None

    def fetch(self, url):
        with self._lock:
            try:
                url = public_url(url)
                if url in self.manifest['fetches']:
                    cached = self.manifest['fetches'][url]
                    for key, hash_key in [('snapshot_path', 'sha256'), ('body_path', 'body_sha256')]:
                        if digest(Path(cached[key]).read_bytes()) != cached[hash_key]:
                            raise ValueError('cached snapshot hash mismatch')
                    return deepcopy(cached)
                usage = self.manifest['usage']
                if usage['fetch_calls'] >= self.limits['max_fetches']:
                    raise ValueError('fetch budget exceeded')
                if len(self.manifest['sources']) >= self.limits['max_sources']:
                    raise ValueError('source budget exceeded')
                usage['fetch_calls'] += 1
                self._save()
                page = self.transport(url, max_bytes=self.limits['max_response_bytes'], timeout=self.timeout)
                final_url = public_url(page.url)
                if page.status_code != 200:
                    raise ValueError(f'HTTP {page.status_code}')
                if len(page.content) > self.limits['max_response_bytes']:
                    raise ValueError('response byte budget exceeded')
                extracted = extract_body(page)
                paragraphs = extracted.pop('paragraphs')
                body = '\n\n'.join(text for _, text in paragraphs)
                if len(body) < self.limits['min_body_chars']:
                    raise ValueError('body unavailable or too short')
                if len(body) > self.limits['max_body_chars']:
                    raise ValueError('body character budget exceeded')
                if usage['body_chars'] + len(body) > self.limits['max_total_chars']:
                    raise ValueError('total body budget exceeded')
                raw_hash, body_hash = digest(page.content), digest(body)
                source_id = short_id('web-', f'{final_url}|{raw_hash}')
                snapshot_dir = self.directory / 'snapshots'
                snapshot_dir.mkdir(exist_ok=True)
                extension = '.pdf' if extracted['media_type'] == 'application/pdf' else '.html'
                snapshot = snapshot_dir / f'{source_id}{extension}'
                text_path = snapshot_dir / f'{source_id}.txt'
                snapshot.write_bytes(page.content)
                text_path.write_text(body, encoding='utf-8')
                retrieved = utcnow()
                source = dict(source_id=source_id, run_id=self.run_id, collection='web',
                    url=final_url, requested_url=url, final_url=final_url,
                    title=extracted['title'], institution=extracted['institution'], author=extracted['author'],
                    published_at=extracted['published_at'], retrieved_at=retrieved, version=raw_hash,
                    source_type=classify(final_url, extracted['title']), allowed_uses=['stakeholder'],
                    snapshot_path=str(snapshot), sha256=raw_hash, body_path=str(text_path), body_sha256=body_hash)
                evidence = []
                for location, paragraph in paragraphs:
                    # Keep quote contexts bounded, without dropping remaining paragraphs.
                    for offset in range(0, len(paragraph), 2000):
                        quote = paragraph[offset:offset + 2000]
                        loc = f'{location}:chars:{offset}-{offset + len(quote)}'
                        eid = short_id('e-web-', f'{source_id}|{loc}|{quote}')
                        evidence.append(dict(evidence_id=eid, source_id=source_id, run_id=self.run_id,
                            collection='web', quote=quote, location=loc, allowed_uses=['stakeholder'],
                            snapshot_path=str(snapshot), sha256=raw_hash))
                if source_id in self.manifest['sources']:
                    # Same final URL/body reached by an alias: reuse exact source metadata.
                    source = self.manifest['sources'][source_id]
                else:
                    self.manifest['sources'][source_id] = source
                    usage['body_chars'] += len(body)
                registry = self.manifest['evidence']
                for item in evidence:
                    kept = registry.get(item['evidence_id'])
                    if kept is not None and kept != item:
                        raise ValueError('evidence ID collision; raise ID_CHARS')
                    registry[item['evidence_id']] = item
                result = dict(status='ok', requested_url=url, final_url=final_url, body=body,
                    quote='\n\n'.join(e['quote'] for e in evidence), source=source, evidence=evidence,
                    title=source['title'], institution=source['institution'], published_at=source['published_at'],
                    retrieved_at=source['retrieved_at'], snapshot_path=str(snapshot), sha256=raw_hash,
                    body_path=str(text_path), body_sha256=body_hash)
                self.manifest['fetches'][url] = result
                self._event('fetch_web_evidence', 'ok', url=final_url, source_id=source_id, body_chars=len(body))
                self._save()
                return deepcopy(result)
            except Exception as exc:
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                failure = dict(status='failed', requested_url=url, error=reason, retrieved_at=utcnow())
                self.manifest['failures'].append(failure)
                self._event('fetch_web_evidence', 'blocked', url=url, error=reason)
                self._save()
                return failure
