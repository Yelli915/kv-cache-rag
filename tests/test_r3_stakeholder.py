import json
from pathlib import Path
import pytest
from src.agents import stakeholder as agent
from src.tools.web_search import WebEvidenceStore, FetchedPage
from tests.test_r3_web import HTML


def make_store(tmp_path, searcher=None, transport=None):
    return WebEvidenceStore('test-run', root=tmp_path,
        searcher=searcher or (lambda q, k: [{'url': 'https://example.org/a', 'title': 'candidate', 'content': 'UNVERIFIED SNIPPET'}]),
        transport=transport or (lambda url, **k: FetchedPage(url, 200, {'content-type': 'text/html'}, HTML.encode())))


def claim(eid, **changes):
    return {'text': 'Kim described TurboQuant integration work.', 'technology': 'TurboQuant',
        'kind': 'fact', 'evidence_ids': [eid], 'explanation': '',
        'stakeholder_group': 'developers', 'actor': 'Kim', 'statement_date': None,
        'context': 'Deployment notes about serving kernels.', **changes}


def factory(store, writer):
    assert hasattr(agent, 'make_stakeholder'), 'new run-bound stakeholder factory is missing'
    return agent.make_stakeholder(store, writer=writer)


def writer_with_claims(store, **changes):
    def writer(**kwargs):
        assert 'UNVERIFIED SNIPPET' not in kwargs['context']
        eid = next(eid for eid, e in store.manifest['evidence'].items() if 'Developer Kim' in e['quote'])
        return {'claims': [claim(eid, **changes)], 'gaps': []}
    return writer


def test_new_assessment_contains_linked_claims_and_only_own_state_keys(tmp_path):
    s = make_store(tmp_path)
    out = factory(s, writer_with_claims(s))({'run_id': 'test-run', 'run_config': {}})
    assert set(out) == {'stakeholder', 'trace'}
    a = out['stakeholder']
    assert a['status'] == 'partial'
    assert a['claims'][0]['kind'] == 'fact'
    assert a['claims'][0]['actor'] == 'Kim'
    assert a['claims'][0]['statement_date'] is None
    ids = {e['evidence_id'] for e in a['evidence']}
    assert set(a['claims'][0]['evidence_ids']) <= ids
    assert len(a['gaps']) == 7
    assert all(e['run_id'] == 'test-run' for e in a['evidence'])
    assert Path(s.directory.parent / 'stakeholder.json').is_file()
    assert all('timestamp' in t and 'attempt' in t for t in out['trace'])


def test_failed_fetch_never_reaches_writer_as_fact(tmp_path):
    s = make_store(tmp_path, transport=lambda url, **k: FetchedPage(url, 403, {}, b''))
    out = factory(s, lambda **k: pytest.fail('no evidence means no LLM call'))({'run_id': 'test-run'})
    assert out['stakeholder']['claims'] == []
    assert len(out['stakeholder']['gaps']) == 8
    assert out['stakeholder']['status'] == 'partial'
    assert s.manifest['usage']['search_calls'] == 16


def test_unknown_evidence_gets_one_repair_then_failed(tmp_path):
    s = make_store(tmp_path)
    attempts = []
    def writer(**kw):
        attempts.append(kw['feedback'])
        return {'claims': [claim('invented')], 'gaps': []}
    out = factory(s, writer)({'run_id': 'test-run'})
    assert len(attempts) == 2
    assert attempts[1]
    assert out['stakeholder']['status'] == 'failed'
    assert out['stakeholder']['claims'] == []
    assert any(t['status'] == 'failed' for t in out['trace'])


def test_repair_uses_allowed_evidence_and_keeps_inference_distinct(tmp_path):
    s = make_store(tmp_path)
    calls = []
    def writer(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return {'claims': [claim('invented')], 'gaps': []}
        return writer_with_claims(s, kind='inference', explanation='Kernel integration creates an adoption task.')( **kw)
    out = factory(s, writer)({'run_id': 'test-run'})
    assert out['stakeholder']['claims'][0]['kind'] == 'inference'
    assert out['stakeholder']['claims'][0]['explanation']
    assert out['stakeholder']['status'] == 'partial'


def test_number_missing_from_evidence_uses_the_repair_and_names_it(tmp_path):
    """개선 후 실행 20261021-123030-55c6fb — 35.7% 를 조건 없이 써서 실행 전체가 failed 였다."""
    s = make_store(tmp_path)
    calls = []
    def writer(**kw):
        calls.append(kw["feedback"])
        text = "Kim said throughput rose 9배." if len(calls) == 1 else "Kim described integration."
        return writer_with_claims(s, text=text)(**kw)
    out = factory(s, writer)({"run_id": "test-run"})
    assert "9배" in calls[1] and len(calls) <= 3      # 초안 · 수치 수정 · 보완 후 재작성
    assert out["stakeholder"]["status"] == "partial"



def test_claim_still_wrong_on_numbers_after_repair_becomes_a_gap_not_a_failed_run(tmp_path):
    """수치만 틀린 Claim 하나 때문에 나머지 정상 Claim 까지 버리지 않는다(인용 오류는 여전히 전체 실패)."""
    s = make_store(tmp_path)
    calls = []
    def writer(**kw):
        calls.append(kw["feedback"])
        eid = next(eid for eid, e in s.manifest["evidence"].items() if "Developer Kim" in e["quote"])
        return {"claims": [claim(eid),
                           claim(eid, text="Kim said adoption rose 9배.", stakeholder_group="adopters")],
                "gaps": []}
    a = factory(s, writer)({"run_id": "test-run"})["stakeholder"]
    assert "9배" in calls[1]                                    # 수정 기회는 먼저 쓴다
    assert a["status"] == "partial"
    assert [c["stakeholder_group"] for c in a["claims"]] == ["developers"]
    dropped = [g for g in a["gaps"] if g["item"] == "adopters" and g["technology"] == "TurboQuant"]
    assert dropped and "9배" in dropped[0]["reason"]


def test_inference_without_explanation_and_fact_without_actor_fail(tmp_path):
    for changes in [{'kind': 'inference', 'explanation': ''}, {'actor': ''}]:
        s = make_store(tmp_path)
        out = factory(s, writer_with_claims(s, **changes))({'run_id': 'test-run'})
        assert out['stakeholder']['status'] == 'failed'


def test_supplement_search_can_recover_missing_body(tmp_path):
    calls = []
    def searcher(q, k):
        calls.append(q)
        return [] if len(calls) <= 8 else [{'url': 'https://example.org/a'}]
    s = make_store(tmp_path, searcher=searcher)
    out = factory(s, writer_with_claims(s))({'run_id': 'test-run'})
    assert out['stakeholder']['claims']
    assert len(calls) == 16
    assert any(t.get('action') == 'supplement_search' for t in out['trace'])


def test_run_id_mismatch_is_not_silently_accepted(tmp_path):
    s = make_store(tmp_path)
    node = factory(s, writer_with_claims(s))
    with pytest.raises(ValueError, match='run_id'):
        node({'run_id': 'different'})


def test_missing_run_id_fails_before_network():
    with pytest.raises(ValueError, match='run_id'):
        agent.stakeholder({})


def test_other_technology_evidence_is_blocked(tmp_path):
    def searcher(q, k):
        return [{'url': 'https://example.org/a'}] if q.startswith('ITME') else []
    s = make_store(tmp_path, searcher=searcher)
    out = factory(s, writer_with_claims(s))({'run_id': 'test-run'})
    assert out['stakeholder']['status'] == 'failed'
    assert not out['stakeholder']['claims']


def test_writer_does_not_receive_unbounded_model_context(tmp_path):
    s = make_store(tmp_path)
    def writer(**kw):
        assert len(kw['context']) <= 1000
        return {'claims': [], 'gaps': []}
    out = factory(s, writer)({'run_id': 'test-run', 'run_config': {'web_context_chars': 1000}})
    assert out['stakeholder']['status'] != 'failed'
    assert any(t.get('context_omitted') for t in out['trace'])


def test_structurally_complete_assessment_still_requires_human_review(tmp_path):
    s = make_store(tmp_path)
    def writer(**kw):
        eid = next(eid for eid, e in s.manifest['evidence'].items() if 'Developer Kim' in e['quote'])
        return {'claims': [claim(eid, technology=t, stakeholder_group=g)
                           for t in ('TurboQuant', 'ITME')
                           for g in ('competitors', 'adopters', 'developers', 'investors')], 'gaps': []}
    out = factory(s, writer)({'run_id': 'test-run'})
    assert out['stakeholder']['status'] == 'completed'
    validation = json.loads((s.directory.parent / 'stakeholder-validation.json').read_text())
    assert validation['review_status'] == 'pending'
    assert out['stakeholder']['gaps'] == []


def test_empty_model_object_is_repaired_then_failed(tmp_path):
    s = make_store(tmp_path)
    calls = []
    def writer(**kw):
        calls.append(kw)
        return {}
    out = factory(s, writer)({'run_id': 'test-run'})
    assert out['stakeholder']['status'] == 'failed'
    assert len(calls) == 2


def test_total_search_outage_is_failed_not_missing_public_evidence(tmp_path):
    def unavailable(q, k):
        raise RuntimeError('service down')
    s = make_store(tmp_path, searcher=unavailable)
    out = factory(s, lambda **kw: pytest.fail('no evidence'))({'run_id': 'test-run'})
    assert out['stakeholder']['status'] == 'failed'
    log = json.loads((s.directory.parent / 'stakeholder-validation.json').read_text())
    assert log['errors']


def test_draft_gap_triggers_one_supplement_even_when_generic_body_was_fetched(tmp_path):
    queries = []
    def searcher(q, k):
        queries.append(q)
        return [{'url': 'https://example.org/a'}]
    s = make_store(tmp_path, searcher=searcher)
    calls = []
    def writer(**kw):
        calls.append(kw)
        return {'claims': [], 'gaps': [{'technology': 'ITME', 'item': 'adopters',
            'reason': '페이지는 있지만 도입 기업의 직접 반응이 없음'}]}
    out = factory(s, writer)({'run_id': 'test-run'})
    assert len(calls) == 2
    assert len(queries) == 16
    assert sum(t.get('action') == 'supplement_search' for t in out['trace']) == 8
    assert out['stakeholder']['status'] == 'partial'


def test_repair_feedback_names_the_reason_for_each_rejected_id():
    """거부 사유를 구분해 알려야 수정 1회를 제대로 쓴다."""
    from src.agents.stakeholder import _diagnose

    allowed = {'TurboQuant': {'e-web-aaaaaaaaaaaa'}, 'ITME': {'e-web-bbbbbbbbbbbb'}}
    visible = {'e-web-aaaaaaaaaaaa', 'e-web-bbbbbbbbbbbb', 'e-web-cccccccccccc'}
    everything = {eid: {} for eid in visible}

    text = _diagnose({'e-web-bbbbbbbbbbbb'}, 'TurboQuant', allowed, visible, everything)
    assert 'ITME 자료로 수집됐다' in text

    text = _diagnose({'e-web-aaaaaaaaaaa'}, 'TurboQuant', allowed, visible, everything)
    assert '잘못 옮겨 적었다' in text and 'e-web-aaaaaaaaaaaa' in text

    text = _diagnose({'e-web-zzzzzzzzzzzz'}, 'TurboQuant', allowed, visible, everything)
    assert '이번 실행에 없는 ID' in text

    text = _diagnose({'e-web-dddddddddddd'}, 'TurboQuant', allowed, visible,
                     dict(everything, **{'e-web-dddddddddddd': {}}))
    assert '모델 입력에서 제외된' in text


def test_web_ids_are_short_enough_to_copy():
    """설계 이유: 64자 hex 를 모델이 옮겨 적다 끝 글자를 흘려 실행이 실패했다."""
    from src.tools.web_store import ID_CHARS, short_id

    assert ID_CHARS == 12
    assert len(short_id('e-web-', 'x')) == len('e-web-') + 12


def test_error_boilerplate_never_becomes_evidence():
    """로딩 실패 문구가 인용 가능한 원문이 되면 안 된다."""
    from src.tools.web_fetch import FetchedPage, extract_body

    html = (b'<html><head><title>Discussion</title></head><body><article>'
            b'<p>Uh oh! There was an error while loading. Please reload this page.</p>'
            b'<p>' + b'A real paragraph with enough substance to be quoted. ' * 4 + b'</p>'
            b'</article></body></html>')
    page = FetchedPage(url='https://example.org/d', status_code=200,
                       headers={'content-type': 'text/html'}, content=html)

    texts = [text for _, text in extract_body(page)['paragraphs']]
    assert not any('error while loading' in text.lower() for text in texts)
    assert any('real paragraph' in text for text in texts)
