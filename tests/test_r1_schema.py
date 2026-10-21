"""공용 schema 가 R3 실물 출력을 그대로 통과시키는지 본다 — 담당: R1

fixture 는 R3 합성 데이터 실행(`scripts.r3_smoke`) 결과다. 새 모델을 만들어 맞추는 게 아니라
이미 돌아가는 노드의 출력이 계약이므로, round-trip 이 깨지면 계약이 깨진 것이다.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schema import Assessment, Claim, Event, Gap, Source

R3 = Path("tests/fixtures/r3")


def load(name):
    return json.loads((R3 / name).read_text(encoding="utf-8"))


def test_r3_assessment_round_trips_unchanged():
    raw = load("stakeholder.json")
    assert Assessment.model_validate(raw).model_dump() == raw


def test_r3_failed_assessment_validates():
    raw = load("failed-assessment-output.json")["stakeholder"]
    assessment = Assessment.model_validate(raw)
    assert assessment.status == "failed" and assessment.claims == []


def test_r3_trace_lines_validate():
    lines = (R3 / "stakeholder-trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines and all(Event.model_validate(json.loads(line)) for line in lines)


def test_broken_citation_chain_is_rejected():
    raw = load("stakeholder.json")
    raw["evidence"] = [e for e in raw["evidence"]
                       if e["evidence_id"] != raw["claims"][0]["evidence_ids"][0]]
    with pytest.raises(ValidationError, match="없는 evidence"):
        Assessment.model_validate(raw)


def test_orphan_evidence_is_rejected():
    raw = load("stakeholder.json")
    raw["evidence"][0]["source_id"] = "web-nonexistent"
    with pytest.raises(ValidationError, match="sources 에 없다"):
        Assessment.model_validate(raw)


def test_failed_status_cannot_keep_claims():
    raw = load("stakeholder.json")
    raw["status"] = "failed"
    with pytest.raises(ValidationError, match="claims 가 남아"):
        Assessment.model_validate(raw)


def test_inference_requires_explanation():
    raw = load("stakeholder.json")["claims"][0]
    with pytest.raises(ValidationError, match="explanation"):
        Claim.model_validate(raw | {"kind": "inference", "explanation": ""})
    Claim.model_validate(raw | {"kind": "inference", "explanation": "전제: ..."})


def test_index_source_and_free_form_gap_pass():
    """웹이 아닌 R2 색인 출처와, 이해관계자가 아닌 역할의 gap 도 같은 모델을 통과한다."""
    Source.model_validate({"source_id": "itme-paper", "run_id": "r1-test",
                           "collection": "papers_core", "allowed_uses": ["maturity"],
                           "scope": "direct", "applies_to": ["ITME"], "page": 7})
    Gap.model_validate({"role": "maturity", "technology": "ITME",
                        "item": "TRL 6 실증 환경 근거", "reason": "공개 자료 미확인"})


def test_event_needs_a_name():
    with pytest.raises(ValidationError, match="node 또는 tool"):
        Event.model_validate({"status": "ok"})
