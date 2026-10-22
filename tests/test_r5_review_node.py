"""내용 검토 노드와 worksheet — 담당: R5 (설계서 §8, 부록 A 의 `R` 노드)

형식 검사와 사람의 내용 검토가 한 노드다. **ID 대조만으로 자동 통과시키지 않는다.**
"""

import pytest

from src.output import review as node
from tests.helpers import assessment, fill_review


def make_state(tmp_path, run_id="r5-test", **config):
    return {"run_id": run_id, "run_config": {"runs_dir": str(tmp_path), **config},
            "market": assessment("market", run_id=run_id)}


@pytest.fixture
def state(tmp_path):
    return make_state(tmp_path)


def worksheet(tmp_path, run_id="r5-test"):
    return tmp_path / run_id / "review.csv"


def reviewed(tmp_path, run_id="run-a"):
    """한 실행을 검토하고 확인으로 채운 뒤 다시 돌려 판정을 원장에 올린다."""
    state = make_state(tmp_path, run_id)
    node.review(state)
    fill_review(worksheet(tmp_path, run_id), "확인")
    node.review(state)


# ── worksheet 행 ──────────────────────────────────────────────────────────────

def test_row_pairs_claim_with_evidence_and_leaves_verdict_blank(state):
    rows = node.build_review_rows(state)

    assert len(rows) == 1
    row = rows[0]
    assert (row["claim_id"], row["node"], row["technology"]) == ("claim-market", "market", "ITME")
    assert (row["evidence_id"], row["source_id"], row["evidence_quote"]) == ("e-market", "s-market", "합성 인용")
    # §3 — Evidence 의 페이지·문단 위치가 검토자가 원문을 찾을 단서다
    assert row["location"] == "p.1"
    assert row["source_type"] == "ecosystem"   # 직접 근거와 인접 생태계를 가르는 단서
    assert row["review_result"] == row["reviewer"] == row["review_comment"] == ""


def test_row_location_falls_back_to_the_source_url(state):
    del state["market"]["evidence"][0]["location"]

    assert node.build_review_rows(state)[0]["location"] == "https://example.org/market"


# ── 노드 ─────────────────────────────────────────────────────────────────────

def test_first_pass_writes_worksheet_and_waits(state, tmp_path):
    """판정이 비어 있으면 통과시키지 않는다 — 검토 대기."""
    update = node.review(state)

    assert set(update) == {"validation", "review_status", "trace"}
    assert update["review_status"] == "pending"
    assert update["validation"]["pending_claims"] == ["claim-market"]
    assert worksheet(tmp_path).exists()


def test_filled_worksheet_passes(state, tmp_path):
    node.review(state)
    fill_review(worksheet(tmp_path), "확인")

    update = node.review(state)
    assert update["review_status"] == "passed"
    assert update["validation"]["reviewers"] == ["권예리"]
    assert update["validation"]["claim_verdicts"]["claim-market"]["verdict"] == "확인"


def test_rejected_claim_passes_review_but_is_listed_for_exclusion(state, tmp_path):
    """§8 — 부결된 주장은 판정된 것이다. 사실 서술에서 빼도록 목록으로 남긴다."""
    node.review(state)
    fill_review(worksheet(tmp_path), "부결")

    update = node.review(state)
    assert update["review_status"] == "passed"
    assert update["validation"]["rejected_claims"] == ["claim-market"]


def test_unknown_verdict_becomes_an_error(state, tmp_path):
    node.review(state)
    fill_review(worksheet(tmp_path), "글쎄요")

    errors = node.review(state)["validation"]["errors"]
    assert any(error["kind"] == "판정 값 미상" for error in errors)


def test_existing_worksheet_is_not_overwritten(state, tmp_path):
    """재개할 때 사람이 채운 판정을 지우지 않는다."""
    node.review(state)
    fill_review(worksheet(tmp_path), "확인")
    before = worksheet(tmp_path).read_text(encoding="utf-8")

    node.review(state)
    assert worksheet(tmp_path).read_text(encoding="utf-8") == before


def test_format_errors_survive_into_validation(state):
    """형식 검사 결과가 같은 validation 에 실린다 — final_check 이 이걸 본다."""
    state["market"]["claims"][0]["evidence_ids"] = ["e-missing"]

    errors = node.review(state)["validation"]["errors"]
    assert any(error["kind"] == "Evidence 없음" for error in errors)


def test_review_reader_groups_rows_by_claim(state, tmp_path, capsys):
    """읽기용 스크립트는 판정을 바꾸지 않는다. Claim 단위로 묶어 보여 주기만 한다."""
    from scripts.review_reader import show

    node.review(state)
    path = worksheet(tmp_path)
    before = path.read_text(encoding="utf-8")

    assert show("r5-test", runs_dir=tmp_path, width=80) == 1
    assert path.read_text(encoding="utf-8") == before      # 읽기 전용

    out = capsys.readouterr().out
    assert "claim-market" in out and "미판정" in out and "market 주장" in out

    fill_review(path, "확인")
    show("r5-test", runs_dir=tmp_path, width=80)
    assert "확인 · 권예리" in capsys.readouterr().out
    assert show("r5-test", runs_dir=tmp_path, only_pending=True, width=80) == 0


# ── 판정 원장 ────────────────────────────────────────────────────────────────

def test_verdict_carries_from_the_ledger(tmp_path):
    """§8 — 사람이 읽은 것이 그대로일 때만 판정을 옮긴다."""
    reviewed(tmp_path)

    update = node.review(make_state(tmp_path, "run-b"))

    assert update["review_status"] == "passed"
    assert update["validation"]["carried_claims"] == ["claim-market"]
    assert update["validation"]["claim_verdicts"]["claim-market"]["carried_from"] == "run-a"


def test_changed_claim_text_is_reviewed_again(tmp_path):
    reviewed(tmp_path)

    changed = make_state(tmp_path, "run-c")
    changed["market"]["claims"][0]["text"] = "market 주장 (문장이 바뀌었다)"
    update = node.review(changed)

    assert update["review_status"] == "pending"
    assert update["validation"]["carried_claims"] == []


def test_no_carry_review_forces_a_fresh_review(tmp_path):
    reviewed(tmp_path)

    update = node.review(make_state(tmp_path, "run-d", no_carry_review=True))

    assert update["review_status"] == "pending"


def test_ledger_records_what_was_judged(tmp_path):
    """지문만 남기면 나중에 무엇을 보고 판정했는지 확인할 수 없다."""
    reviewed(tmp_path)

    entry = next(iter(node.load_ledger(node.LEDGER).values()))
    assert entry["claim_text"] == "market 주장" and entry["reviewer"] == "권예리"
    assert entry["node"] == "market" and entry["evidence_ids"] == ["e-market"]
