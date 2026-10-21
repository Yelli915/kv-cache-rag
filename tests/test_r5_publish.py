"""제출본 저장 노드 — 담당: R5 (설계서 §9, 부록 A 의 `Z` 노드)

PDF 렌더링 자체는 fixture 로 대체한다. 확인하는 것은 계약이다 — 어디에 무엇을 남기고
무엇을 막는지. 노드 아래의 가드(`build_submission`)와 품질 점검(`quality_checks`)도 여기서 본다.
"""

import json

import pytest

from src.output import pdf as node

REPORT = """# KV cache 최적화 기술 다관점 평가

## SUMMARY

네 관점에서 공통으로 확인된 사항은 다음과 같다.

## 4. 관점별 평가

| 접근 | 기술 |
|---|---|
| 압축 | TurboQuant |

## 6. 한계

- 확인된 근거 공백

## REFERENCE

- 원문 목록
"""


@pytest.fixture
def state(tmp_path):
    return {"run_id": "r5-test", "run_config": {"runs_dir": str(tmp_path)}, "report": REPORT}


@pytest.fixture
def rendered(monkeypatch):
    calls = []

    def fake_render(markdown_path, pdf_path):
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.7 fixture")
        calls.append(pdf_path)

    monkeypatch.setattr(node, "render_pdf", fake_render)
    return calls


def test_saves_markdown_and_submission_pdf(state, rendered, tmp_path):
    update = node.publish(state)

    directory = tmp_path / "r5-test"
    assert (directory / "report.md").read_text(encoding="utf-8") == REPORT
    assert update["report_paths"] == [str(directory / "report.md"),
                                      str(directory / "final" / node.FINAL_FILENAME)]
    assert update["trace"][0]["node"] == "publish"


def test_submission_path_is_separate_from_the_draft(state, rendered, tmp_path):
    """§8 — 검토용 초안과 제출본 경로를 구분한다."""
    node.publish(state)

    final = tmp_path / "r5-test" / "final" / node.FINAL_FILENAME
    assert final.exists()
    assert final.name == "RAG-Output_판교_8반_권수진+권예리+김민+박인기+정승원.pdf"


def test_validation_errors_block_the_submission(state, rendered, tmp_path):
    """§8 — 무효 인용이 남으면 제출용 출력을 막는다."""
    state["validation"] = {"errors": [{"node": "market", "kind": "없는 인용 ID"}]}

    update = node.publish(state)

    assert rendered == []
    assert update["report_paths"] == [str(tmp_path / "r5-test" / "report.md")]
    result = json.loads((tmp_path / "r5-test" / "submission.json").read_text())
    assert result["generated"] is False and "validation_errors" in result["reason"]


def test_layout_check_failure_blocks_the_submission(state, rendered, tmp_path):
    """§9 — 변환 후 SUMMARY 분량·한글 글꼴·표 잘림·참고문헌 위치를 확인한다."""
    state["report"] = REPORT.replace("## REFERENCE", "## 참고 없음")

    update = node.publish(state)

    assert rendered == []
    assert len(update["report_paths"]) == 1
    result = json.loads((tmp_path / "r5-test" / "submission.json").read_text())
    assert result["reason"] == "PDF 품질 사전 점검 실패"
    assert result["checks"] and any(not check["passed"] for check in result["checks"])


def test_missing_pdf_dependency_does_not_kill_the_graph(state, monkeypatch, tmp_path):
    """weasyprint·pango 가 없어도 Markdown 까지는 남기고 정상 종료한다."""
    def boom(markdown_path, pdf_path):
        raise RuntimeError("PDF 의존성 미설치")

    monkeypatch.setattr(node, "render_pdf", boom)
    update = node.publish(state)

    assert update["report_paths"] == [str(tmp_path / "r5-test" / "report.md")]
    assert update["trace"][0]["status"] == "partial"
    result = json.loads((tmp_path / "r5-test" / "submission.json").read_text())
    assert result["reason"] == "RuntimeError: PDF 의존성 미설치"


def test_system_library_failure_does_not_kill_the_graph(state, monkeypatch, tmp_path):
    """weasyprint 가 깔려 있어도 libpango 가 없으면 OSError 가 난다. 실행을 버리지 않는다."""
    def boom(markdown_path, pdf_path):
        raise OSError("cannot load library 'libpango-1.0-0'")

    monkeypatch.setattr(node, "render_pdf", boom)
    update = node.publish(state)

    assert update["report_paths"] == [str(tmp_path / "r5-test" / "report.md")]
    assert update["trace"][0]["status"] == "partial"
    assert "libpango" in json.loads((tmp_path / "r5-test" / "submission.json").read_text())["reason"]


def test_basic_report_passes_every_quality_check():
    assert all(check["passed"] for check in node.quality_checks(REPORT))


def test_missing_markdown_blocks_submission(tmp_path):
    result = node.build_submission(markdown_path=tmp_path / "missing.md",
                                   final_path=tmp_path / "final.pdf")

    assert result["generated"] is False and "Markdown 파일 없음" in result["reason"]
