"""Markdown → PDF — 담당: R5

검토용 Markdown을 PDF로 변환하고 제출 전 품질 점검을 수행한다.

원칙:
- validation error가 있으면 제출본을 생성하지 않는다.
- 검토용 초안과 최종 제출본 경로를 분리한다.
- SUMMARY / 한글 / 표 / REFERENCE 기본 품질을 점검한다.
- PDF 라이브러리 문제가 전체 그래프 실행을 중단시키지 않는다(`publish`).

PDF 의존성: `uv sync --extra pdf` (macOS 는 `brew install pango`).
"""

from pathlib import Path

from src.state import run_dir
from src.tools.web_store import save_json

FINAL_FILENAME = "RAG-Output_판교_8반_권수진+권예리+김민+박인기+정승원.pdf"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body {
    font-family: "Apple SD Gothic Neo", "AppleSDGothicNeo", "NanumGothic", "Malgun Gothic", sans-serif;
    font-size: 10.5pt; line-height: 1.6;
}
h1 { font-size: 18pt; }
h2 { font-size: 14pt; margin-top: 1.4em; page-break-after: avoid; }
h3 { font-size: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; table-layout: fixed; }
th, td {
    border: 1px solid #999; padding: 4px 6px; text-align: left;
    overflow-wrap: anywhere; word-break: break-word;
}
tr { page-break-inside: avoid; }
code { font-size: 9pt; overflow-wrap: anywhere; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
"""


def _extract_section(markdown_text: str, heading: str, next_heading: str) -> str:
    """Markdown의 두 heading 사이 내용을 가져온다."""
    start = markdown_text.find(heading)
    if start == -1:
        return ""
    start += len(heading)
    end = markdown_text.find(next_heading, start)
    return markdown_text[start:end if end != -1 else None].strip()


def check_summary(markdown_text: str) -> tuple[bool, str]:
    """SUMMARY가 과도하게 긴지 간단히 점검한다.

    실제 반 페이지 여부는 렌더링 환경에 따라 달라지므로 문자 수는 사전 경고용 heuristic이다.
    """
    summary = _extract_section(markdown_text, "## SUMMARY", "## 1. 분석 배경")
    if not summary:
        return False, "SUMMARY를 찾지 못함"
    chars = len(summary.replace("\n", " ").strip())
    if chars > 1200:
        return False, f"SUMMARY가 길 수 있음 ({chars}자)"
    return True, f"SUMMARY 길이 확인 ({chars}자)"


def check_korean(markdown_text: str) -> tuple[bool, str]:
    if not any("가" <= char <= "힣" for char in markdown_text):
        return False, "한글 텍스트를 찾지 못함"
    return True, "한글 텍스트 존재 확인"


def check_tables(markdown_text: str) -> tuple[bool, str]:
    """Markdown 표 존재 여부와 과도하게 긴 행을 점검한다."""
    table_lines = [line for line in markdown_text.splitlines() if line.strip().startswith("|")]
    if not table_lines:
        return True, "Markdown 표 없음"
    longest = max(len(line) for line in table_lines)
    if longest > 500:
        return False, f"매우 긴 표 행 존재 ({longest}자)"
    return True, f"표 기본 점검 완료 ({len(table_lines)}개 행)"


def check_reference(markdown_text: str) -> tuple[bool, str]:
    """REFERENCE가 존재하고 본문 뒤쪽에 위치하는지 확인한다."""
    pos = markdown_text.rfind("## REFERENCE")
    if pos == -1:
        return False, "REFERENCE 섹션 없음"
    if pos / max(len(markdown_text), 1) < 0.6:
        return False, "REFERENCE가 문서 후반부에 있지 않음"
    return True, "REFERENCE 위치 확인"


CHECKS = {"summary": check_summary, "korean": check_korean,
          "table": check_tables, "reference": check_reference}


def quality_checks(markdown_text: str) -> list[dict]:
    """제출 전 자동 품질 점검."""
    return [{"check": name, "passed": passed, "message": message}
            for name, fn in CHECKS.items() for passed, message in [fn(markdown_text)]]


def render_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """Markdown을 PDF로 변환한다."""
    try:
        import markdown
        from weasyprint import CSS as WCSS
        from weasyprint import HTML
    except ImportError:
        raise RuntimeError("PDF 의존성 미설치. uv sync --extra pdf 를 실행하고, "
                           "macOS에서는 필요 시 brew install pango 를 실행하세요.")

    html = markdown.markdown(markdown_path.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code"])
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(pdf_path, stylesheets=[WCSS(string=CSS)])


def _result(reason: str, *, checks=(), validation_errors=(), path=None) -> dict:
    return {"generated": path is not None, "reason": reason,
            "validation_errors": list(validation_errors), "checks": list(checks),
            "path": str(path) if path else None}


def build_submission(markdown_path: Path, final_path: Path,
                     validation_errors: list | None = None) -> dict:
    """검증 통과 시에만 최종 제출 PDF를 만든다."""
    if validation_errors:
        return _result("validation_errors 존재", validation_errors=validation_errors)
    if not markdown_path.exists():
        return _result(f"Markdown 파일 없음: {markdown_path}")

    checks = quality_checks(markdown_path.read_text(encoding="utf-8"))
    if not all(check["passed"] for check in checks):
        return _result("PDF 품질 사전 점검 실패", checks=checks)

    render_pdf(markdown_path, final_path)
    return _result("ok", checks=checks, path=final_path)


def publish(state) -> dict:
    """레이아웃 확인 후 제출본 저장 (설계서 §9, 부록 A 의 `Z` 노드).

    Markdown 을 실행 저장소에 남기고, 검증을 통과했을 때만 제출본 PDF 를 만든다.
    §8 — 검토용 초안은 별도 경로에 저장하며 제출본과 구분한다. 제출본은
    `runs/<run_id>/final/` 에 둔다.

    PDF 의존성(weasyprint·pango)이 없어도 그래프를 죽이지 않는다. 그 사실을
    submission.json 과 trace 에 남기고 Markdown 경로만 돌려준다.
    """
    directory = run_dir(state)
    markdown_path = directory / "report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(state.get("report", ""), encoding="utf-8")

    errors = (state.get("validation") or {}).get("errors") or []
    try:
        result = build_submission(markdown_path, directory / "final" / FINAL_FILENAME, errors)
    except Exception as exc:
        # 선택 의존성이다. weasyprint 가 깔려 있어도 libpango 같은 시스템 라이브러리가
        # 없으면 OSError 가 난다. PDF 하나 때문에 실행 전체를 버리지 않는다.
        result = _result(f"{type(exc).__name__}: {exc}".strip()[:300],
                         checks=quality_checks(state.get("report", "")))

    save_json(directory / "submission.json", result)

    return {
        "report_paths": [str(markdown_path)] + ([result["path"]] if result["generated"] else []),
        "trace": [{
            "node": "publish",
            "status": "ok" if result["generated"] else "partial",
            "attempt": 1,
            "reason": result["reason"],
            "checks": {check["check"]: check["passed"] for check in result["checks"]},
        }],
    }
