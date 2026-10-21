"""State — 담당: R1 (설계서 §7 표 그대로)

§7 표는 14행이고 그중 세 행이 묶음 표기(`run_id / run_config`,
`validation / review_status`, `report / report_paths`)다. Python 필드로 풀어 17개다.

**병렬 구간에서는 각 평가 노드가 자기 결과 키만 쓴다.** 웹 자료는 전역에 쌓지 않고 각
Assessment 에 넣는다. 네 결과가 모두 도착하면 `collect_evidence` 가 출처와 근거를 한 번에
병합한다. 병렬 노드는 공용 gaps 나 레지스트리를 직접 수정하지 않는다.

`trace` 만 append reducer 를 쓴다. `gaps` 는 합류 후 수집하고 종합 후 순차 병합한다 —
병렬이 아니라 순서가 정해져 있어 reducer 가 필요 없다.

값은 전부 dict/list 다. 형식은 `src/schema.py` 가 정의하고 `collect_evidence` 가 검증한다.
"""

import operator
from pathlib import Path
from typing import Annotated, TypedDict


# 평가 노드가 Assessment 를 쓰는 State 키. 병합·검증·검토·참고문헌·보고서가 모두 이 순서로 읽는다.
ASSESSMENT_NODES = ("research", "maturity", "market", "stakeholder", "domain_assessment")


class ReportState(TypedDict, total=False):
    # ── 초기화 ──
    run_id: Annotated[str, "실행 식별자. runs/<run_id>/ 아래에 원문 스냅샷·결과를 남긴다"]
    run_config: Annotated[dict, "기술·모델·자료 버전·호출 한도. app.py 가 만들고 setup 이 자료 버전을 채운다"]

    # ── 기술 조사 + 병렬 평가: 각 노드가 자기 키에만 쓴다 ──
    research: Annotated[dict, "Assessment — 기술 조사 결과와 근거"]
    maturity: Annotated[dict, "Assessment — TRL 평가 결과"]
    market: Annotated[dict, "Assessment — 시장성 평가 결과"]
    stakeholder: Annotated[dict, "Assessment — 이해관계자 평가 결과"]
    domain_assessment: Annotated[dict, "Assessment — 도메인 평가 결과"]

    # ── 합류 ──
    source_registry: Annotated[dict, "{source_id: Source} — collect_evidence. 서지정보와 저장 위치"]
    evidence_registry: Annotated[dict, "{evidence_id: Evidence} — collect_evidence. 인용 구절과 원문 위치"]
    gaps: Annotated[list[dict], "[Gap] — 합류 후 수집, 종합 후 순차 병합"]

    # ── 종합 · 검증 ──
    synthesis: Annotated[dict, "Synthesis — 일치·차이·가설과 주장별 근거 연결"]
    validation: Annotated[dict, "검증 단계. 오류와 내용 검토 결과"]
    review_status: Annotated[str, "검증 단계. pending 이면 검토용 초안만 저장한다"]

    # ── 제어 · 출력 ──
    run_status: Annotated[str, "제어 단계. running / partial / completed / failed"]
    report: Annotated[str, "출력 단계. Markdown 본문"]
    report_paths: Annotated[list[str], "출력 단계. Markdown·PDF 저장 경로"]

    # ── 누적 ──
    trace: Annotated[list[dict], operator.add]


def run_dir(state) -> Path:
    """실행 저장 루트 `runs/<run_id>/`. R3 의 웹 스냅샷도 이 아래에 쌓인다.

    State 에서 바로 나오는 값이라 여기 둔다. graph 에 두면 노드가 graph 를 import 하게 돼
    순환이 생긴다.
    """
    config = state.get("run_config") or {}
    return Path(config.get("runs_dir", "runs")) / state["run_id"]
