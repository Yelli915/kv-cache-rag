"""기술 조사 — 담당: R4 | papers_core RAG | bind_document_search 바인딩

- 계약 준수: State 14필드 중 'research' 및 'trace' 단독 갱신
- 컬렉션 바인딩: bind_document_search("research") -> papers_core (scope=direct 전용 자동 강제)
- 핵심 임무: TurboQuant(SW 압축) 및 ITME(HW 메모리 접근) 시스템 계층, 적용 범위, 한계 도출
- 생성·인용 검증·Assessment 조립은 `common.evaluate` 가 한다.
"""

from typing import Any, Dict

from src.agents.common import TECHS, domain_of, evaluate, unique
from src.tools.docs import bind_document_search

INSTRUCTION = """당신은 데이터센터 및 클라우드 인프라 관점의 기술 조사 수석 연구원(R4)입니다.
선정 기술 2건(TurboQuant, ITME)의 논문 원문에서 다음 핵심 항목을 추출하여 객관적으로 기술하십시오.

[필수 분석 항목 — 기술마다]
1. 접근 방식: 어느 시스템 계층(SW 알고리즘/컴파일러 vs HW CXL/메모리 컨트롤러)에서 KV cache 병목을 해결하는가
2. 적용 범위: 어떤 모델 아키텍처, 하드웨어 스펙, 동시성 부하 조건에서 검증되었는가
3. 기술적 한계: 원문이 스스로 밝힌 제약 사항, 런타임 오버헤드, 미해결 과제

[작성 및 인용 규칙]
- 우열 판정(어느 기술이 더 우월하다는 승자 판정)은 엄격히 금지합니다. 두 접근법의 본질적 차이를 기술하십시오.
- 사실(fact) 주장은 반드시 본문 내에 검색된 청크의 12자리 hex ID를 `[12자리hex]` 형식으로 인용하십시오.
- scope=comparison 문서는 계열 비교에만 사용하며, 선정 기술의 성능 주장 근거로 대체하지 마십시오.

[성능 수치 해석 규칙 — 설계서 2절]
아래 수치는 인용한 근거 청크에 실제로 있을 때만 쓰고, 쓸 때는 괄호 안의 조건을 같은 문단에
함께 적으십시오. 조건에 들어가는 실험 설정값(동시 대화 수, 메모리 용량, 턴 수 등)도 근거에
있는 것만 씁니다. 근거에 없으면 수치도 조건도 본문에 쓰지 말고 gaps 에만 적으십시오.
이 규칙 자체를 본문에 옮겨 적지 마십시오.
- TurboQuant 3.5비트 (특정 모델·과제의 벤치마크 결과)
- TurboQuant 8배 가속 (H100 GPU의 어텐션 로짓 연산 한정, E2E 추론 처리량 아님)
- ITME 1.80배 (서론의 NVMe-oF 대비 처리량)
- ITME 35.7% (CPU 오프로드 대비, 호스트 메모리가 소진된 이후 구간)
- ITME 1.81배 (GPU 재계산 대비, 특정 턴 시점의 TTFT)"""

QUERIES = [
    "KV 캐시 메모리 병목 접근 방식과 핵심 알고리즘",
    "실험 설정, 검증 모델, 적용 범위 및 하드웨어 환경",
    "성능 한계, 오버헤드, 양자화 손실 및 미해결 제약사항",
]
WIDEN = " 벤치마크 처리량 지연시간 정확도 오버헤드 theoretical bound"


def research(state: Dict[str, Any]) -> Dict[str, Any]:
    """papers_core에서 양 기술 1차 근거를 수집하고 Assessment를 반환한다."""
    doc_search = bind_document_search("research")

    def search(queries):
        return [c for tech in TECHS for q in queries
                for c in doc_search.invoke({"query": q, "technology": tech, "top_k": 5})]

    chunks = unique(search(QUERIES))
    # 미확보 기술이 있으면 WIDEN 질의로 1회 보완 (부록 A — 노드 내부 최대 1회)
    covered = {t for c in chunks if c.get("scope") == "direct" for t in c.get("applies_to", [])}
    supplemental = [q + WIDEN for q in QUERIES] if len(covered) < len(TECHS) else []
    chunks = unique(chunks + search(supplemental))

    return evaluate(node="research", role="research", run_id=state.get("run_id", "default_run"),
                    domain=domain_of(state), instruction=INSTRUCTION, chunks=chunks,
                    gap_reason="제공된 근거에서 확인하지 못함",
                    extra_trace={"supplemental_searches": len(supplemental)})
