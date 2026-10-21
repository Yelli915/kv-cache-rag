"""도메인 적용 평가 — 담당: R4 | papers_core RAG + context(배경용) | bind_document_search 바인딩

- 계약 준수: State 14필드 중 'domain_assessment' 및 'trace' 단독 갱신
- 스키마 규격: src.schema.Assessment (Claim, Evidence, Source, Gap 체계)
- 컬렉션 바인딩:
    - bind_document_search("domain") -> papers_core (직접 실증 벤치마크)
    - bind_document_search("domain", "context") -> context (배경 설명용, scope=comparison/secondary)
- 핵심 임무: 대규모 동시성 및 비용 민감 환경에서의 단독 적용성 및 결합 가설 평가
- 가드레일 반영:
    1) TurboQuant 처리량 수치 부재 시 '처리량 근거 미확인' 공백 명시
    2) 서로 다른 실험 환경의 성능 배수 직접 비교 금지
    3) 결합 상호작용은 본문과 분리하여 '결합 가설' 절에 추론/가설로 명시
- 생성·인용 검증·Assessment 조립은 `common.evaluate` 가 한다. 결합 가설은 hypothesis 필드로
  따로 받아 kind=hypothesis Claim 이 된다.
"""

from typing import Any, Dict, List

from src.agents.common import TECHS, DomainDraft, evaluate
from src.tools.docs import bind_document_search

# 도메인 적용 평가 시스템 프롬프트 및 가드레일
INSTRUCTION = """당신은 데이터센터 및 클라우드 AI 추론 인프라 관점의 도메인 적용성 수석 평가관(R4)입니다.
대규모 동시성(High Concurrency)과 비용 민감도(Cost Efficiency)를 기준으로 기술 적용성을 평가하십시오.

[본문 필수 구성]
turboquant — TurboQuant 단독 적용성:
   - 알고리즘 압축을 통한 GPU 메모리 절감 및 서빙 배치 크기(Batch Size) 확장 효과
   - 양자화 비트 수 축소에 따른 정확도(Perplexity/Task Accuracy) 손실 폭
   - ★ [핵심 가드레일] TurboQuant 논문에 전체 추론 처리량(Throughput) 수치가 명시되지 않은 경우, 절대로 숫자를 지어내지 말고 "처리량 근거 미확인"으로 기술하고 공백에 남기십시오.

itme — ITME 단독 적용성:
   - CXL 메모리 풀링/오프로드를 통한 동시 세션(Concurrent Sessions) 수용 한계 확장
   - 호스트-디바이스 간 데이터 전송 지연(Latency) 및 TTFT(첫 토큰 지연) 영향도

[출처 및 인용 가드레일]
- 모든 성능 벤치마크는 반드시 모델명, 하드웨어 사양, 동시성 부하 조건을 병기하여 인용하십시오.
- 서로 다른 전제와 조건에서 측정된 배수 수치를 직접 비교(예: TurboQuant 8배 vs ITME 1.8배)하지 마십시오.
- collection=context 청크(서베이/배경 자료)는 인프라 배경 설명에만 사용하며, 기술의 성능 주장 근거로 인용하지 마십시오.
- scope=comparison 문서(InfiniGen, PIM 등)는 계열 비교에만 사용하고 선정 기술의 주장 근거로 대체하지 마십시오.
- 사실(fact)은 반드시 검색된 청크의 12자리 hex ID인 `[12자리hex]` 형식으로 인용하십시오.

[결합 상호작용 표기 규칙]
- 두 기술의 결합(SW 압축 + HW 확장) 시너지는 turboquant·itme 에 절대 섞지 마십시오.
- 결합 시너지는 hypothesis 에만 쓰고, turboquant·itme 로부터 도출한 '실측 근거가 없는 추론/가설'임을
  명시하며 전제가 된 근거를 인용하십시오."""

QUERIES = [
    "처리량 배치 크기 동시 요청 처리 serving batch throughput concurrency",
    "지연 정확도 손실 비용 오버헤드 latency accuracy degradation TTFT",
]


def domain_assessment(state: Dict[str, Any]) -> Dict[str, Any]:
    """papers_core와 context 문서를 검색하여 도메인 적용성 Assessment를 반환한다."""
    run_id = state.get("run_id", "default_run")
    domain = state.get(
        "domain",
        state.get("run_config", {}).get("domain", "데이터센터/클라우드 추론 인프라 (대규모 동시성, 비용 민감)"),
    )
    core_search = bind_document_search("domain")               # papers_core (direct/comparison)
    context_search = bind_document_search("domain", "context") # context (comparison/secondary)
    chunks: List[Dict[str, Any]] = []
    seen = set()

    def _add(hits):
        for c in hits:
            cid = c.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                chunks.append(c)

    for tech in TECHS:                                         # 실증 벤치마크
        for q in QUERIES:
            _add(core_search.invoke({"query": q, "technology": tech, "top_k": 4}))
    _add(context_search.invoke({                               # 인프라 배경·비용 구조
        "query": "데이터센터 추론 동시성 메모리 비용 구조 TCO",
        "technology": "both",
        "top_k": 3,
    }))

    return evaluate(node="domain_assessment", role="domain", run_id=run_id, domain=domain,
                    instruction=INSTRUCTION, chunks=chunks, schema=DomainDraft,
                    gap_reason="도메인 인프라 실증 또는 결합 워크로드 실측 근거 미확인")
