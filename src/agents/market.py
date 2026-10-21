"""시장성 평가 — 담당: R4 | ecosystem RAG (설계서 §5·§6)

시장 근거는 논문 본문에 없다. 그래서 근거를 ecosystem 컬렉션으로 색인해서 RAG 로 찾는다.
papers_core 를 조회하면 의미적으로 가까운 기술 서술이 시장 근거 자리에 놓이므로
`bind_document_search("market")` 로 컬렉션을 코드에서 고정한다(§3).

절이 기술별이 아니라 ①②③ 항목별이라 `MarketDraft` 가 세 절을 필드로 받는다. 생성·수치
재작성·인용 검증·Assessment 조립은 다른 평가 노드와 같은 `common.evaluate` 가 한다.
"""

from typing import ClassVar

from pydantic import BaseModel, Field

from src.agents.common import GAP_RULES, TECHS, DraftGap, domain_of, evaluate, unique
from src.schema import Assessment, Gap
from src.tools.docs import bind_document_search
from src.tools.web_store import utcnow

INSTRUCTION = """너는 시장성 평가 담당이다. 아래 세 절을 각 필드에 채운다.

size — ① 시장 규모/성장성
정량 리포트가 있으면 그대로 인용한다.

adoption — ② 상용화/채택 현황
**"발표됨" 과 "배포 확인됨" 을 반드시 구분한다.**
발표 자료만 있으면 "발표 단계, 배포 확인 미확인" 으로 적는다.

ecosystem — ③ 생태계 지지
지원 프레임워크, 표준화 동향.
**CXL 제품·PoC 근거와 ITME 채택 근거를 분리한다.** 삼성·SK하이닉스의 CXL 제품이
존재한다는 사실은 ITME 아키텍처가 채택되었다는 근거가 아니다. 해당 근거가 LLM 추론
워크로드 특화인지 개별 확인해 적고, 불명이면 "일반 CXL 근거이며 추론 특화 여부 미확인".

논문 성능 수치와 시장 매출을 섞지 않는다."""

QUERIES = [
    ("시장 규모 성장성 추론 메모리 비용", "both"),
    ("상용화 제품 출시 발표 실제 배포 채택", "both"),
    ("서빙 프레임워크 지원 현황 표준화 동향", "both"),
]
SECTIONS = [("①", "시장 규모/성장성", "size"), ("②", "상용화/채택 현황", "adoption"),
            ("③", "생태계 지지", "ecosystem")]


class MarketDraft(BaseModel):
    """`evaluate` 가 쓰는 것은 `rules` · `gaps` · `parts()` 셋뿐이다(`common.Draft` 와 같은 모양)."""

    rules: ClassVar[str] = GAP_RULES

    size: str = Field(default="", description="① 시장 규모/성장성. 사실 문장 끝에 [12자리hex] 인용")
    adoption: str = Field(default="", description="② 상용화/채택 현황. 사실 문장 끝에 [12자리hex] 인용")
    ecosystem: str = Field(default="", description="③ 생태계 지지. 사실 문장 끝에 [12자리hex] 인용")
    gaps: list[DraftGap] = Field(default_factory=list)

    def parts(self):
        # 보고서 §4.2 가 절 제목을 본문째 싣는다 — 텍스트 출력 시절과 같은 모양을 유지한다.
        return [(mark, label, "both", "fact",
                 f"## {mark} {label}\n{getattr(self, field).strip()}" if getattr(self, field).strip() else "")
                for mark, label, field in SECTIONS]


def market(state) -> dict:
    search = bind_document_search("market")      # ecosystem 고정 (§3)
    chunks = unique(c for query, technology in QUERIES
                    for c in search.invoke({"query": query, "technology": technology, "top_k": 5}))

    if not chunks:
        # §5 "자료가 없으면 미확인" — 근거 0건으로 시장성을 생성하지 않는다.
        assessment = Assessment(status="failed", claims=[], sources=[], evidence=[], gaps=[
            Gap(role="market", technology=technology, item="시장 근거 미확보",
                reason="ecosystem 컬렉션 검색 결과 0건") for technology in TECHS])
        return {"market": assessment.model_dump(),
                "trace": [{"node": "market", "status": "failed", "attempt": 1,
                           "timestamp": utcnow(), "chunks": 0}]}

    return evaluate(node="market", role="market", run_id=state.get("run_id", "default_run"),
                    domain=domain_of(state), instruction=INSTRUCTION, chunks=chunks,
                    schema=MarketDraft, covers=("direct", "ecosystem"),
                    gap_reason="ecosystem 색인 내 근거 미확인")
