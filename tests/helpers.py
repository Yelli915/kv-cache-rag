"""테스트 공용 합성 데이터와 헬퍼. 값은 계약 형식만 맞춘 합성 데이터다."""

import csv
from types import SimpleNamespace

TQ_ID = "aaaaaaaaaaaa"
ITME_ID = "bbbbbbbbbbbb"

# 노드별 허용 컬렉션이 다르다(§3). 시장 근거를 papers_core 에서 주면 검증이 막는다.
PAPERS = [
    {"chunk_id": TQ_ID, "source_id": "turboquant-paper", "collection": "papers_core",
     "page": 3, "text": "TurboQuant reports 3.5-bit KV cache quantization.",
     "applies_to": ["TurboQuant"], "scope": "direct", "cosine_score": 0.7},
    {"chunk_id": ITME_ID, "source_id": "itme-paper", "collection": "papers_core",
     "page": 7, "text": "ITME reports 1.80x throughput over an NVMe-oF baseline.",
     "applies_to": ["ITME"], "scope": "direct", "cosine_score": 0.6},
]


def search(chunks):
    """질의와 무관하게 고정 청크를 돌려주는 가짜 바인딩 검색 도구."""
    return SimpleNamespace(invoke=lambda payload: list(chunks))


def assessment(role, text=None, *, run_id="r5-test"):
    """Claim·Evidence·Source 가 닫힌 completed Assessment 하나."""
    return {
        "status": "completed",
        "claims": [{"claim_id": f"claim-{role}", "text": text or f"{role} 주장",
                    "technology": "ITME", "kind": "fact", "evidence_ids": [f"e-{role}"]}],
        "evidence": [{"evidence_id": f"e-{role}", "source_id": f"s-{role}", "run_id": run_id,
                      "collection": "ecosystem", "quote": "합성 인용", "location": "p.1",
                      "allowed_uses": [role]}],
        "sources": [{"source_id": f"s-{role}", "run_id": run_id, "collection": "ecosystem",
                     "allowed_uses": [role], "title": f"{role} 출처",
                     "url": f"https://example.org/{role}", "published_at": "2026-01-01",
                     "retrieved_at": "2026-09-22"}],
        "gaps": [],
    }


class FakeLLM:
    """구조화 출력 LLM 자리. 받은 스키마와 마지막 프롬프트를 붙잡아 둔다.

    result 가 없으면 스키마의 빈 인스턴스를 돌려준다(필수 필드 검증 없이).
    """

    def __init__(self, result=None):
        self.result, self.schema, self.prompt, self.prompts = result, None, None, []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        self.prompts.append(prompt)
        return self.result if self.result is not None else self.schema.model_construct()


def fill_review(path, verdict, reviewer="권예리"):
    """review.csv 의 판정 칸을 채운다. verdict 는 문자열 또는 row -> 문자열."""
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    for row in rows:
        row["review_result"] = verdict(row) if callable(verdict) else verdict
        row["reviewer"] = reviewer
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
