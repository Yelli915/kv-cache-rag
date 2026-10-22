"""BM25용 키워드 변환 — 담당: R2

BM25 는 어휘 매칭이라 한국어 질의로 영어 논문을 찾지 못한다.
**dense 검색은 변환하지 않은 원문 질의를 쓴다** — e5-small 이 cross-lingual 이라
한국어 질의를 그대로 넣는 게 이 모델을 쓰는 이유다.

LLM 번역을 쓰지 않는다: 검색마다 지연·비용이 붙고 실행 간 결과가 흔들린다.
"""

GLOSSARY = {
    "KV 캐시": "KV cache", "캐시": "cache", "양자화": "quantization",
    "압축": "compression", "처리량": "throughput", "지연": "latency",
    "정확도": "accuracy", "메모리": "memory", "대역폭": "bandwidth",
    "배치": "batch", "동시성": "concurrency", "추론": "inference",
    "오프로딩": "offloading", "계층": "tiered hierarchy", "확장": "expansion",
    "비용": "cost", "오버헤드": "overhead", "벤치마크": "benchmark",
    "실험": "experiment evaluation", "하드웨어": "hardware", "구현": "implementation",
    "공개": "open source release", "배포": "deployment", "표준": "standard",
    "채택": "adoption", "손실": "loss degradation", "전송": "transfer",
    "용량": "capacity", "성숙도": "maturity readiness", "한계": "limitation",
}


def to_keywords(query: str) -> str:
    """한국어 질의를 BM25 가 매칭할 수 있는 영어 키워드 열로 바꾼다."""
    out = query
    for ko, en in sorted(GLOSSARY.items(), key=lambda x: -len(x[0])):
        out = out.replace(ko, en)
    return out
