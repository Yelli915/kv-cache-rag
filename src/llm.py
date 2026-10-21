"""LLM 팩토리 — 담당: R1

모델은 config/settings.yaml 기본값을 쓰되 .env 의 LLM_PROVIDER · LLM_MODEL 이 우선한다.

**재현성은 모델이 아니라 캐시가 담당한다.** `gpt-5.6-luna` 같은 추론형 모델은
langchain 이 temperature 를 아예 보내지 않고 seed 도 대개 무시한다. 실측에서 인용 근거가
완전히 같은데도 본문이 74% 달랐다. 그래서 프롬프트 전문을 키로 응답을 캐시한다 —
같은 코퍼스·같은 지시문이면 같은 응답이 나온다. 모델의 결정성에 기대지 않는다.

캐시 적중은 "이번 실행에서 모델이 새로 판단하지 않았다" 는 뜻이라 기록해서 보고서에
밝힌다. 설계서 §6 — 실제 비용은 토큰 사용량과 실행 기록으로 확인한다.
"""

import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.globals import set_llm_cache

from src.settings import settings

CACHE_PATH = ".cache/llm.sqlite"

_cache = None
_effective: dict = {}


class _CountingCache:
    """적중·미적중을 세는 SQLite 캐시. langchain 캐시 인터페이스는 lookup/update 둘뿐이다."""

    def __init__(self, database_path: str):
        from langchain_community.cache import SQLiteCache

        self._inner = SQLiteCache(database_path=database_path)
        self.hits = 0
        self.misses = 0

    def lookup(self, prompt: str, llm_string: str):
        found = self._inner.lookup(prompt, llm_string)
        if found is None:
            self.misses += 1
            return None
        self.hits += 1
        # 저장된 응답은 원래 호출의 토큰 수를 들고 있다. 그대로 두면 사용량 콜백이
        # 재사용분을 이번 실행의 비용으로 센다 — 적중은 호출이 없었으므로 0 이다.
        for gen in found:
            message = getattr(gen, "message", None)
            if getattr(message, "usage_metadata", None):
                gen.message = message.model_copy(update={"usage_metadata": {
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})
        return found

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        self._inner.update(prompt, llm_string, return_val)


def enable_cache(path: str = CACHE_PATH):
    """프롬프트 전문을 키로 응답을 재사용한다. 프롬프트가 바뀌면 자동으로 새로 생성한다."""
    global _cache
    if _cache is None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _cache = _CountingCache(path)
        set_llm_cache(_cache)
    return _cache


def model_name() -> str:
    return os.getenv("LLM_MODEL") or settings()["llm"]["model"]


def provider() -> str:
    return os.getenv("LLM_PROVIDER") or settings()["llm"]["provider"]


def get_llm():
    """설정값이 아니라 **실제로 적용된** 파라미터를 기록해 둔다.

    langchain 은 모델에 따라 temperature 를 조용히 버린다. 설정 파일의 값을 그대로
    run.json 에 적으면 거짓말이 된다(§6 — 온도 등 지원되는 설정도 저장한다).
    """
    cfg = settings()["llm"]
    extra = {"seed": cfg["seed"]} if cfg.get("seed") is not None else {}
    llm = init_chat_model(
        model_name(),
        model_provider=provider(),
        temperature=cfg["temperature"],
        **extra,
    )
    _effective.update(
        provider=provider(),
        model=model_name(),
        temperature=getattr(llm, "temperature", None),
        seed=getattr(llm, "seed", None),
    )
    return llm


def llm_report() -> dict:
    """실효 파라미터와 캐시 적중. LLM 을 한 번도 부르지 않았으면 빈 dict 다."""
    report = dict(_effective)
    if _cache is not None and (_cache.hits or _cache.misses):
        report |= {"cache_hits": _cache.hits, "cache_misses": _cache.misses}
    return report
