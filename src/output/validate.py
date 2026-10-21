"""R5 자동 검증.

LLM 채점이 아니라 결정적 검사를 수행한다.

검사 대상은 `schema.Assessment` 다 — claims / evidence / sources / gaps / status.

자동 검증은 구조와 출처 연결, 그리고 수치가 인용 원문에 있는지를 확인한다.
Claim 내용이 실제 근거에서 의미적으로 도출되는지는
human review worksheet에서 사람이 확인한다.
"""

import re
from functools import lru_cache

from src.state import ASSESSMENT_NODES

CITATION_RE = re.compile(r"\[([0-9a-f]{12})\]")

# 한국어 주장과 영어 원문의 단위 표기를 같은 종류로 묶는다. 영문 단위 뒤에 글자가 붙으면
# 단위가 아니다(8xlarge, 4bitwise). x·× 뒤에 숫자가 오면 크기·16진수다(1x1, 4×4, 0x1f).
UNITS = {
    "배": r"(?:배|[x×](?![\w×]|\.\d)|times(?![A-Za-z])|-?fold(?![A-Za-z]))",
    "%": r"(?:%|퍼센트|percent)",
    "비트": r"(?:비트|-?bits?)(?![A-Za-z])",
    "GB": r"GB(?![A-Za-z])",
    "TB": r"TB(?![A-Za-z])",
}
_NUMBER = r"(?<![\d.])(\d+(?:\.\d+)?)"

# 설계서 §2 — 이 수치는 실험 조건과 함께 쓸 때만 맞는 말이다. 같은 문단에 조건어가 하나도
# 없으면 다른 조건의 결과처럼 읽힌다. 조건어는 같은 뜻의 표기를 모두 둔다 — ITME 원문 초록은
# 35.7% 의 조건을 128GB 가 아니라 "beyond host memory limits" 로 적는다. 특정 표기만 요구하면
# 원문에 없는 수치를 쓰라는 요구가 된다. ponytail: 조건어 존재만 본다. 조건 문구의 의미는 사람이 본다.
# 규칙은 그 수치의 기술에만 건다 — InfiniGen 의 "8배 메모리 절감" 에 H100 을 요구하지 않는다.
# 기술이 both 인 Claim(market 등)은 어느 기술 이야기인지 모르므로 모든 규칙을 건다.
QUALIFIERS = {
    (8.0, "배"): ("TurboQuant", ("H100",)),
    (1.8, "배"): ("ITME", ("NVMe",)),
    (1.81, "배"): ("ITME", ("TTFT", "첫 토큰")),
    (35.7, "%"): ("ITME", ("128GB", "128 GB", "호스트 메모리", "host memory")),
}


def qualifiers_for(value: float, unit: str, technology: str) -> tuple[str, ...]:
    """이 기술의 이 수치에 필요한 조건어. 규칙이 없으면 빈 튜플."""
    rule = QUALIFIERS.get((value, unit))
    return rule[1] if rule and technology in (rule[0], "both") else ()


_UNIT_RES = {unit: re.compile(rf"{_NUMBER}(?:\s*[-–~]\s*(\d+(?:\.\d+)?))?\s*{pattern}", re.I)
              for unit, pattern in UNITS.items()}


@lru_cache(maxsize=4096)
def numbers(text: str) -> frozenset[tuple[float, str]]:
    """단위가 붙은 수치만 뽑는다. `2.5-3.5 bits` 같은 범위는 양 끝을 모두 넣는다.

    같은 인용 원문을 Claim·줄·문장마다 다시 대조하므로 결과를 원문별로 기억한다.
    """
    return frozenset((float(v), unit) for unit, rx in _UNIT_RES.items()
                     for m in rx.finditer(text) for v in m.groups() if v)


@lru_cache(maxsize=4096)
def bare_numbers(text: str) -> frozenset[float]:
    """단위 없이 홀로 선 수치. 논문 표(`KV Size 2.5`)·수식(`b = 2`)은 단위를 따로 적는다.

    `8B`(모델 크기)처럼 글자가 붙은 숫자와, `128 MB` 처럼 띄어 쓴 다른 단위가 붙은 숫자는
    뺀다 — 그림 축 눈금 `128 MB` 로 `128GB` 를 통과시킨 적이 있다.
    """
    return frozenset(float(v) for v in re.findall(
        r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w]|\.\d|\s*[A-Za-z%×])", text))


def describe(problems: list[dict]) -> str:
    """노드의 재작성 요청에 넣는 오류 목록. 빠진 조건어를 함께 알려 준다."""
    return "\n".join(
        f"- {p['value']} ({p['kind']}" + (f", 필요한 조건: {'/'.join(p['needs'])}" if p.get("needs") else "")
        + f"): {p['line'][:160]}" for p in problems)


# 인용 묶음 — `[a] , [b]` 처럼 이어 붙은 인용은 한 묶음이다.
_CITATION_RUN = re.compile(r"(?:\[[0-9a-f]{12}\][\s,]*)+")


def segments(line: str, quotes: dict[str, str]) -> list[tuple[str, list[str]]]:
    """줄을 인용 단위로 자른다 — (본문 조각, 그 조각이 인용한 ID).

    조각은 앞 인용 묶음 뒤부터 이번 인용 묶음까지다. 문장 끝마다 인용을 다는 작성 규칙
    (`… 이다 [id].`)에서는 문장 하나가 된다. 마지막 인용 뒤의 꼬리처럼 인용이 없는 조각은
    그 줄의 인용 전체(줄에도 없으면 Claim 근거 전체)와 대조한다.
    """
    line_ids = [i for i in CITATION_RE.findall(line) if i in quotes] or list(quotes)
    parts, start = [], 0
    for run in _CITATION_RUN.finditer(line):
        ids = [i for i in CITATION_RE.findall(run.group()) if i in quotes]
        parts.append((line[start:run.start()], ids or line_ids))
        start = run.end()
    if line[start:].strip():
        parts.append((line[start:], line_ids))
    return parts


def numeric_errors(text: str, quotes: dict[str, str], technology: str = "both") -> list[dict]:
    """주장의 수치가 인용 원문에 있는지, 조건이 필요한 수치에 조건어가 붙었는지 본다.

    수치는 **그 문장이 인용한 원문**과 대조한다(`segments`). 같은 문단의 다른 문장이 인용한
    원문에 우연히 같은 값이 있어도 통과시키지 않는다. 배수는 원문에도 배수 단위가 붙어 있어야
    한다 — 표의 아무 `8` 로 `8배` 를 통과시키지 않는다. 나머지 단위는 표처럼 홀로 선 숫자도
    인정한다. 조건어는 줄(문단) 단위로 본다 — 조건을 앞 문장에 쓰고 수치를 뒤 문장에 쓰는
    서술이 흔하다.
    """
    errors = []
    for line in text.splitlines():
        if line.strip().startswith("근거 공백"):      # Gap 목록이지 주장이 아니다
            continue
        whole = CITATION_RE.sub("", line)
        for part, ids in segments(line, quotes):
            source = set().union(*(numbers(quotes[i]) for i in ids))
            loose = set().union(*(bare_numbers(quotes[i]) for i in ids))
            for value, unit in sorted(numbers(part)):
                label = f"{value:g}{unit}"
                if (value, unit) not in source and (unit == "배" or value not in loose):
                    errors.append({"kind": "근거에 없는 수치", "value": label, "line": line.strip()})
                    continue
                needed = qualifiers_for(value, unit, technology)
                if needed and not any(q.lower() in whole.lower() for q in needed):
                    errors.append({"kind": "수치 조건 누락", "value": label,
                                   "needs": list(needed), "line": line.strip()})
    return errors


RAG_NODES = {
    "research": {"papers_core", "ecosystem", "context"},
    "maturity": {"papers_core"},
    "market": {"ecosystem"},
    "domain_assessment": {"papers_core", "context"},
}


def _structured_errors(node_name: str, assessment: dict, current_run_id: str | None,
                       numeric: bool = True) -> list[dict]:
    """신형 Claim → Evidence → Source 구조를 검사한다."""
    errors = []
    evidence_by_id = {e.get("evidence_id"): e for e in assessment.get("evidence") or []
                      if e.get("evidence_id")}
    source_by_id = {s.get("source_id"): s for s in assessment.get("sources") or []
                    if s.get("source_id")}
    # allowed_uses 는 역할 이름이다. State 키 domain_assessment 의 역할은 domain 이다.
    expected_use = "domain" if node_name == "domain_assessment" else node_name
    seen_claim_ids = set()

    for claim in assessment.get("claims") or []:
        claim_id = claim.get("claim_id")
        if not claim_id:
            errors.append({"node": node_name, "kind": "claim_id 누락", "ids": []})
            continue

        def error(kind, ids=()):
            errors.append({"node": node_name, "claim_id": claim_id, "kind": kind, "ids": list(ids)})

        if claim_id in seen_claim_ids:
            errors.append({"node": node_name, "kind": "claim_id 중복", "ids": [claim_id]})
        seen_claim_ids.add(claim_id)

        evidence_ids = claim.get("evidence_ids") or []
        if not evidence_ids:
            if claim.get("kind") == "fact":                       # 사실 Claim은 직접 근거 필수
                error("사실 Claim 근거 누락")
            elif claim.get("kind") in {"inference", "hypothesis"}:  # 추론/가설도 전제 근거 필수
                error("추론·가설 전제 근거 누락")

        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                error("Evidence 없음", [evidence_id])
                continue

            source_id = evidence.get("source_id")
            if not source_id or source_id not in source_by_id:
                error("Source 연결 실패", [evidence_id])
            if current_run_id and evidence.get("run_id") not in (None, "", current_run_id):
                error("다른 run의 Evidence", [evidence_id])
            source_run_id = source_by_id.get(source_id, {}).get("run_id")
            if current_run_id and source_run_id and source_run_id != current_run_id:
                error("다른 run의 Source", [source_id])

            # allowed_uses가 존재하면 노드 사용 권한 검사
            allowed_uses = evidence.get("allowed_uses")
            if allowed_uses is not None and expected_use not in allowed_uses:
                error("허용되지 않은 Evidence 사용", [evidence_id])

            # RAG 자료의 collection 범위 검사
            collection = evidence.get("collection")
            if node_name in RAG_NODES and collection and collection not in RAG_NODES[node_name]:
                error("허용되지 않은 collection", [evidence_id])

        # ID가 맞아도 수치는 지시문이나 다른 실험에서 올 수 있다. 인용 원문과 대조한다.
        quotes = {eid: evidence_by_id[eid].get("quote") or ""
                  for eid in evidence_ids if eid in evidence_by_id}
        if numeric and quotes:
            for problem in numeric_errors(claim.get("text") or "", quotes,
                                          claim.get("technology") or "both"):
                errors.append({"node": node_name, "claim_id": claim_id,
                               "ids": [problem["value"]], **problem})

    return errors


def check(state) -> dict:
    """현재 State의 자동 검증 결과를 State 필드 `validation` 으로 반환한다.

    설계서 §8 *형식 검사* — Claim에 근거 ID가 연결됐는지, ID가 이번 실행에서 확보한
    자료인지, 노드별 허용 자료인지 확인한다. 내용 검토(사람)는 `review.py` 가 맡는다.

    재시도 횟수는 세지 않는다. 부록 A: *재시도는 노드 내부의 최대 1회 처리다.*
    """
    current_run_id = state.get("run_id")
    # 수치 검사는 Evidence.quote 가 청크 전문인 실행에만 한다. 그 전 실행은 quote 가 300자로
    # 잘려 있어 맞는 수치도 "근거에 없는 수치" 가 된다 — `app.py --resume <옛 run_id>` 가 그렇다.
    numeric = bool((state.get("run_config") or {}).get("numeric_check"))

    errors = []
    for node_name in ASSESSMENT_NODES:
        assessment = state.get(node_name) or {}

        if not assessment:
            continue

        errors.extend(_structured_errors(node_name, assessment, current_run_id, numeric))

    # conflicts == [] 는 정상일 수 있으므로 오류로 처리하지 않는다.

    return {
        "validation": {
            "errors": errors,
            "checked_nodes": list(ASSESSMENT_NODES),
        },
        "trace": [{
            "node": "validate",
            "status": "failed" if errors else "ok",
            "errors": len(errors),
            "numeric_check": numeric,
        }],
    }