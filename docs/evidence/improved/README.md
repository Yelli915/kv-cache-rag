# 기준선 → 개선 실물 실행 — 2026-10-21

개선 작업(구조화 출력·수치 검증) **전** 코드로 실제 LLM·색인·웹 검색을 끝까지 돌린 기준선과, 거기서
드러난 세 문제를 고친 뒤 같은 코퍼스·같은 모델로 다시 돌린 기록이다. R1·R3 스모크와 달리
mock·합성 데이터가 아니다. 실행 원본은 `../baseline/20261021-121122-d0a3d6/`와 이 폴더에 있다.

| 항목 | 기준선 `20261021-121122-d0a3d6` | 개선 후 `20261021-123919-5c7710` |
|---|---|---|
| 결과 | partial · 검토 대기 | partial · 검토 대기 |
| 검증 오류 | 0건 (수치 검사 없음) | 0건 (수치 검사 통과) |
| research Claim | **both 1건** (3,956자) | TurboQuant 1 · ITME 1 (854 · 956자) |
| domain Claim | both fact 1 · 가설 1 | TurboQuant 1 · ITME 1 · 가설 1 |
| 근거 없는 수치 (Claim 본문) | **4건** — 지시문 가드레일 누수 | 0건 |
| stakeholder ITME Claim | **0건** | 1건 (developers) |
| Claim 합계 | 11 | 13 |

모델 `google_genai` / `gemini-3.8-flash`(temperature 0, seed 20260922 실효값), 코퍼스
`data/manifest.json` 2026-10-21판(두 실행 동일 — PDF 6건은 9월과 해시 동일, 웹 4건은 원문이 바뀌어 재수집).
기준선은 5분 4초 · LLM 호출 7회(캐시 적중 0) · 토큰 89,684(입력 59,757 / 출력 29,927).

포함: `run.json` · `state.json` · `trace.jsonl` · `review.csv` · `web/events.jsonl`.
**제외:** `web/snapshots/`(제3자 원문 HTML, interface ③ — 운영 스냅샷은 Git에 넣지 않는다), `web/search/`,
그리고 `state.json`에 이미 든 사본 — `draft.json`, `web/manifest.json`(출처·근거는 `state.stakeholder`,
수집 실패·검색은 `web/events.jsonl`).

### 기준선 노드별 결과

| 노드 | status | Claim (기술·kind·근거 수) | Gap |
|---|---|---|---|
| research | completed | **both**·fact·7 — 본문 3,956자 한 덩어리 | 8 |
| maturity | completed | TurboQuant·fact·1 / ITME·fact·1 | 5 |
| market | completed | both·fact·3 / both·fact·5 | 7 |
| stakeholder | partial | TurboQuant 4건(fact 3·inference 1), **ITME 0건** | 5 |
| domain_assessment | completed | both·fact·5 / both·hypothesis·5 | 6 |
| synthesis | — | 일치 5 · 상충 0 · 결합 가설 1 | 44 |

기준선 research Claim에 지시문 가드레일이 근거 없이 그대로 새어 나온 예 — 8배의 출처(`turboquant-blog`)는
ecosystem 컬렉션이라 research(papers_core)가 볼 수 없는데도 `kind=fact`, 근거 7건으로 자동 검증을 통과했다.

> [가드레일 ② - 8배 가속]: 8배 가속 주장은 H100 GPU 환경의 어텐션 로짓 연산에 국한된
> 가속이며 … 제공된 청크에는 H100 환경 및 8배 가속 관련 구체적 수치 근거가 명시되지 않아 미확인입니다

**캐시 주의** — 이 실행의 research·maturity·domain·market 응답은 캐시 적중이다(`run.json`
`cache_hits: 4`). 프롬프트가 같은 직전 실행 `20261021-123030-55c6fb`가 생성했고, 그 실행에서
세 노드 모두 생성 직후 수치 문제 0건(`attempt: 1`)이었다. 그 실행은 아래 2-b 때문에 failed라
보관하지 않았다.

## 무엇을 고쳤나

**1. 기술별 절 → 구조화 출력** (`src/agents/common.py`)
research·maturity·domain이 자유 텍스트를 헤더 정규식으로 자르던 것을 `turboquant`·`itme`·
`gaps`(domain은 `hypothesis` 추가) 필드로 받는다. 세 노드에 세 벌 있던 인용 파싱·Source/Evidence
등록·status 판정·Gap 파싱을 `evaluate` 하나로 합쳤다. market은 기술별이 아니라 `## ①②③` 항목별
구조이고 기준선에서 정상 분리돼 그대로 뒀다.

**2. 수치 가드레일 → 코드** (`src/output/validate.py` `numeric_errors`)
- 단위 붙은 수치를 그 문장이 인용한 원문과 대조한다(`8배`↔`8x`, `비트`↔`-bit`).
- 설계서 §2 조건부 수치(8배·1.80배·1.81배·35.7%)는 같은 줄에 조건어가 있어야 한다.
- 평가 노드는 생성 직후 같은 검사로 1회 재작성, stakeholder는 기존 인용 수정 1회를 같이 쓴다.
  그래도 남으면 `review` 검증이 실행을 막는다.
- Evidence `quote`를 300자에서 청크 전문으로 바꿨다. 잘린 quote로는 대조할 원문이 없다.
- research 지시문의 가드레일은 "근거에 있을 때만 조건과 함께 쓴다, 규칙 자체를 옮겨 적지 않는다"로 바꿨다.

**2-b. 조건어는 같은 뜻의 표기를 모두 인정한다.** 처음에는 35.7%에 `128GB`만 요구했다. 개선 후
첫 실행(`20261021-123636-6e37c2`, failed)에서 stakeholder가 수정 기회를 쓰고도 실패했다. ITME 원문
초록은 이 조건을 128GB가 아니라 "beyond host memory limits"로 적는다. 128GB를 요구하면 원문에 없는
수치를 쓰라는 요구가 된다. `호스트 메모리`·`host memory`를 조건어에 넣었다. 이 실행의 trace에는
stakeholder가 보완 후 재작성에서 조건을 또 빠뜨렸다가 수정 1회로 고친 기록이 있다(`draft_validation`
failed 2 → ok 3).

**3. stakeholder ITME 검색 질의** (`src/agents/stakeholder.py` `CONTEXT`)
기준선 질의가 "ITME competing technology vendor response statement"라 도서관 안내·FTC 지침 같은 무관한
페이지만 돌아왔다. 질의에 `"Inference Tiered Memory Expansion" CXL KV cache LLM`을 붙였다. ITME
근거는 developers 1건을 확보했고 competitors·adopters·investors는 여전히 Gap이다. 2026-06 공개 논문이라
공개 반응 자체가 적은 것으로 보이지만, 단정하려면 질의를 바꿔 몇 번 더 봐야 한다.

## 절 분리 안정성 — `eval/both_fallback.py`

같은 검색 결과로 research·maturity·domain을 **응답 캐시 없이** 반복 실행해 개선 전(`891c6e6`,
헤더 정규식)과 개선 후(구조화 출력)를 비교했다. 원자료 `both_fallback_20261021.json`.

| | 노드 | both fallback | 기술별 분리 | 섞인 Claim | API 한도 초과 |
|---|---|---|---|---|---|
| 개선 전 | research | 0 / 10 | 10 / 10 | 10 | 0 |
| 개선 전 | maturity | 0 / 10 | 10 / 10 | 10 | 0 |
| 개선 전 | domain | 설계상 both (fact를 나누지 않음) | — | — | 0 |
| 개선 후 | research | 0 / 7 | 7 / 7 | 0 | 3 |
| 개선 후 | maturity | 0 / 3 | 3 / 3 | 0 | 7 |
| 개선 후 | domain | — | — | — | 10 |

**측정이 끝나지 않았다.** 도중에 Gemini 프로젝트 월 지출 한도에 걸려(429 `RESOURCE_EXHAUSTED`)
개선 후 30회 중 20회가 실패했다. 개선 전을 먼저 몰아 돌린 탓에 개선 후만 비었다 — 스크립트는
이제 회차마다 두 쪽을 번갈아 돌린다. 한도를 올린 뒤 `uv run python -m eval.both_fallback --runs 10`
으로 다시 잰다.

지금 자료로 말할 수 있는 것:

- **개선 전 코드도 20회 동안 fallback 0회였다.** 기준선 실행의 fallback은 드문 사건이다. 20회로는
  발생률을 추정할 수 없고, "fallback 횟수가 줄었다"를 개선 효과로 주장하지 않는다. 구조화 출력의
  이점은 빈도 감소가 아니라 **both fallback이 구조적으로 나올 수 없다**는 것이다.
- **개선 전 기술별 Claim은 20/20 섞였다** (TurboQuant Claim에 ITME, 또는 그 반대가 나온다).
  추정 원인: 정규식이 `2. ITME`부터 본문 끝까지를 ITME Claim으로 자르므로 뒤에 오는 요약 절(maturity
  지시문의 "3. … 요약")과 `근거 공백:` 줄이 ITME Claim에 붙는다. 이 측정은 Claim 본문을 저장하지
  않아 원인을 확인하지 못했다 — 스크립트는 이제 본문을 저장한다.
- 개선 후 완료된 10회는 모두 분리됐고 섞인 Claim이 없다. 표본이 작아 경향으로만 읽는다.

## 수치 검사 — `eval/numeric_guard_eval.py`

실행 기록의 Claim에서 근거로 뒷받침되는 수치를 골라 변조하고 검사기가 잡는지 센다. API 키 불필요.
기준선의 `--index` 수치는 PDF 추출 수정(`docs/evidence/retrieval/`) **전** 색인으로 잰 값이다. 수정 후에는
모든 chunk_id가 바뀌어 `--index`가 옛 기록의 청크를 찾지 못한다.
아래는 최종 검사기(인용 단위 대조) 기준이다.

| | 기준선 (`--index`) | 개선 후 |
|---|---|---|
| Claim / 단위 붙은 수치 | 11 / 22 | 13 / 16 |
| 수치가 있는 줄 / 원본 오류 | 13 / 4 | 11 / **1** |
| 변조 (수치 1.5배) | 16 / 16 | 12 / 12 |
| 조건 누락 | 표본 없음 | 1 / 1 |
| 섞기 (다른 Claim 근거의 같은 단위 수치) | 11 / 11 | 10 / 10 |
| 원문에 있는 값이라 못 만든 변조 | 11 | 6 |

**원본 오류는 오탐이 아니다.** 기준선 4건은 지시문 가드레일의 수치다. 개선 후 1건은 research ITME
Claim의 "(CPU 오프로드 대비, 256개 동시 대화, 호스트 메모리 **128GB** 소진 후 구간)"이다. 그 문장이
인용한 청크 `479f00cad7ea`에는 128GB가 없다 — 개선한 research 지시문의 조건 괄호를 모델이 그대로
옮겼다. 이 실행 당시의 검사기(줄 단위)는 놓쳤다. 같은 문단이 인용한 다른 청크의 그림 축 눈금
`128 MB`의 128을 단위 없는 숫자로 인정했기 때문이다. 이후 두 가지를 고쳤고 각각 단독으로도 이 건을 잡는다.

- 수치는 **그 문장이 인용한 원문**과 대조한다(`validate.segments` — 인용 묶음 기준으로 줄을 자른다).
- 단위 없는 숫자는 뒤에 다른 단위가 붙지 않은 것만 인정한다(`128 MB` ≠ `128GB`).
- research 지시문의 조건에서 실험 설정값(256개, 128GB, 128개, 5턴)을 뺐다. 조건의 뜻만 남긴다.

이 수정 뒤의 실물 실행은 아직 없다 — API 월 한도에 걸려 있다. 따라서 이 폴더의 실행 기록은
지금 검사기로는 research Claim 1건이 검증에서 걸리는 상태다.

**사람 검토 부담** — 설계서 §8에 따라 모든 Claim은 여전히 사람이 검토하므로 검토할 Claim 수는
줄지 않는다. 줄어드는 것은 두 가지다. ID만 보던 이전 검증이었다면 두 실행 모두 **수치 오류가 든
Claim 1건씩이 사람에게 그대로 넘어갔다** — 이제 생성 단계 재작성 또는 검증에서 막힌다. 그리고
단위 붙은 수치(22개, 16개)가 인용 원문에 있는지는 코드가 확인하므로, 사람은 그 수치가 *어느 실험의
값인지*만 본다.

**맹점** — 못 만든 변조 17건은 **모두 비트 수(2·3·4)**다. TurboQuant 원문은 표(`KIVI 3`)와
수식(`b = 2`)에 작은 정수를 단위 없이 쓴다. 그래서 "2비트"를 "3비트"로 바꾸면 원문의 다른 3과
맞아 원문 대조로는 구분할 수 없다. 단위 없는 숫자를 인정하지 않으면 원문이 `b = 2`로만 적은
"2비트" 같은 정상 주장이 오탐이 된다. 이 경우는 사람 검토가 맡는다.

- 만든 변조는 모두 잡았다. 정의상 "원문에 없는 값"으로 변조하므로 높게 나오는 게 당연하다.
  못 만든 것까지 세면 67건 중 50건(75%).
- 표본이 작다(실행 2회). 경향으로만 읽는다.

## 재개

두 폴더는 검토 대기 정지 시점 기록이다. 검토를 마치려면 `runs/<run_id>/review.csv`를 채우고
`uv run python app.py --resume <run_id>`.
