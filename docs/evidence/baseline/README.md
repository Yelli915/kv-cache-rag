# 기준선 실물 실행 — 2026-10-21

개선 작업(구조화 출력·수치 검증) **전** 코드로 실제 LLM·색인·웹 검색을 끝까지 돌린 기록이다.
`docs/evidence/r1`·`r3`와 달리 mock·합성 데이터가 아니다.

| 항목 | 값 |
|---|---|
| run_id | `20261021-121122-d0a3d6` |
| 모델 | `google_genai` / `gemini-3.8-flash`, temperature 0, seed 20260922 (실효값) |
| 코퍼스 | `data/manifest.json` 2026-10-21판 — PDF 6건은 9월과 해시 동일, 웹 4건은 원문이 바뀌어 재수집 |
| 결과 | `run_status=partial`, `review_status=pending` (검토 대기 — 정상 정지) |
| 소요 | 5분 4초 · LLM 호출 7회(캐시 적중 0) · 토큰 89,684 (입력 59,757 / 출력 29,927) |
| 검증 오류 | 0건 |

포함: `run.json` · `state.json` · `trace.jsonl` · `review.csv` · `stakeholder*.json[l]` ·
`web/manifest.json` · `web/events.jsonl`. **제외:** `web/snapshots/`(제3자 원문 HTML,
interface ③ — 운영 스냅샷은 Git에 넣지 않는다), `web/search/`, `draft.json`(`state.json`과 동일).

## 노드별 결과

| 노드 | status | Claim (기술·kind·근거 수) | Gap |
|---|---|---|---|
| research | completed | **both**·fact·7 — 본문 3,956자 한 덩어리 | 8 |
| maturity | completed | TurboQuant·fact·1 / ITME·fact·1 | 5 |
| market | completed | both·fact·3 / both·fact·5 | 7 |
| stakeholder | partial | TurboQuant 4건(fact 3·inference 1), **ITME 0건** | 5 |
| domain_assessment | completed | both·fact·5 / both·hypothesis·5 | 6 |
| synthesis | — | 일치 5 · 상충 0 · 결합 가설 1 | 44 |

## 관찰 — 개선 작업의 출발점

**1. research 기술별 절 분리 실패 → `both` fallback.** 모델이 `### 1. 개요` 절을 먼저 두고
`### 2. TurboQuant 심층 분석` · `### 3. ITME 심층 분석`으로 번호를 붙였다.
`_split_technology_sections`의 정규식은 `1. TurboQuant` / `2. ITME`만 찾으므로 두 기술이
Claim 하나로 뭉쳤다. 사람이 검토할 때 어느 문장이 어느 근거에 기대는지 나눠 볼 수 없다.

**2. 프롬프트 가드레일의 수치가 Claim 본문으로 새어 나왔다.** research Claim에 다음 문장이 있다.

> [가드레일 ② - 8배 가속]: 8배 가속 주장은 H100 GPU 환경의 어텐션 로짓 연산에 국한된
> 가속이며 … 제공된 청크에는 H100 환경 및 8배 가속 관련 구체적 수치 근거가 명시되지 않아 미확인입니다

1.80배·1.81배·35.7%도 같은 형태로 들어갔다. 이 수치들은 **인용 청크가 아니라 지시문에서 왔다.**
8배의 출처(`turboquant-blog`)는 ecosystem 컬렉션이라 research(papers_core)가 볼 수 없다.
그런데도 Claim은 `kind=fact`, 근거 7건으로 자동 검증을 통과했다(검증은 ID 존재·범위만 본다).

**3. stakeholder ITME 근거 0건.** 웹 검색으로 ITME의 경쟁·도입·개발자·투자 반응을 하나도
확보하지 못해 네 주체가 모두 Gap이다. 2026-06 공개 논문이라 공개 반응이 아직 없는 것과
검색 질의 문제를 이 기록만으로는 구분할 수 없다.

## 재개

검토를 마치려면 `runs/20261021-121122-d0a3d6/review.csv`를 채우고
`uv run python app.py --resume 20261021-121122-d0a3d6`. 이 폴더는 정지 시점 기록이다.
