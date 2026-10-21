# 팀 인터페이스 계약

담당자끼리 **먼저 고정해야 하는 것만** 모았다. 여기 정의된 형식은 소유자 승인 없이 바꾸지 않는다.
구현 세부는 각 파일의 docstring, 이 계약이 왜 이렇게 됐는지는 [history.md](history.md).

**기준은 설계서 PDF다** — `RAG-Design_판교-8반_권수진 권예리 김민 박인기 정승원.pdf`
(§3~§9·부록 A). 코드끼리 어긋나면 PDF를 따르고, PDF가 정하지 않은 것만 이 문서가 정한다.

| 계약 | 소유 | 소비 |
|---|---|---|
| ① State (§7 14행 / 17필드) + `src/schema.py` | R1 | 전원 |
| ② 문서 검색 도구 + chunk 형식 | R2 · R3 | R4 |
| ③ 웹 근거 도구 + 실행별 저장소 | R3 | stakeholder · R5 |
| ④ `validation` · `review_status` | R5 | R1 (라우팅) |
| ⑤ 코퍼스 — `sources.json` 분류 값과 매니페스트 | R2 (선별 R3) | R3 · R4 |

---

## ① State + 공용 schema — `src/state.py` · `src/schema.py` (R1)

§7 표는 **14행**이고 그중 세 행이 묶음 표기(`run_id / run_config`, `validation / review_status`,
`report / report_paths`)라 Python 필드로는 **17개**다.

`TypedDict(total=False)`. **병렬 구간에서는 각 평가 노드가 자기 결과 키만 쓴다.** 웹 자료는
전역에 쌓지 않고 각 Assessment에 넣는다. 누적 reducer는 `trace` 하나뿐이다.

| §7 행 | 필드 | 형식 | 쓰기 주체와 용도 |
|---|---|---|---|
| 1 | `run_id` | `str` | 초기화. 실행 식별자 |
| 1 | `run_config` | `dict` | 초기화(`app.py`) + `setup`이 자료 버전을 채운다 |
| 2 | `research` | Assessment | `research` |
| 3 | `maturity` | Assessment | `maturity` |
| 4 | `market` | Assessment | `market` |
| 5 | `stakeholder` | Assessment | `stakeholder` |
| 6 | `domain_assessment` | Assessment | `domain_assessment` |
| 7 | `source_registry` | `dict[str, Source]` | **`collect_evidence`만** |
| 8 | `evidence_registry` | `dict[str, Evidence]` | **`collect_evidence`만** |
| 9 | `gaps` | `list[Gap]` | `collect_evidence` 합류 후 수집 → `synthesis` 종합 후 순차 병합 |
| 10 | `synthesis` | Synthesis | `synthesis` |
| 11 | `validation` | `dict` | `review`. 오류·내용 검토 결과 |
| 11 | `review_status` | `str` | `review`. `pending`이면 검토용 초안만 저장 |
| 12 | `run_status` | `str` | `final_check` · `app.py` |
| 13 | `report` | `str` | `report`. Markdown 본문 |
| 13 | `report_paths` | `list[str]` | `publish`. Markdown·PDF 저장 경로 |
| 14 | `trace` | `Annotated[list[dict], operator.add]` | 전 노드 (누적) |

```python
run_config = {"domain": "데이터센터/클라우드", "technologies": ["TurboQuant", "ITME"],
              "model": {...}, "limits": {...}, "runs_dir": "runs",
              "web_top_k": 3, "web_context_chars": 40000, "web": {...},
              "sources": [...]}      # setup 이 적재한 자료 버전(매니페스트)
```

- **단일 쓰기 주체.** 병렬 노드는 공용 `gaps`나 레지스트리를 직접 수정하지 않는다. 자기
  Assessment 안에만 Source·Evidence·Gap을 담고, 병합은 `collect_evidence`가 한다.
  같은 ID에 다른 내용이 오면 병합 오류(`MergeConflict`)로 실행을 끝낸다. 단 `allowed_uses`는
  원문 속성이 아니라 사용 권한이라 합집합으로 합친다.
- **재시도 횟수는 State 필드가 아니다.** 부록 A: *재시도는 노드 내부의 최대 1회 처리다.*

### 공용 모델 — `src/schema.py`

`Claim` / `Assessment` / `Source` / `Evidence` / `Gap` / `Synthesis` / `Conflict` / `Event`.

- **검증은 dict 경계에서 한 번만.** `collect_evidence`가 `Assessment.model_validate`를
  부르고, 그 뒤로는 State를 dict로 흐른다.
- `Assessment` = `claims` / `sources` / `evidence` / `gaps` / `status`(§6). claim → evidence →
  source가 자기 안에서 닫히는지, `status=failed`인데 claims가 남았는지를 본다.
- `Source`·`Evidence`는 `extra="allow"` — 웹 출처와 색인 출처의 필드가 다르다.
  필수는 `source_id` · `run_id` · `collection` · `allowed_uses` 넷.
- `Claim`·`Gap`·`Assessment`는 `extra="forbid"`. 오타난 필드가 조용히 통과하면 안 된다.
- `Claim.evidence_ids`는 최소 1개. 근거 없는 항목은 Claim이 아니라 Gap이다(§6).
- `Claim`의 `stakeholder_group`·`actor`·`context`는 선택이다. TRL·시장성 Claim에는 발언
  주체가 없다. 이해관계자 Claim의 엄격한 검사는 `stakeholder_contract.ClaimDraft`가 맡는다.
- `Gap.item`은 자유 문자열이다("TRL 6 실증 환경 근거" 식이라 Literal로 묶을 수 없다).
- `Role` = `research` `maturity` `market` `stakeholder` `domain` `synthesis`. State 필드명은
  `domain_assessment`지만 역할 이름은 `perspectives`와 맞춰 `domain`이다.
- `Synthesis.conflicts`는 비어 있을 수 있다 — §5 *상충하는 의견을 억지로 만들지는 않는다.*

### trace 한 줄 — `schema.Event`

```python
{"node": "maturity", "status": "ok", "attempt": 1, "timestamp": "...", "chunks": 14}
{"tool": "search_market_signals", "node": "stakeholder", "status": "ok", ...}
```

`node` 또는 `tool` 중 하나는 반드시 넣는다. 나머지는 자유(`extra="allow"`).

---

## ② 문서 검색 — `src/tools/docs.py` · `src/rag/retrieve.py` (R2 · R3)

**Dense FAISS 코사인 단일 모드다.** BM25·RRF는 제거했다(`retrieve.py`가 `mode != "dense"`를
거부). 검색 품질 측정으로 근거가 생긴 뒤 별도로 도입한다. 2026-10-21 실측(28문항)에서 RRF가 Hit@5
+2문항으로 경향만 보여 보류했다 — `docs/evidence/retrieval/`. 비교 구현은 `eval/retrieval_metrics.py`
안에만 있다.

### 평가 노드는 바인딩 도구만 쓴다

```python
from src.tools.docs import bind_document_search
research_search = bind_document_search("research")
domain_context_search = bind_document_search("domain", "context")
hits = research_search.invoke({"query": "...", "technology": "ITME", "top_k": 5})
```

바인딩 도구의 입력은 `query` · `technology` · `top_k`뿐이다. **컬렉션과 관점은 Python이
고정한다** — 모델이 생성한 인자로 고를 수 없다.

| 역할 | 컬렉션 | 허용 scope |
|---|---|---|
| research | papers_core | direct |
| maturity | papers_core | direct · comparison |
| market | ecosystem | direct · ecosystem |
| domain | papers_core · context | direct · comparison · secondary |

- 필터는 랭킹 **뒤**에 건다. 컬렉션 전체를 검색한 뒤 기술·관점으로 걸러 top_k로 자른다.
  먼저 자르면 비대칭 코퍼스에서 한 기술이 통째로 사라진다.
- 반환 뒤에도 범위를 다시 검사하고(`validate_hits`), 잘못된 청크가 섞이면 호출 전체가 실패한다.
- 빈 질의, 미지원 컬렉션·기술, `top_k` 1..100 밖은 오류다.
- `search_source_documents`(컬렉션을 직접 받는 원본 도구)는 신뢰하는 Python 코드 전용이다.
  LLM에 직접 연결하지 않는다.

### chunk 형식

```python
{
  "chunk_id": "a1b2c3d4e5f6",      # 12자 hex. 모델이 [a1b2c3d4e5f6] 로 인용한다
  "source_id": "itme-paper",       # sources.json 의 id. "source" 는 호환 별칭
  "collection": "papers_core",
  "applies_to": ["ITME"],          # 목록이다. 한 문서가 두 기술에 걸릴 수 있다
  "perspectives": ["research", "maturity", "domain"],
  "scope": "direct",
  "page": 7,                       # PDF 실제 페이지 (웹 문서는 1)
  "text": "...",
  "cosine_score": 0.61,            # 정규화 벡터의 코사인. RRF 점수가 아니다
}
```

인덱스가 없어도 이 형식만 지키면 노드를 짤 수 있다(`tests/mock_nodes.py`).

---

## ③ 웹 근거 — `src/tools/web_store.py` · `web_search.py` · `web_fetch.py` (R3)

**이해관계자 전용.** 색인하지 않으므로 200쪽 가드 대상이 아니다.

`WebEvidenceStore(run_id, root="runs", ...)`가 실행별 수집·예산·저장을 소유하고,
`bind_web_tools(store)`가 그 store에 묶인 두 도구를 반환한다. LLM 입력에 run_id/root를 노출하지 않으며
**다른 실행의 캐시로 대체하지 않는다.**

- `search_market_signals(query, technology="both", top_k=5)` — URL·제목·게시일 후보·snippet·
  조회일·source_type. `status="candidate"`. **검색 결과는 사실 근거가 아니다.**
- `fetch_web_evidence(url)` — 본문 수집에 성공한 자료만 Source/Evidence가 된다. 실패는
  `status="failed"`와 error를 반환한다. 날짜 미확인은 null이며 조회일로 대체하지 않는다.
- ID: `source_id = "web-" + sha256(final_url|raw_hash)[:12]`,
  `evidence_id = "e-web-" + sha256(source_id|location|quote)[:12]`. 모델이 옮겨 적는 값이라
  12자로 줄였다. 원문 해시 전체는 `sha256` 필드에 남는다. 같은 ID에 다른 내용이 등록되면 즉시 실패한다.
- 본문 추출은 렌더 실패·쿠키 안내 같은 상용구 문단을 버린다. 그 뒤 `min_body_chars` 미달이면 실패.
- 인용 검증 실패는 거부된 ID마다 이유를 돌려준다(오타·다른 기술 자료·입력에서 제외된 근거·없는 ID).
- 정적 HTML/text/PDF만 지원한다. JS 렌더링·로그인·유료벽은 우회하지 않는다.
- 저장: `runs/<run_id>/web/` 아래 `manifest.json` · `events.jsonl` · `search/*.json` · `snapshots/*`.
  운영 스냅샷은 Git에 넣지 않는다. 테스트 fixture는 합성 데이터 실행(`scripts/r3_smoke.py` → `tests/fixtures/r3/`)으로 만든다.
- 한 run_id에는 한 프로세스만 붙는다. 여러 프로세스의 동시 쓰기 잠금은 제공하지 않는다.

### 이해관계자 노드 동작

- 두 기술 × 네 주체(경쟁·도입·개발자·투자). Claim마다 kind·발언 주체·시점·맥락·evidence_ids.
- 근거를 못 찾으면 Gap을 남기고 `partial`. 검색 제공자가 전부 실패하면 `failed`.
- 본문이 없거나 초안이 근거 공백이라 판단한 기술×주체 항목은 **한 번만** 보완 검색한다.
- LLM 호출 상한 3회 — 초안 1회, 보완 후 재작성 최대 1회, 형식·인용·수치 수정 최대 1회.
- 수정 후에도 **무효 인용**이 남으면 claims를 비우고 `failed` — 근거 연결이 깨졌으므로 전체를 막는다.
- 수정 후에도 **수치 오류**(인용은 맞고 문장의 수치가 원문과 다르거나 조건이 빠짐)만 남은 Claim은
  그것만 빼고 그 기술×주체를 Gap(사유: 수치 근거 불일치)으로 남긴다. 나머지 Claim은 유지하고 `partial`.
- `completed`는 구조·출처 연결 검사 완료일 뿐, 내용 검토 완료가 아니다.

### 기본 한도 — `config/settings.yaml` 의 `run.web`

| 설정 | 기본값 | 의미 |
|---|---:|---|
| `web_top_k` | 3 | 기술×주체별 검색 후보 수 |
| `web_context_chars` | 40,000 | 모델에 전달할 근거 총 문자 수. 초과분은 레코드 단위로 제외하고 인용 금지 |
| `max_sources` | 24 | 성공한 원문 수 |
| `max_fetches` | 40 | 본문 수집 시도 수, 실패도 집계 |
| `max_searches` | 16 | 최초 8회 + 보완 최대 8회 |
| `max_body_chars` | 50,000 | 원문 한 건의 추출 본문 길이 |
| `max_total_chars` | 300,000 | 실행 전체 본문 문자 수 |
| `max_response_bytes` | 2,000,000 | HTTP 응답 하나의 바이트 상한 |
| `min_body_chars` | 120 | 빈 문서·짧은 오류 페이지 차단 |
| `timeout` | 15초 | HTTP 연결·read timeout |

예산은 manifest에 저장하며 같은 run_id를 재개할 때 바꾸지 않는다.

---

## ④ `validation` · `review_status` — 검증과 내용 검토 (R5 → R1)

R5가 쓰고 `final_check`가 읽는다. 설계서 §8: *형식 검사와 내용 검토*.

```python
validation = {
  "errors": [{"node": "maturity", "claim_id": "claim-...", "kind": "Evidence 없음", "ids": [...]}],
  "checked_nodes": [...],
  "reviewer": "권예리",
  "claim_verdicts": {"claim-...": "확인"},
}
review_status = "pending" | "passed"
```

**프로그램이 보는 것**(`src/output/validate.py`) — Claim에 근거 ID가 연결됐는지, ID가 **이번
실행에서 확보한** 자료인지, 노드별 허용 자료(`allowed_uses`·컬렉션)인지, 그리고 **본문의 수치가
인용 원문에 있는지**. 결정적 검사이며 LLM 채점이 아니다.

수치 검사(`numeric_errors`) — 단위가 붙은 수치(배·%·비트·GB·TB)를 **그 문장이 인용한 원문**과
대조한다. 문장은 인용 묶음(`… 이다 [id], [id].`)을 기준으로 자르고, 인용이 없는 꼬리는 그 줄의
인용 전체와 대조한다. 한국어 단위와 영어 표기(`8배`↔`8x`·`×`·`times`, `비트`↔`-bit`)는 같은 것으로 본다.

- `근거에 없는 수치` — 원문에 없다. 배수는 원문에도 배수 단위가 붙어 있어야 하고, 나머지는
  표처럼 홀로 선 숫자도 인정한다(표의 아무 `8`로 `8배`를 통과시키지 않는다). 단 `128 MB`처럼
  다른 단위가 붙은 숫자는 인정하지 않는다.
- `수치 조건 누락` — 설계서 §2의 조건부 수치(TurboQuant 8배→H100, ITME 1.80배→NVMe-oF,
  ITME 1.81배→TTFT·첫 토큰, ITME 35.7%→128GB·호스트 메모리)가 같은 줄에 조건어 없이 쓰였다.
  규칙은 **그 수치의 기술 Claim에만** 건다 — 다른 기술의 정상적인 "8배"에 H100을 요구하지 않는다.
  기술이 `both`인 Claim(market 등)은 어느 기술 이야기인지 모르므로 모든 규칙을 건다. 조건어는 같은 뜻의 표기를 모두
  인정한다. 원문마다 조건을 다르게 적기 때문이다(ITME 초록은 128GB 대신 "beyond host memory limits").
- `수치 조건 누락`의 조건어는 줄(문단) 단위로 본다 — 조건을 앞 문장에 쓰고 수치를 뒤 문장에
  쓰는 서술이 흔하다.
- 한계: 원문에 단위 없이 흔한 작은 정수(비트 수 2·3·4 등)는 다른 값으로 바뀌어도 원문의 다른
  숫자와 맞아 통과할 수 있다. 수치의 *의미*(어느 실험의 값인지)는 여전히 사람이 본다.

평가 노드(research·maturity·domain·market)는 같은 검사를 생성 직후에도 돌려, 걸리면 문제 문장을
짚어 **한 번** 다시 쓰게 한다. 그래도 남으면 이 검증이 실행을 막는다 — 이 노드들의 Claim은 기술 절
전체라 빼면 그 기술 평가가 통째로 사라지기 때문이다. 구조화 출력이 실패해도 같은 1회 기회를 쓰고,
두 번 다 실패하면 `failed` Assessment를 낸다. stakeholder는 위 ③대로 문제 Claim만 Gap으로 바꾼다.

수치 검사는 `run_config.numeric_check`가 있는 실행에만 한다. 그 전 실행은 quote가 300자로 잘려 있어
`--resume`하면 맞는 수치도 오류가 된다.

Evidence의 `quote`는 **청크 전문**이다. 잘라 두면 수치 검증과 사람의 검토가 원문 일부만 본다.

**사람이 보는 것**(`src/output/review.py`) — 원문 구절이 주장을 뒷받침하는지. 성능 수치의
단위·비교 기준선·실험 조건, TRL 단계의 근거, 직접 채택과 인접 생태계 자료의 구분.
**ID 대조만으로 자동 통과시키지 않는다.**

### `final_check` 판정 (부록 A)

| 조건 | run_status | 다음 |
|---|---|---|
| Assessment 중 `failed`, 또는 `validation.errors` 잔존 | `failed` | `fail` — 오류·로그 저장, 제출용 출력 차단 |
| `review_status == "pending"` | `partial` | `save_draft` — 검토용 초안만 저장, 실행 정지 |
| Assessment 중 `partial` | `partial` | `report` — 보고서는 낸다 |
| 그 외 | `completed` | `report` → `publish` |

failed를 검토 대기보다 먼저 본다. 되돌아오는 엣지는 `save_draft → review` 하나뿐이고,
검토 대기에서는 `interrupt_after=["save_draft"]`로 실행을 멈춘 뒤 `app.py --resume`으로 재개한다.

---

## ⑤ 코퍼스 — `sources.json` · `data/manifest.json` (R2, 자료 선별 R3)

| 필드 | 값 | 의미 |
|---|---|---|
| `collection` | `papers_core` \| `ecosystem` \| `context` | 색인 분리 단위 |
| `scope` | `direct` | 선정 기술의 1차 근거 |
| | `comparison` | 계열 비교군. 선정 기술의 성능 근거로 대체하지 않는다 |
| | `ecosystem` | 인접 생태계. **직접 채택 근거가 아니다** — 인용 시 표시 |
| | `secondary` | 2차 자료(서베이). 성능 주장 근거로 인용하지 않는다 |
| `perspectives` | `research` `maturity` `market` `domain` `stakeholder` | 이 문서를 볼 수 있는 노드 |
| `applies_to` | `["TurboQuant"]` `["ITME"]` `[둘 다]` | **목록이다** |
| `excerpt_only` | `true` | 전문을 색인하지 않는다 → `excerpt_pages: [시작, 끝]` 필수. 없으면 `scripts/ingest.py`가 경고 |

`scripts/prepare_sources.py`가 매니페스트에 남기는 것(설계서 §3):

| 항목 | 설계서 근거 |
|---|---|
| `sha256`, `retrieved_at`, `url`, `version` | 원문 URL, 문서 ID, 버전, 해시를 보존한다 |
| `pdf_pages` | PDF는 실제 페이지 수를 센다 |
| `pages_counted`, `page_basis` | 웹 문서는 A4·9.5pt 기준 **1쪽 = 3,200자**로 환산한다 |
| `total_pages`, `within_budget` | 합계가 200쪽을 넘으면 적재를 중단한다 |

- 기본 동작은 해시 검증이고, 재수집은 `--refresh`를 명시해야 한다. `setup` 노드도 적재 시
  해시를 대조하고 어긋나면 실행을 멈춘다.
- 청킹은 **임베딩 모델과 같은 토크나이저**(`intfloat/multilingual-e5-small`)로 자른다. 범용
  토크나이저나 글자 수로 자르면 512토큰 한도를 넘어 뒷부분이 조용히 잘린 채 임베딩된다.
- 임베딩 revision은 커밋 해시로 고정한다(`src/rag/embed.py`). `main`은 움직이는 포인터다.
- 색인은 실행마다 메모리에 새로 만들되, 비싼 두 단계를 `data/index/`(Git 제외)에 캐시한다.
  PDF 페이지 추출은 원문 sha256 + 추출 코드 소스로, 청크 벡터는 chunk_id 목록 + 임베딩 revision으로
  키를 만든다. 원문·추출 코드·청크 본문·모델 중 하나라도 바뀌면 키가 바뀌어 다시 만든다. 지워도 안전하다.
- PDF는 `x_tolerance=1.5`로 읽고 2단 페이지는 단마다 따로 읽는다(`index._page_text`). 기본값으로
  읽으면 2단 학회 논문의 공백이 사라지고 두 단의 줄이 섞여 ITME 검색 Hit@5가 0.29였다(수정 후 0.71).

---

## 변경 절차

1. 이 문서를 먼저 고친다.
2. 소유자(맨 위 표)의 승인을 받는다.
3. 코드를 고친다.

**`chunk_id`를 바꾸는 변경은 비싸다** — 청킹 파라미터(`chunk_tokens`·`chunk_overlap`),
원문 재수집(`--refresh`), 임베딩 revision. 모든 `chunk_id`가 달라져 이전 실행의 인용이
끊기고 `eval/goldenset.json`의 정답 청크 라벨을 다시 해야 한다.
