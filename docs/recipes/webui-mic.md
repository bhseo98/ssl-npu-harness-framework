# Recipes — WebUI Push-to-Talk 실마이크 (① 서버 디딤돌 → ③ VP 통합)

> 실제 **마이크**로 브라우저 push-to-talk → STT→LLM→TTS 1턴.
> **2단계로 본다:**
> - **목표 = ③ VP 통합**(§0, Option A): 브라우저=마이크/스피커, 호스트가 WebUI를 서빙, **추론(STT/LLM/TTS)은 VP 게스트 엔진**이 CPU-only로 수행. ← *아직 실행 불가, 선행조건 미완.*
> - **지금 실행 가능 = ① 서버 디딤돌**(§1~): `web_demo.py`가 **서버에서 직접** 추론. UI·파이프라인·KO 1턴을 먼저 검증하는 디딤돌(추론 위치는 서버이지 VP 아님).
>
> 구현: `speech:scripts/web_demo.py`(FastAPI+WS, 라우트 `GET /`·`GET /health`·`WS /ws`, 브라우저 16kHz PCM16 AudioWorklet). Docker `/dev/snd` 불필요.
> 그림: [diagrams/vp-runtime-architecture.drawio](../diagrams/vp-runtime-architecture.drawio)(Option A).

---

## 0. 목표 아키텍처 — ③ VP 통합 (Option A)  ⚠ 아직 실행 불가

원래 목적("엣지 임베디드 시스템이 대화") = **추론이 VP에서** 돌아야 한다. 브라우저는 마이크/스피커일 뿐.

```
[브라우저] 마이크 ─WS─▶ [호스트 server4] FastAPI WebUI(web_demo, 그대로)
                                     │  VoicePipeline stage = VP 엔진 어댑터(vp_*)
                                     ▼  콘솔(stdin 주입) + 모델 볼륨
                           [VP 게스트] whisper.cpp(tiny INT8) → llama-cli(Llama Q8) → TTS
                                     ▼
[브라우저] 스피커 ◀──────────────── TTS wav
```

- **무엇만 바뀌나**: 브라우저·web_demo·**프레임워크 코어는 불변**. YAML에서 stage `type:`만 `vp_whisper`/`vp_llama`/`vp_tts`로 → **모델 swap 철학 그대로**. 어댑터가 콘솔로 게스트 엔진을 호출하고 모델 볼륨으로 I/O.
- **compute = VP**(CPU-only, 합 ≤2GB), 계측(per-stage latency JSONL)·오케스트레이션은 호스트.

**선행조건(미완, 순서대로) — 이게 끝나야 이 절이 "실행 가능"이 된다:**

| # | 선행조건 | 상태 | 참조 |
|---|---|---|---|
| 1 | VP 게스트에서 **실제 토큰 생성**(작은 GGUF) | ❌ (`--version`만 OK) | [RECIPES-vp §6](vp.md) |
| 2 | **whisper.cpp(tiny INT8)·TTS** riscv64 크로스빌드 → rootfs/볼륨 | ❌ | [RECIPES-vp §4](vp.md) |
| 3 | **`vp_*` 어댑터**(`@register("llm","vp_llama")` 등): 콘솔 호출 + 볼륨 I/O + 출력 파싱 | ❌ 코드 없음 | — |
| 4 | **모델 볼륨**(sdcard.img) 마운트 + 가중치 배치 | ❌ | [RECIPES-vp §5](vp.md) |
| 5 | 호스트 web_demo 기동 + 브라우저 터널 접속(§3) | (1~4 후) | §3 |

**검증 게이트(③ 충족 기준):** 브라우저 1턴 성공 **AND** compute가 VP에서(게스트 RSS·per-stage latency 캡처) **AND** 합 ≤2GB **AND** CPU-only(NPU 미접근). → 현재 ❌. **다음 1수 = 선행조건 1(VP-A 토큰 생성).**

---

## 1. 지금 실행 가능 — ① 서버 디딤돌

> ⚠ 여기서는 **추론이 서버(venv-shark)에서** 돈다(VP 아님). UI·파이프라인·KO 1턴·budget을 먼저 검증하는 디딤돌.

```
[브라우저] 마이크 → PCM16 16kHz → WS /ws ─┐
                                          ▼
[서버] web_demo(FastAPI) → VoicePipeline: STT → LLM → TTS  ← 추론이 여기(서버)
                                          │
[브라우저] 스피커 ◄─ TTS wav ◄────────────┘
```

target 조합(서버 검증용): STT=whisper-tiny INT8 · LLM=Llama-3.2-1B · TTS=조합 최적 KO(선정 PENDING). 현 `configs/default.yaml`은 interim.

### 1.1 사전 준비 (서버, 한 번)

```bash
git switch speech                       # 또는 git worktree add ../speech speech
source /home/bohyun/venv-shark/bin/activate
pip install fastapi uvicorn soundfile numpy pyyaml
python -c "import fastapi, uvicorn, soundfile; print('web deps ok')"
```

### 1.2 서버 띄우기

```bash
python scripts/web_demo.py -c configs/default.yaml --host 0.0.0.0 --port 8000
#   TARGET_APP_CONFIG=configs/embedded.yaml 으로도 지정 가능
curl -s http://localhost:8000/health | python -m json.tool   # config·stt/llm/tts identity
```

### 1.3 브라우저 접속 — ⚠ secure context (마이크의 핵심 함정)

브라우저 `getUserMedia`(마이크)는 **secure context**(`https://` 또는 **`localhost`**)에서만 동작.
원격 `http://<서버IP>:8000`로 바로 열면 **마이크 차단**. 택1:

- **(권장) SSH 로컬 포워딩** — 내 PC에서:
  ```bash
  ssh -L 8000:localhost:8000 bohyun@<gpu5-lab>
  # 브라우저: http://localhost:8000   (localhost = secure context → 마이크 OK)
  ```
- HTTPS 리버스 프록시(caddy/nginx + 인증서).
- (dev 한정) Chrome `chrome://flags` → *Insecure origins treated as secure*.

접속 후 **Start**(마이크 허용) → 말하기 → **Stop** → STT→LLM→TTS → 응답 음성 자동 재생.

### 1.4 1턴 검증 (가이드라인)

| 체크 | 통과 기준 | 어디서 |
|---|---|---|
| STT 전사 | 말한 한국어가 텍스트로 맞게 | transcript / 서버 로그 |
| LLM 응답 | 한국어로 맥락 맞는 답 | transcript |
| TTS 재생 | 응답 음성이 들림 | 브라우저 자동재생 |
| per-stage latency | STT/LLM/TTS·E2E ms | profiler JSONL(`logs/*.jsonl`) |
| 2GB budget | 초과 시 경고 | `default.yaml: budget_mb` / `embedded.yaml` sequential |

- 무음/낮은 RMS → **silence reject**(`MIN_INPUT_RMS`) "No audio captured"(정상). 녹음 **최대 60초**.
- budget 엄격 확인은 `configs/embedded.yaml`(sequential_load, CPU, whisper-tiny)로 stage peak ≤2GB.

### 1.5 마이크 없이 회귀(배치)

```bash
python scripts/bench_swap.py    # 조합별 E2E/STT/LLM/TTS latency+RAM → logs/bench-swap/
```

---

## 2. gotcha

| 증상 | 원인 | 해결 |
|---|---|---|
| 마이크 권한 안 뜸 / `getUserMedia undefined` | secure context 아님(원격 http) | §1.3 SSH 포워딩(localhost)/HTTPS |
| 페이지 열리는데 Start 무반응 | 브라우저 마이크 차단 | 자물쇠 → 마이크 허용, localhost 접속 |
| "No audio captured" | 무음/낮은 RMS silence reject | 또렷이·가까이 재시도 |
| WS 연결 끊김 | 포트 미개방/프록시 | `/health` 200, 포워딩 포트 일치 |
| 모델 로드 OOM/느림 | 큰 LLM + GPU 부재 | `embedded.yaml`(CPU/sequential)/경량 LLM |
| "VP에서 도는 줄 알았는데 서버서 돔" | §1은 ① 디딤돌(추론=서버) | ③ VP 통합은 §0 선행조건 완료 후 |

---

## 3. 참고
- 구현: `speech:scripts/web_demo.py` · 데모 스냅샷: `speech:docs/reports/2026-05-21-phase1-demo.md`
- VP 런타임 아키텍처(Option A) 상세: [RECIPES-vp.md §0.5](vp.md) · 결정 인덱스: [DECISIONS-QA.md](../decisions-qa.md)
- 모델 swap/플러그인: `speech:RECIPES.md` · 코어 불변량: [GUIDELINES.md](../guidelines.md)
