# Status — NPU Harness Framework (2026-06-10)

> 세 deliverable의 현재 달성도 + Phase A 실측 + 미해결 결정사항을 한곳에 정리한 **living status**.
> 판정은 코드/로그 직접 검증 + 마일스톤 실측 근거. 재현은 각 RECIPES 문서.
> 📊 상태 그림: [diagrams/deliverables-status.drawio](diagrams/deliverables-status.drawio) · 문서 지도: [README](README.md)

---

## 1. Deliverable 달성도 (실측 기반 더블체크)

| # | Deliverable | 판정 | 핵심 근거 | 가장 큰 갭 |
|---|---|---|---|---|
| ① | 음성 앱(STT→LLM→TTS) + 코어 | **달성** | 코어 4파일 계약 일치, 테스트 28/28, profiler가 2GB budget 실측(초과 시 경고+JSONL), swap=@register+YAML 한 줄, KO 1턴 E2E ~2.1s(리포트) | **TTS 음질** — 현 lock=espnet_kss(무거움). 2GB 내 KO TTS 교체 PENDING(MeloTTS-Korean은 후보 개념, 코드 0건) |
| ② | PyTorch→MLIR lowering(for target app) | **부분달성** | `block_scaled_q8` swap 이음새 실재, WM3(whisper INT8 전체 vmfb IREE-CPU 실행, `server_side_op_hits {}`, rel 0.61%/argmax 100%, vmfb 103MB — proj_out tie 수정), RK1/RK2(torch 패리티 3.5e-7, RSS 3–4.6× 작음) | **target 모델 정렬** — zoo lowering=Llama+whisper인데 앱 baseline=Qwen+espnet. "for target application"의 *target* 불일치(아래 §3) |
| ③ | CPU-only SoC VP / RISC-V64 | **부분(부팅까지)** | VP `ssl-soc-gen2`(DDR3 2GB) 부팅 성공(OpenSBI v1.5→Linux 6.16→login), 게스트 `llama-cli --version` EXIT=0, VP0 riscv64 vmfb 컴파일 가능 | **실제 토큰 생성 0건** — 게스트에 runnable GGUF 없음(20MB initramfs 한계). STT/TTS/오케스트레이션 미착수 |

불변 제약(2GB·CPU-only·KO·"advanced 자제") **위반 없음**. INT8은 명시 예외.

---

## 2. Phase A 마일스톤 (완료, 실측)

| 마일스톤 | 게이트 | 결과 |
|---|---|---|
| WM3 | whisper-tiny INT8 전체 vmfb IREE-CPU 실행 + FP32 대조 | ✅ rel 0.61% / argmax 100% (proj_out tie 수정 후; 수정 전 0.8%) |
| (bench) | FP32 vs INT8 정적/지연 | vmfb 150.5→103.0MB(−32%, proj_out tie 수정 후; 수정 전 125.4MB), 지연 동등 — [04-int8-footprint](kernel-design/04-int8-footprint.md) |
| RK1 | Rust 네이티브 커널 == torch | ✅ rel 3.5e-7 |
| RK2 | IREE vs Rust 풋프린트/지연 | ✅ Rust peak RSS 3–4.6× 작음, 지연 비등 |
| VP0 | riscv64 vmfb 컴파일 | ✅ 커널 178KB / whisper 125MB (수정 전; proj_out tie 수정 후 호스트 103MB, riscv64 재측정 대기) |
| VP boot | ssl-soc-gen2 부팅 + 게스트 엔진 실행 | ✅ login + `llama-cli --version` EXIT=0 |

재현: [RECIPES-kernels.md](recipes/kernels.md) · [RECIPES-vp.md](recipes/vp.md).

---

## 3. 미해결 결정사항 (사용자 확인 필요)

1. **모델 통일(②↔①) — ✅ 해소(2026-06-11)**: **target = Llama-3.2-1B 통일**(앱·lowering·VP 동일). Qwen2.5-0.5B는 비교 reference로 강등. swap bench가 llama+piper E2E 이미 입증 → 전환은 config 중심 + 회귀 확인. 확정 target: WebUI push-to-talk → whisper-tiny INT8 → Llama-3.2-1B → TTS(조합 최적 KO, 선정 PENDING).
2. **2GB 정의** — bench의 `ram_after_mb ~4.4GB`는 하네스(PyTorch+컴파일) RSS이지 배포 RSS 아님. VP 실배포 RSS(최소 런타임+mmap)를 별도 측정해야 "2GB fit"을 진짜로 말할 수 있음.
3. **STT 정책 — ✅ 해소(2026-06-11)**: target STT = **whisper-tiny INT8**(우리 lowering, WM3). park 조건이던 MLIR lowering 완료로 park 해제. speech 브랜치 `default.yaml`의 `faster_whisper` lock은 interim(과도기).
4. **VP 토큰 생성** — 작은 GGUF로 initramfs 확장해 첫 토큰 생성 → 1.3GB Llama는 모델 볼륨(sdcard.img). ※ VP CPU 트랙은 **GGUF/ggml(llama.cpp/whisper.cpp)** — 우리 IREE vmfb는 NPU 트랙(미래). 호스트 harness가 콘솔로 stage 오케스트레이션.

---

## 4. 다음 우선순위 (deliverable 횡단)

1. **모델 통일 결정 반영**(§3-1) — 앱 LLM을 Llama로 정렬, ②의 lowering target과 일치.
2. **VP-A 토큰 생성**(§3-4) — 경량 GGUF로 게스트 실제 추론 1건 + 콘솔 dump 캡처(③ 미달→달성 게이트).
3. **배포 RSS 분리 측정**(§3-2) — 최소 IREE 런타임 RSS 1건 → 임베딩 INT8+mmap(vmfb 103→~46MB; [04-int8-footprint](kernel-design/04-int8-footprint.md) S3).

---

## 5. 문서 지도

- 결정/Q&A 인덱스: [DECISIONS-QA.md](decisions-qa.md) · 가이드라인: [GUIDELINES.md](guidelines.md)
- 레시피: [RECIPES-zoo.md](recipes/zoo.md)(일반 lowering) · [RECIPES-kernels.md](recipes/kernels.md)(INT8 커널/Rust/riscv64) · [RECIPES-vp.md](recipes/vp.md)(VP 부팅/배포) · [RECIPES-webui-mic.md](recipes/webui-mic.md)(실마이크 WebUI)
- 그림: [diagrams/vp-runtime-architecture.drawio](diagrams/vp-runtime-architecture.drawio)
- 설계: [kernel-design/](kernel-design/) (01 가이드·02 lowering 계층·03 커널 전수+메모리)
- 학습 노트(개인, 별도): [easy-series/](easy-series/)
