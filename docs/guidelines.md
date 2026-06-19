# Guidelines — NPU Harness Framework (Engineering)

> 프로젝트 전반의 **포멀 엔지니어링 가이드라인** — 무엇이 내 영역인가, 무엇을 "완료"로 보는가,
> 커널을 어떻게 짜는가, 메모리를 어떻게 줄이는가, 작업 규율은 무엇인가.
> 행위 규칙(LLM 코딩 실수 방지)은 [`CLAUDE.md`](../CLAUDE.md), 상세 설계는 [kernel-design/](kernel-design/),
> 재현 절차는 [RECIPES-zoo.md](recipes/zoo.md)/[RECIPES-kernels.md](recipes/kernels.md)/[RECIPES-vp.md](recipes/vp.md).
>
> **개인 학습 노트([easy-series/easy-all.md](easy-series/easy-all.md))는 본 포멀 가이드라인의 일부가 아니라 별도 트랙으로 유지한다.**

---

## 1. 스택 경계 — 어디까지가 내 영역인가

target NPU compiler stack에서 내 영역은 **최상단 점선 박스**(PyTorch Model Zoo + Torch-MLIR + Torch-MLIR Compiler).

| 레이어 | 담당 | 지금 가능? |
|---|---|---|
| PyTorch Model Zoo / Torch-MLIR / Torch-MLIR Compiler | **나** | ✅ 전부(IR 생성·패스) |
| IREE Compiler / Runtime / VM | 기존 오픈소스 | ✅ **CPU로 컴파일·실행 검증** |
| IREE HAL Driver (SSLNPU 타깃) | runtime팀 | ⏳ 대기 |
| SSLNPU Library / Runtime / Driver / HW | runtime·HW팀 | ⏳ 대기 |

- **내 천장 = CPU-via-IREE end-to-end 컴파일 + 실행 + 수치 검증.** 그 위에서 IREE HAL을 접점으로 runtime팀과 합쳐지면 실제 칩 실행이 완성된다.
- 시간 임계경로는 내 Python class가 아니라 그 아래 **SSLNPU 백엔드 bring-up**이다 → 인터페이스는 고정하고 본체만 swap.

---

## 2. 세 Deliverable — 정의와 "완료" 기준

| # | Deliverable | 완료 기준 |
|---|---|---|
| ① | **음성 앱**(STT→LLM→TTS) + 프레임워크 코어 | KO 1턴 동작 · `@register`+YAML 한 줄 모델 swap · 2GB budget profiler 실측 · BaseStage/registry/pipeline 불변량 |
| ② | **PyTorch→MLIR lowering** 프레임워크(for target application) | `server_side_op_hits == {}` · CustomOp+마이크로커널(`block_scaled_q8`) · vmfb IREE-CPU 실행 + FP32 수치 일치 · swap 이음새(`MICROKERNEL_BACKEND`) |
| ③ | **CPU-only SoC VP / RISC-V64 배포** | VP 위 voice app 1턴 · CPU only(NPU 미접근) · 합 ≤2GB · per-stage latency 기록 · llama.cpp 동작 |

불변 제약(모든 deliverable 공통): **합 ≤2GB DRAM · CPU-only(NPU는 나중) · 한국어 · "advanced 기법 자제"**(INT8은 명시 예외).

---

## 3. 검증 게이트 규율

**모든 마일스톤은 실제 실행 출력으로 증명한다(string-match 금지).** 약한 기준("동작하게")은 끝없는 재질문을 부르고, 강한 기준은 독립 루프를 가능케 한다.

- 태스크 → 검증 가능한 목표로 변환: "validation 추가"→"invalid 입력 테스트 작성 후 통과", "버그 수정"→"재현 테스트 작성 후 통과".
- 양자화/경량화 등 트레이드오프는 **before/after 실측**으로만 정당화(예: FP32 vmfb 150.5MB → INT8 125.4MB).
- Phase A 게이트 실측 기록: WM3(rel 0.8%/argmax 100%) · RK1(rel 3.5e-7) · RK2(RSS 3–4.6×) · VP0(riscv64 vmfb 컴파일). [RECIPES-kernels.md](recipes/kernels.md).

---

## 4. 커널 구성 규칙

- **CustomOp 4-파트**: 등록 + `select`(shape 계약) + dispatch + 모델 클래스. 이 Python 층은 **백엔드 무관 → 지금 전부 설계·검증 가능**. 마이크로커널 *본체*(.mlir)만 SSLNPU IR 인계 후 채운다.
- **custom vs 표준 결정규칙**: SSLNPU 전용 데이터패스/intrinsic가 필요하면 custom(예: INT8 block-scaled matmul), 그 외엔 표준 분해(SDPA/RMSNorm/SwiGLU/TopK)로 컴파일러 fusion에 맡긴다.
- **단일 swap 이음새**: `MICROKERNEL_BACKEND = "standard" → "sslnpu"`. `select`(인터페이스)는 고정, `generate`의 방출 타깃만 표준 linalg(지금, 포터블) → SSLNPU intrinsic(나중)으로 교체.
- 상세 [kernel-design/01](kernel-design/01-guideline-and-plan.md) · [02](kernel-design/02-lowering-layer.md).

---

## 5. 메모리 최적화 — 우선순위

원칙: **(1) 측정해 큰 덩이부터 → (2) compute↔memory 트레이드오프 의식 → (3) 메모리 계층(SRAM↔DRAM↔storage) 이용.** 정적/동적은 독립이라 효과가 곱해진다.

```
1. 측정                  → 큰 덩이 식별 (whisper: 임베딩 80MB가 범인)
2. weight mmap 외부화      → 정확도 0손실, RSS 구조적 절감 (먼저)
3. 임베딩 INT8 + re-tie    → vmfb 125 → ~45MB
4. INT4 / 혼합정밀          → matmul weight 추가 ~2×, 정확도 게이트
5. flash-attention         → activation O(T²) → O(T), 긴 오디오
6. paged + 양자화 KV(생성형) → 같은 예산에 ~4× 컨텍스트
```

**양자화는 MLIR lowering 후 마지막.** FP32 path가 동작하고 budget 위반이 *측정*된 후에만 적용. 상세 [kernel-design/03](kernel-design/03-kernel-survey-and-memory.md).

**온디바이스(VP) 풋프린트는 *런타임*이 아니라 *부팅*이 천장일 수 있다 (2026-06-16 실측).** 블록디바이스 없는 VP는 모델을 initramfs(rootfs.cpio)로 RAM에 적재한다. 부팅 시 커널이 cpio를 unpack하며 **원본 cpio + 풀린 ramfs가 동시 상주**해 ~2× RAM이 든다 → 2GB 칩의 부팅 천장은 **rootfs ~620MB**(실측: 617 성공, 675/722/833 `Initramfs unpacking failed: write error` → panic). 즉 런타임 여유(qwen 추론 peak ~950MB/2GB, 여유 ~1GB)가 충분해도 **큰 모델은 부팅에서 막힌다**. 결론: **2GB 온칩에선 "작은 모델 + Q8" > "큰 모델 + 극단 양자화"**(Llama-3.2-1B는 Q2_K만 부팅되나 한국어 깨짐 → qwen-0.5B-Q8 유지). ⚠ DTB `linux,initrd-end` 창을 키우면 오히려 악화(창은 *예약*이라 unpack용 available RAM↓). 더 큰 모델을 올리려면 **부팅 2× 우회**(압축 initrd=고엔트로피 모델엔 무효 / in-place·no-copy initrd / 블록디바이스)가 필요 — *미해결 과제*. 재현/상세 [RECIPES-vp.md §6.2](recipes/vp.md).

---

## 6. 작업 규율 (불변)

- **언어/태도**: 한국어 · 신중한 step-by-step · 실측 기반. Haiku 모델 사용 금지.
- **venv**: 외부 의존 작업 전 항상 가상환경(ML=venv-shark, 빌드=.buildenv, Rust=~/.cargo 유저로컬). 시스템 pip 금지.
- **surgical change**: 요청에 직접 추적되는 변경만. 인접 코드 "개선"·미요청 추상화 금지. 내 변경이 만든 orphan만 정리.
- **push 규율**: `push` 명시 시에만 origin(**private** `bhseo98/npu-harness-framework`)만. 공개 미러는 `ssl-mirror-sync` 에이전트로만. 의도한 파일만 staging(`git add -A`/`commit -a` 금지).
- **로컬 전용**: weekly/연구노트(`docs/reports/*weekly*`, `*research-note*`)는 git 금지·로컬.
- **푸시 전 민감어 스캔**: 내부 인물·조직·회의·주간보고·미러 키워드가 산출물에 없는지 확인.
- **파괴적 작업 가드**: force-push / `branch -D` / main 직접 push / history rewrite는 명시 승인 없이 금지.
- **비밀정보**: SSH 비밀번호·토큰을 채팅/문서에 붙이지 않는다(키 인증 선호).

---

## 7. 개인 학습 트랙 (별도)

[easy-series/easy-all.md](easy-series/easy-all.md)(easy1~8 통합 단일 문서: Torch-MLIR lowering 학습 노트, 커널 우회 분석, CustomOp 동작)는 **개인 학습 기록**으로 본 포멀 가이드라인과 분리해 유지한다. 포멀 문서는 결론·계약·재현 절차를, easy-series는 그 결론에 이르는 탐색 과정을 담는다. (분리됐던 easy1~8 파일들을 2026-06-19 단일 파일로 통합.)
