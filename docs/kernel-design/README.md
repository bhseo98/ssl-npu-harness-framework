# Kernel Design — SSLNPU CustomOp 커널 설계

> SSLNPU용 커널을 *직접 저작*하는 단계의 설계 문서 모음.
> 배경 학습 노트(Torch-MLIR lowering·커널 우회 분석)는 [../easy-series/](../easy-series/)에 있고, 이 폴더는 그 위에서 "우리가 무엇을 어떻게 짜는가"의 설계·계획을 다룬다.

| 문서 | 한 줄 |
|---|---|
| [01-guideline-and-plan.md](01-guideline-and-plan.md) | 스택 담당 경계(최상단 점선=내 영역 / IREE 기존 / SSLNPU Library·Runtime=runtime팀, 합치는 지점=IREE HAL, 내 천장=CPU-via-IREE) + 커널 구성 가이드(CustomOp 4-파트, custom vs 표준 결정규칙) + 호출함수 3형태 + 마이크로커널 시퀀싱(class 지금/intrinsic 인계 후 `generate` swap) + 작업 계획 + 시간 임계경로 |
| [02-lowering-layer.md](02-lowering-layer.md) | 내 몫의 "컴파일러" = compile-time lowering 계층(`select` 계약 → `generate` emit → `.mlir` splice)의 구조. swap 이음새 1곳(`MICROKERNEL_BACKEND`) + 안정적 계약(`select`) + 마이크로커널 해부 + whisper-tiny INT8 워크드 예시(65 call → 4 unique 커널) + 검증 게이트(M1·WM2 green / WM3 다음) |
| [03-kernel-survey-and-memory.md](03-kernel-survey-and-memory.md) | block_scaled_q8 외 커널 전수 점검(임베딩·attention·norm·softmax·RoPE·conv…) 제약/개선 표 + whisper lowering 제약(easy7 대비, 새 블로커 없음) + 시스템 관점 메모리 최적화(weight mmap·임베딩 INT8·flash-attn·paged/quant KV) + 기대효과·우선순위 |

## 그림

- [../diagrams/sslnpu-kernel-anatomy.drawio](../diagrams/sslnpu-kernel-anatomy.drawio) — Page1 스택·담당 경계 / Page2 CustomOp 4-파트 + now(표준 linalg)→later(SSLNPU intrinsic) swap

## 관련

- [../easy-series/](../easy-series/) — Torch-MLIR lowering 학습 노트(easy1~8): 커널 우회 분석, CustomOp 동작, on-device 전략
- 스캐폴드(예정): `src/torch_mlir_zoo/kernels/` — INT8 block-scaled matmul CustomOp + 모델 클래스
