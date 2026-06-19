# 📂 docs — 문서 지도 (Documentation Map)

> 문서를 **시기/성격별로 분리**했다. **🟢 활성(최신)** 은 상단, **🧊 구버전** 은 `archive/`, **📓 개인 학습** 은 `easy-series/`.
> 전체 구조 그림: [diagrams/doc-map.drawio](diagrams/doc-map.drawio)

```
docs/
├── README.md            ← 여기 (문서 지도)
├── 🟢 architecture.md    시스템 아키텍처 (컨테이너·컴포넌트·시퀀스·ADR)
├── 🟢 status.md          현황 단일 소스 (deliverable 달성도·Phase A·미해결 결정)
├── 🟢 guidelines.md      엔지니어링 가이드라인
├── 🟢 decisions-qa.md    세션 질문 → 결정 인덱스
├── 🟢 recipes/           재현 절차
│     ├── kernels.md      INT8 커널·whisper lowering·Rust·riscv64
│     ├── vp.md           VP 부팅·런타임 아키텍처·SoC 버스/메모리 맵
│     ├── webui-mic.md    실마이크 WebUI push-to-talk
│     └── zoo.md          일반 PyTorch→MLIR→vmfb
├── 🟢 kernel-design/     SSLNPU 커널 설계 (01 가이드·02 lowering·03 커널+메모리·04 INT8 footprint)
├── 🟢 diagrams/          시각화 (drawio) — 아래 표
├── 🧊 archive/           구버전(~2026-05, superseded): architecture·shark-ai-analysis·development·dated status
├── 📓 easy-series/        개인 학습 노트(별도 트랙, easy1~9)
└── 📁 reports/            로컬 전용 (git 제외 — weekly/research note)
```

---

## 🟢 활성 문서 (recent)

| 문서 | 한 줄 | 그림 |
|---|---|---|
| [architecture.md](architecture.md) | 컨테이너·컴포넌트·런타임 시퀀스·ADR (빌드=5번/호스트=4번/추론=VP, 2 swap 이음새, 2 양자화 트랙) | [system-architecture](diagrams/system-architecture.drawio) |
| [status.md](status.md) | 3 deliverable 달성도 + Phase A 실측 + 미해결 결정 (**현황 단일 소스**) | [deliverables-status](diagrams/deliverables-status.drawio) |
| [guidelines.md](guidelines.md) | 스택 경계·deliverable 정의·검증 게이트·커널 규칙·메모리 우선순위·작업 규율 | |
| [decisions-qa.md](decisions-qa.md) | 세션 질문 → 답/결정 → 문서 위치 검증 인덱스 | |
| [recipes/kernels.md](recipes/kernels.md) | WM3·INT8/FP32·RK1/RK2·VP0·swap 이음새 재현 | [lowering-pipeline](diagrams/lowering-pipeline.drawio) |
| [recipes/vp.md](recipes/vp.md) | ssl-soc-gen2 부팅·런타임 아키텍처·riscv64 빌드·SoC 맵 | [vp-runtime-architecture](diagrams/vp-runtime-architecture.drawio) · [vp-boot-chain](diagrams/vp-boot-chain.drawio) |
| [recipes/webui-mic.md](recipes/webui-mic.md) | WebUI push-to-talk(① 서버 디딤돌 → ③ VP 통합) | [vp-runtime-architecture](diagrams/vp-runtime-architecture.drawio) |
| [recipes/zoo.md](recipes/zoo.md) | 일반 PyTorch→MLIR→vmfb lowering | |
| [kernel-design/](kernel-design/) | SSLNPU 커널 설계 01~04 + README | [sslnpu-kernel-anatomy](diagrams/sslnpu-kernel-anatomy.drawio) |

## 🗂 시각화 (diagrams/)

| 그림 | 무엇 |
|---|---|
| [doc-map.drawio](diagrams/doc-map.drawio) | 문서 구조 지도(이 README의 그림판) |
| [deliverables-status.drawio](diagrams/deliverables-status.drawio) | 3 deliverable + 현재 상태 보드 |
| [vp-runtime-architecture.drawio](diagrams/vp-runtime-architecture.drawio) | VP 런타임 2-plane(호스트+게스트), CPU/NPU 트랙, SoC 버스 |
| [vp-boot-chain.drawio](diagrams/vp-boot-chain.drawio) | ssl-soc-gen2 부팅 체인 + 메모리 맵 |
| [lowering-pipeline.drawio](diagrams/lowering-pipeline.drawio) | PyTorch→INT8→MLIR→block_scaled_q8→vmfb→IREE-CPU + swap 이음새 |
| [sslnpu-kernel-anatomy.drawio](diagrams/sslnpu-kernel-anatomy.drawio) | CustomOp 4-파트 + 스택 담당 경계 |
| [kernel-bypass.drawio](diagrams/kernel-bypass.drawio) · [fusion-concept.drawio](diagrams/fusion-concept.drawio) | 커널 우회·fusion 개념 (학습) |
| [lowering-order.drawio](diagrams/lowering-order.drawio) · [export-vs-lowering.drawio](diagrams/export-vs-lowering.drawio) | export vs lowering 2계층 (학습) |

## 🧊 archive/ — 구버전 (superseded, ~2026-05)

[architecture](archive/architecture.md) · [shark-ai-analysis](archive/shark-ai-analysis.md) · [development](archive/development.md) · [2026-05-27 audit](archive/2026-05-27-duplication-audit.md) / [verification](archive/2026-05-27-verification.md) · [2026-05-28 status](archive/2026-05-28-project-status.md) · [llama-step5](archive/llama-step5-visual.md)

## 📓 easy-series/ — 개인 학습 노트 (별도 트랙)

[easy-series/](easy-series/) — Torch-MLIR lowering 학습(easy1~9). 포멀 문서와 분리 유지.
