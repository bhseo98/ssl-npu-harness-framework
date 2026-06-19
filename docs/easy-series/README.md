# Easy Series — Torch-MLIR Lowering 학습 노트

> 본인 학습용 노트 모음. PyTorch/HF 모델을 우리 NPU compiler stack 의 최상단
> "PyTorch Model Zoo" 로 lowering 할 때 *무엇이 되고 무엇이 막히는지* 를
> 실측 기반으로 풀어쓴 시리즈.

| 문서 | 한 줄 요약 |
|---|---|
| [easy-sum.md](easy-sum.md) | **한 장 요약** — 두 계층 구조 + 제약 9종↔고칠 코드 + "결론: modelClass 교체가 6개 동시 제거" |
| [easy.md](easy.md) | 우리 `LlamaOnDevice` 에 real Llama-3.2-1B 가중치를 주입해 직접 MLIR → `.vmfb` 까지 뽑은 과정 (Step 5) |
| [easy2.md](easy2.md) | 유명 HF/PyTorch 모델 7종을 같은 export pipeline 에 태운 실측 — 4 통과 / 3 실패 + 근본 원인 (입고 기준) |
| [easy3.md](easy3.md) | amdsharktank 로 real Llama-3.2-1B 를 export 했을 때의 제약 9종 → 1개 근본원인, IRPA/RoPE/PagedAttention 정의, "fixed IRPA 만으론 부족" |
| [easy4.md](easy4.md) | easy3 의 짝 — pure Llama-3.2-1B 를 amdsharktank 로 export/lowering 할 때 *쓴 클래스/코드 27종* 을 HF→IRPA→MLIR 호출 순서대로 해부 (on-device 제외) |
| [easy4-detail.md](easy4-detail.md) | **정밀판** — *export 에 쓰는 클래스(traced nn.Module)* vs *lowering 에 쓰는 클래스(@mlir_kernel)* 두 계층 구분 + 실측 IR 27,269줄과 op count 1:1 매핑 검증 + 커스터마이징 5레버 |
| [easy5-fig.md](easy5-fig.md) | **그림판** — 위 내용을 도식으로. 원본 [../diagrams/export-vs-lowering.drawio](../diagrams/export-vs-lowering.drawio) (3페이지) + Mermaid 미리보기 |
| [easy6.md](easy6.md) | **후속 Q&A** — ①input dim(dynamic/static)은 *어느 클래스*가 만드나(모델 아님, export 드라이버 `torch.export.Dim`) + "dynamic이라 NPU 못 쓴다"는 절반만 ②lowering 제약 kernel 카탈로그(기본 경로 RoPE+gather 2개 / 예비 풀) + "nn.Module이 aten만이면 lowering?" 충분조건 4개 |
| [easy7.md](easy7.md) | **Whisper-tiny 확장(encoder-decoder)** — PDF Step 6 타깃. faster-whisper-int8은 CTranslate2라 입구컷 → PyTorch whisper-tiny FP32로 export. 문제 kernel=opaque SDPA 1개(Llama와 동일, eager로 해결), conv·cross-attn은 표준·RoPE 없음. 성공 클래스=`WhisperForwardOnly` 4-가드 wrapper(~10 LoC) |
| [easy6-sum.md](easy6-sum.md) | **Llama-3.2-1B 종합표** — 전권을 한 모델로 압축. 표 A(클래스 27종이 어떻게 쓰이나) + 표 B(막히는 커널 4곳: rope concat·KV gather·SDPA 불투명·affinity) + 표 C(제약9↔클래스↔레버) + easy5-fig Fig 1/2 count 매핑 연동 |
| [easy8.md](easy8.md) | **amd가 왜·어떻게 커널을 우회했나** — RoPE concat·KV gather·`mmt_block_scaled_q8`(INT8)·flash attention 4종을 실소스로 분석. 공통 동기=fusion/materialize 회피, 수법=`@mlir_kernel`로 linalg/iree_linalg_ext splice. INT8 q8 표준 재표현 스케치 포함 |
| [easy8-detail.md](easy8-detail.md) | **easy8 정독판(서술형)** — 같은 4커널을 소스 한 줄씩 따라가며 풀어쓴 노트 + verbatim 코드 + [kernel-bypass.drawio](../diagrams/kernel-bypass.drawio)/Mermaid 그림. "AMD가 손으로 한 fusion을 우리는 컴파일러에 떠넘긴다"로 마무리 |
| [easy8-qna.md](easy8-qna.md) | **easy8 후속 Q&A** — ④ softmax→LayerNorm 경량화 대체 가능한가(아니오: 축·의미·마스킹 차이, 게다가 막는 건 fused flash op) + ③ `CustomOp`/`mmt_block_scaled_q8` 동작 6문답(Python=컴파일시점 메타코드, `.mlir`=실제계산, import 등록→select→generate, `ops.matmul` 자동 디스패치) |
| → 다음 단계 | **SSLNPU 커널 *설계* 문서는 [../kernel-design/](../kernel-design/)로 졸업** — easy 시리즈는 lowering 분석(easy1~8)까지. 커널 직접 저작 가이드라인·계획은 [kernel-design/01-guideline-and-plan.md](../kernel-design/01-guideline-and-plan.md) |

## 그림

5페이지 다이어그램으로 전체 순서(두 경로 비교 / amdsharktank 순서 / LlamaOnDevice 순서 / 6단계 로드맵 / 임의 모델 입고 순서)를 정리:
[../diagrams/lowering-order.drawio](../diagrams/lowering-order.drawio) (draw.io / VS Code draw.io 확장에서 열기).

페이지별 SVG 익스포트:
- [../diagrams/compare-2path.svg](../diagrams/compare-2path.svg) — 두 경로 비교
- [../diagrams/amdsharktankexport.svg](../diagrams/amdsharktankexport.svg) — 경로 A (amdsharktank) 순서 + 9제약
- [../diagrams/llamaondevicepipeline.svg](../diagrams/llamaondevicepipeline.svg) — 경로 B (LlamaOnDevice) 4-stage
- [../diagrams/attachrandommodels.svg](../diagrams/attachrandommodels.svg) — 임의 모델 입고 순서

easy4-detail / easy5-fig 전용 (클래스↔IR):
- [../diagrams/export-vs-lowering.drawio](../diagrams/export-vs-lowering.drawio) — ①export-time vs lowering-time 클래스 ②prefill_bs1 1:1 매핑 ③커스터마이징 5레버

easy8 전용 (커널 우회):
- [../diagrams/kernel-bypass.drawio](../diagrams/kernel-bypass.drawio) — Page1 4커널 우회 한눈에(표준 aten→우회 IR→동기→NPU 판정) / Page2 ③ q8 INT8 dequant-fused matmul dataflow
- [../diagrams/fusion-concept.drawio](../diagrams/fusion-concept.drawio) — Page1 fusion 개념(A\*B+C, naive 6왕복 vs fusion 4왕복) / Page2 fusion은 누가 하나(AOTInductor vs iree-turbine/IREE, 공통 torch.export)

## 읽는 순서

easy.md (1개 모델 성공) → easy2.md (임의 모델은?) → easy3.md (server-side 는 왜 막히나) → easy4.md/easy4-detail.md (어떤 클래스로 그렇게 되나) → easy5-fig.md (그림으로).

## 관련 문서

- [../SHARK_AI_ANALYSIS.md](../SHARK_AI_ANALYSIS.md) — amdsharktank 사용 가드 (Step 3)
- [../../development.md](../../development.md) — 전체 status board
- [../../my.md](../../my.md) — transformer vs KV cache 해석 노트
