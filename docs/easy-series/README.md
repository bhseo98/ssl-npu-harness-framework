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

## 읽는 순서

easy.md (1개 모델 성공) → easy2.md (임의 모델은?) → easy3.md (server-side 는 왜 막히나) → easy4.md/easy4-detail.md (어떤 클래스로 그렇게 되나) → easy5-fig.md (그림으로).

## 관련 문서

- [../SHARK_AI_ANALYSIS.md](../SHARK_AI_ANALYSIS.md) — amdsharktank 사용 가드 (Step 3)
- [../../development.md](../../development.md) — 전체 status board
- [../../my.md](../../my.md) — transformer vs KV cache 해석 노트
