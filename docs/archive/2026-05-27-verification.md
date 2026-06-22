> ⚠ **ARCHIVED (구버전, ~2026-05, superseded).** 최신 현황은 [status](../status.md) · 문서 지도 [docs/README](../README.md). 본문 링크는 원래 위치 기준일 수 있음.

# Verification Report — 2026-05-27 (pre-push)

> Push 직전 검증 결과. 사용자 원래 요구사항 ↔ 실제 산출물 1:1 검증 + 모든 결과 시각화.
>
> Verified commit: `0d62acc` (branch `torch-mlir-zoo`)
> Verifier: pytest + ls + git + grep + cat + iree-compile metadata

---

## 1. 원래 요구사항 ↔ 이행 결과 1:1 검증

> 사용자 원문: *"amd shark tank로 pytorch llama3.2-1b 모델을 MLIR lowering하고
> internal-design-spec 의 
> 바를 설명해주고 무엇을 어떻게 순차적으로 진행해야하는지 알려줘."*

```mermaid
flowchart LR
    classDef done fill:#bbf7d0,stroke:#16a34a,color:#052e16
    classDef proc fill:#dbeafe,stroke:#2563eb,color:#1e3a8a

    R1["요구 ①<br/>프로젝트 의미 설명"]:::proc --> O1["✅ development.md §1-3<br/>+ REPORT-2026-05-27 §1<br/>+ Plan agents-twinkly-pike.md Context"]:::done
    R2["요구 ②<br/>amd shark tank 로<br/>Llama-3.2-1B MLIR lowering"]:::proc --> O2["✅ iree.turbine.aot.export<br/>(AMD-SHARK-AI export pipeline)<br/>+ artifacts/...mlir 12 GB"]:::done
    R3["요구 ③<br/>PDF 
    R4["요구 ④<br/>무엇을 어떻게<br/>순차적 진행"]:::proc --> O4["✅ Step A→B→C→D plan<br/>+ 단계별 명령 + Mermaid 7개<br/>+ drawio diagram"]:::done
```

---

## 2. 산출물 존재 + 무결성 검증

```mermaid
flowchart TD
    classDef ok fill:#bbf7d0,stroke:#16a34a,color:#052e16
    classDef header fill:#fef3c7,stroke:#d97706,color:#78350f

    T["검증 카테고리"]:::header

    T --> A1["MLIR 파일<br/>artifacts/...iree-turbine.mlir"]
    A1 --> A1a["✅ 존재 (12 021 952 245 B = 12.0 GB)"]:::ok
    A1 --> A1b["✅ Header: 'module @module { util.global ...'<br/>(IREE turbine format 정상)"]:::ok
    A1 --> A1c["✅ Tail: '} } #-}'<br/>(truncate 없음 — 정상 종료)"]:::ok
    A1 --> A1d["✅ wc -l = 6 218<br/>(claim 일치)"]:::ok

    T --> A2[".vmfb 파일<br/>artifacts/...iree-turbine.vmfb"]
    A2 --> A2a["✅ 존재 (6 010 785 023 B = 6.0 GB)"]:::ok
    A2 --> A2b["✅ file 결과: Zip archive v4.5<br/>(IREE VMFB format 정상)"]:::ok

    T --> A3["summary.json"]
    A3 --> A3a["✅ server_side_op_hits = ∅"]:::ok
    A3 --> A3b["✅ has_dynamic_dim = false"]:::ok
    A3 --> A3c["✅ op_counts 정합:<br/>RMSNorm 33 · bmm 32 · softmax 16 · silu 16"]:::ok
```

---

## 3. MLIR op 분포 (실제 산출물 기반)

```mermaid
xychart-beta
    title "Llama-3.2-1B 16-layer MLIR — op count distribution (top 14)"
    x-axis ["view", "transpose", "unsqueeze", "mul", "mm", "add", "expand", "clone", "rsqrt", "pow", "mean", "slice", "split", "neg"]
    y-axis "count" 0 --> 450
    bar [402, 193, 160, 146, 113, 97, 96, 48, 33, 33, 33, 32, 32, 32]
```

```mermaid
xychart-beta
    title "Transformer 구성 op 검증 — 16 layer 정합성"
    x-axis ["RMSNorm (pow+mean+rsqrt /3)", "Attention bmm", "Attention softmax", "SwiGLU silu", "MLP mm 비율"]
    y-axis "count per Llama-1B" 0 --> 50
    bar [33, 32, 16, 16, 16]
```

> **정합 확인**: 16-layer × 2 RMSNorm + final = **33** ✓ · 16 × 2 bmm (QK^T + AV) = **32** ✓ · 16 × softmax = **16** ✓ · 16 × silu = **16** ✓

---

## 4. 파일 크기 변환 흐름 (실측)

```mermaid
xychart-beta
    title "Pipeline 단계별 데이터 크기 (GB)"
    x-axis ["HF safetensors (입력)", "PyTorch in-memory (peak)", "MLIR text (산출)", ".vmfb (산출)"]
    y-axis "GB" 0 --> 20
    bar [2.5, 12.3, 12.0, 6.0]
```

| 단계 | 크기 | 비고 |
|---|---:|---|
| HF safetensors | 2.5 GB | bf16/fp16 mixed (HF default) |
| PyTorch in-memory (peak) | 12.3 GB | trace 시 FX graph + weights 복제 |
| MLIR text | 12.0 GB | tensor literal 이 ASCII 직렬화로 inflate |
| .vmfb | 6.0 GB | raw FP32 binary 로 packed (자연 압축) |

---

## 5. 사용자 메모리 가드레일 ↔ 실제 검증

```mermaid
flowchart LR
    classDef guard fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef ok fill:#bbf7d0,stroke:#16a34a,color:#052e16

    G1["guard ①<br/>amd-shark 의 transformer<br/>안 씀, export pipeline 만 활용"]:::guard
    G1 --> G1a["✅ grep 'import amdsharktank' = 0<br/>✅ grep 'from amdsharktank' = 0<br/>✅ 5건 grep hit 은 docstring 의 *비교/대조* 언급<br/>   (실제 instantiation 0)"]:::ok

    G2["guard ②<br/>양자화는 MLIR lowering<br/>후 마지막"]:::guard
    G2 --> G2a["✅ grep 'Q8_0|int8|INT8|GGUF' = 0<br/>(src/torch_mlir_zoo/ 안 FP32 유지)"]:::ok

    G3["guard ③<br/>venv 항상"]:::guard
    G3 --> G3a["✅ 모든 명령 /home/bohyun/venv-shark/bin/python<br/>절대경로 호출"]:::ok

    G4["guard ④<br/>각 명령 확인 받기"]:::guard
    G4 --> G4a["✅ Step A→B→C→D 단계마다<br/>사용자 동의 받음 / 결과 보고"]:::ok

    G5["Design D4<br/>framework core invariant"]:::guard
    G5 --> G5a["✅ git diff main HEAD<br/>-- src/npu_harness_framework/ = 0 lines"]:::ok
```

---

## 6. 6-step 완료 매트릭스

```mermaid
flowchart TB
    classDef done fill:#bbf7d0,stroke:#16a34a,color:#052e16,stroke-width:2px
    classDef pending fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:2px

    S1["Step 1<br/>Framework 이해<br/>(Torch-MLIR/IREE/AMD-SHARK/HF)"]:::done
    S2["Step 2<br/>문제 상황 인식<br/>(HF transformers ↛ PyTorch only)"]:::done
    S3["Step 3<br/>AMD-SHARK-AI 분석<br/>(transformer 차용 X / export O)"]:::done
    S4["Step 4<br/>단위 op 모델링<br/>Attention/RMSNorm/MLP/Top-K"]:::done
    S5["Step 5<br/>LLM E2E + IR dump<br/>Llama-3.2-1B + .vmfb"]:::done
    S6["Step 6<br/>Q8_0 GGUF + Whisper-tiny-INT8<br/>(다음 turn — 양자화 마지막 원칙)"]:::pending

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

**진행률**: **5 / 6 step ✅** (83 %).

---

## 7. pytest sanity (변경 후 회귀 검증)

```mermaid
flowchart LR
    classDef ok fill:#bbf7d0,stroke:#16a34a,color:#052e16

    P["pytest tests/test_zoo_ops.py<br/>tests/test_iree_turbine_export.py<br/>-v --no-header -q"]
    P --> R["✅ 11 passed in 3.79s<br/>0 failed · 0 errored"]:::ok
    R --> N1["warning 1: NVIDIA RTX PRO 6000 sm_120<br/>vs PyTorch 2.5.1 비호환 (CPU fallback, 무해)"]
    R --> N2["warning 2: numpy 2.0 __array__ copy<br/>kwarg deprecation (iree.turbine 측, 무해)"]
```

---

## 8. Git 상태 + push 준비

```mermaid
flowchart TD
    classDef committed fill:#bbf7d0,stroke:#16a34a,color:#052e16
    classDef untracked fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef ignored fill:#e5e7eb,stroke:#6b7280,color:#374151,stroke-dasharray: 5 3

    HEAD["HEAD = 0d62acc<br/>feat(llama-step5): ... → .vmfb 완료"]:::committed
    HEAD --> C1["agents/roadmap.md<br/>scripts/run_llama_export.py<br/>configs/zoo/llama_on_device_iree_turbine.yaml<br/>REPORT-2026-05-27-...md"]:::committed

    NEXT["다음 commit (이번 turn push 대상)"]:::untracked
    NEXT --> U1["development.md (신규)"]:::untracked
    NEXT --> U2["docs/diagrams/llama-step5-visual.md (신규)"]:::untracked
    NEXT --> U3["docs/diagrams/llama-step5.drawio (신규)"]:::untracked
    NEXT --> U4["docs/diagrams/verification-2026-05-27.md (신규, 본 문서)"]:::untracked

    IGN["artifacts/ — .gitignore 자동 제외"]:::ignored
    IGN --> I1["llama-3.2-1b-on-device-iree-turbine.mlir (12 GB)"]:::ignored
    IGN --> I2["llama-3.2-1b-on-device-iree-turbine.vmfb (6 GB)"]:::ignored
    IGN --> I3["llama-3.2-1b-on-device-iree-turbine.summary.json"]:::ignored
```

**Push 정책 (사용자 명시)**:
- ✅ **origin (`bhseo98/ssl-npu-harness-framework`) 만**
- ✗ ssl-mirror (`bhseo98/ssl-npu-harness-framework`) 는 본 turn **push 안 함**
- 인증 우회: HTTPS+PAT (SSH 가 `bhseo98/Application` 으로 인증되어 본 repo 접근 불가). PAT 은 1회성 사용 후 `.git/config` 즉시 정리.

---

## 9. 검증 최종 판정

| Category | Items | Status |
|---|---:|:---:|
| 사용자 원래 요구 4종 | 4 / 4 | ✅ |
| 산출물 무결성 (MLIR, vmfb, summary) | 8 / 8 | ✅ |
| 사용자 메모리 가드레일 4종 | 4 / 4 | ✅ |
| Framework core invariant (D4) | 1 / 1 | ✅ |
| Step 1-5 완료 | 5 / 5 | ✅ |
| pytest sanity | 11 passed / 0 failed | ✅ |
| **Total** | **33 / 33** | **✅** |

→ **PUSH 준비 완료** (origin only).
