# Llama-3.2-1B Step 5 — 결과 시각화

> 2026-05-27 결과의 시각 자료. GitHub 에서 직접 렌더링 (Mermaid),
> draw.io 에서는 *Extras → Edit Diagram → Mermaid* 또는 *Insert → Advanced → Mermaid*
> 로 같은 코드 붙여넣기 가능. 별도 `.drawio` XML 은 [`llama-step5.drawio`](llama-step5.drawio).

---

## 1. PDF 의 

```mermaid
flowchart TD
    classDef done fill:#bbf7d0,stroke:#16a34a,stroke-width:2px,color:#052e16
    classDef purple fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#2e1065
    classDef next fill:#f1f5f9,stroke:#94a3b8,stroke-width:1px,color:#334155,stroke-dasharray: 5 3

    subgraph PURPLE ["━━ 
        direction TB
        PY["PyTorch Model Zoo<br/><b>LlamaOnDevice (171 LoC)</b><br/>RMSNorm · GQA · SwiGLU · RoPE<br/>16-layer FP32 + HF gated weights"]:::purple
        TM["Torch-MLIR / iree.turbine.aot.export<br/>FxProgramsBuilder + aot.export<br/>(AMD-SHARK-AI export pipeline)"]:::purple
        MLIR["<b>Top-level Torch dialect MLIR</b><br/>12 GB · 6 218 lines<br/>server_side_op_hits = ∅<br/>has_dynamic_dim = false"]:::done
        COMP["Torch-MLIR Compiler<br/><b>iree-compile --iree-hal-target-backends=llvm-cpu</b><br/>36.86 s · peak RSS 18 GB"]:::purple
        VMFB["<b>.vmfb (IREE bytecode)</b><br/>6.0 GB · Zip v4.5 · exit 0<br/>✅ 설계 §3 유의미한 결과 충족"]:::done
        PY --> TM --> MLIR --> COMP --> VMFB
    end

    subgraph NEXT ["다음 turn — IREE Runtime / target NPU stack (외부 (compiler stack) 측)"]
        direction TB
        IREE_C["IREE Compiler<br/>(compiler/api.h)"]:::next
        IREE_R["IREE Runtime<br/>(runtime/api.h)"]:::next
        IREE_HAL["IREE HAL Driver<br/>(hal/drivers/halapi.h)"]:::next
        IREE_VM["IREE VM<br/>(vm/api.h)"]:::next
        TGT_LIB["target NPU Library"]:::next
        TGT_RT["target NPU Runtime"]:::next
        TGT_DRV["target NPU Driver"]:::next
        TGT_HW["target NPU HW"]:::next
        IREE_C --> IREE_R
        IREE_R --> IREE_HAL
        IREE_R --> IREE_VM
        IREE_HAL --> TGT_LIB --> TGT_RT --> TGT_DRV --> TGT_HW
    end

    VMFB -."실행 검증<br/>(다음 turn)".-> IREE_R
```

---

## 2. Pipeline data flow + 정량 결과

```mermaid
flowchart LR
    classDef input fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef proc fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef out fill:#bbf7d0,stroke:#16a34a,color:#052e16
    classDef warn fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    M0["HF safetensors<br/>~2.5 GB"]:::input
    M1["PyTorch nn.Module<br/>load_hf_weights<br/>~4 GB peak"]:::proc
    M2["MLIR text<br/>12 GB · 6 218 lines"]:::out
    M3[".vmfb<br/>6.0 GB"]:::out
    B["⚠ Budget 2 GB 위반<br/>peak RAM 12.3 GB<br/>→ Step 6 Q8_0 정당화 근거"]:::warn

    M0 -- "from_pretrained" --> M1
    M1 -- "iree.turbine.aot.export<br/>45.9 s · peak 12.3 GB" --> M2
    M2 -- "iree-compile llvm-cpu<br/>36.86 s · peak 18 GB" --> M3
    M1 -.측정.-> B
```

---

## 3. 본 turn Step A-D 타임라인

```mermaid
flowchart LR
    classDef ok fill:#bbf7d0,stroke:#16a34a,color:#052e16
    classDef done fill:#dcfce7,stroke:#15803d,color:#14532d

    A["<b>Step A</b><br/>환경 검증<br/>venv-shark · iree.turbine · HF cache<br/>&lt; 5 min"]:::ok
    B["<b>Step B</b><br/>real Llama export<br/>tokenize → load_model → export → analyze<br/>~ 2 min"]:::ok
    C["<b>Step C</b><br/>iree-compile<br/>--iree-hal-target-backends=llvm-cpu<br/>36.86 s"]:::ok
    D["<b>Step D</b><br/>(내부 문서) + roadmap Gap G'<br/>~ 5 min"]:::ok
    P["<b>Push</b><br/>commit 0d62acc<br/>origin/torch-mlir-zoo (신규)"]:::done

    A -- "✅" --> B -- "✅ 6218 lines<br/>server_side_op_hits = ∅" --> C -- "✅ .vmfb 6 GB" --> D -- "✅" --> P
```

---

## 4. op count 정합성 (16-layer Llama 분해 검증)

```mermaid
flowchart TB
    classDef layer fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef op fill:#fce7f3,stroke:#be185d,color:#831843

    L["16 × Transformer Layer"]:::layer

    subgraph RMS ["RMSNorm × 33"]
        R1["pow × 33"]:::op
        R2["mean × 33"]:::op
        R3["rsqrt × 33"]:::op
    end

    subgraph ATT ["Attention × 16"]
        A1["bmm × 32 (QK^T + AV)"]:::op
        A2["softmax × 16"]:::op
        A3["transpose × 193"]:::op
    end

    subgraph MLP ["SwiGLU × 16"]
        S1["silu × 16"]:::op
        S2["mm × 113 (gate/up/down + Q/K/V/O)"]:::op
    end

    L --> RMS
    L --> ATT
    L --> MLP
```

> **정합 검증**: 16 layer × 2 RMSNorm + final = **33** ✓, 16 layer × 2 bmm = **32** ✓,
> 16 layer × softmax/silu = **16** 각각 ✓ → PyTorch 정의가 ATen op 로 1:1 lower, 누락 0.

---

## 5. Budget enforce — embedded target 미충족 (정상 신호)

```mermaid
xychart-beta
    title "Stage 별 RSS 메모리 (MB) vs Budget 2048 MB"
    x-axis ["tokenize", "load_model", "export", "analyze"]
    y-axis "RSS (MB)" 0 --> 14000
    bar [6414, 6414, 12282, 12295]
    line [2048, 2048, 2048, 2048]
```

> 모든 stage 에서 RSS > budget. peak 12.3 GB → DRAM 2 GB embedded target 못 들어감.
> **이게 Step 6 (Q8_0 GGUF, Whisper-tiny-INT8) 양자화의 정당화 근거** —
> *사용자 메모리 "양자화는 MLIR lowering 후 마지막" 원칙 부합*.

---

## 6. AMD-SHARK-AI 통합 범위 (사용자 가드레일)

```mermaid
flowchart LR
    classDef used fill:#bbf7d0,stroke:#16a34a,color:#052e16
    classDef notused fill:#fecaca,stroke:#dc2626,color:#7f1d1d,stroke-dasharray: 5 3

    subgraph AMD ["AMD-SHARK-AI (nod-ai/amd-shark-ai)"]
        T["amdsharktank.models.llm.llm<br/>PagedLlmModelV1<br/>ThetaLayer / Dataset / Theta"]:::notused
        E["iree.turbine.aot<br/>FxProgramsBuilder<br/>@export_program<br/>aot.export"]:::used
    end

    subgraph OURS ["본 zoo (torch-mlir-zoo)"]
        OWN["src/torch_mlir_zoo/models/llama_on_device.py<br/>(171 LoC, 직접 작성)"]:::used
        EXP["src/torch_mlir_zoo/exporters/iree_turbine_export.py"]:::used
    end

    E --> EXP
    T -. "안 씀<br/>(사용자 메모리 가드)" .-> OWN
```

> **사용자 메모리 가드**: *"amd-shark 의 transformer 안 씀, export pipeline 만 활용"*.
> transformer 구현은 자체 작성 (171 LoC), export pipeline 만 차용.

---

## 7. Files changed (이번 commit `0d62acc`)

```mermaid
flowchart TD
    classDef new fill:#bbf7d0,stroke:#16a34a,color:#052e16
    classDef mod fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef ignored fill:#e5e7eb,stroke:#6b7280,color:#374151,stroke-dasharray: 5 3

    C["commit 0d62acc<br/>feat(llama-step5): real Llama-3.2-1B → iree.turbine → .vmfb"]

    F1["configs/zoo/llama_on_device_iree_turbine.yaml<br/>(신규 · 23 LoC)"]:::new
    F2["scripts/run_llama_export.py<br/>(--config argparse · +7 LoC)"]:::mod
    F3["(내부 문서)<br/>(Gap G' 섹션 · +9 LoC)"]:::mod
    F4["(내부 문서)<br/>(신규 · 정량 보고서)"]:::new
    F5["artifacts/*.mlir, *.vmfb, *.summary.json<br/>(.gitignore artifacts/ 로 제외)"]:::ignored

    C --> F1
    C --> F2
    C --> F3
    C --> F4
    C -.untracked.-> F5
```
