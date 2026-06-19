# Recipes — Torch-MLIR Zoo (Lowering 관점)

> 본 문서는 `torch-mlir-zoo` 브랜치의 *lowering* 작업 recipe.
> voice-app 의 [`RECIPES.md`](https://github.com/bhseo98/npu-harness-framework/blob/voice-app/RECIPES.md)
> 가 *application plugin* 추가 recipe 라면, 본 문서는 *PyTorch → MLIR → `.vmfb`*
> end-to-end 의 recipe.
>
> Last updated: **2026-05-27** (commit base: `ecf7efb`)

---

## 0. 사전 준비 (한 번만)

```bash
# 1) 본 repo clone 및 venv (Python 3.11 권장 — iree-turbine 호환)
git clone <origin-url> npu-harness-framework
cd npu-harness-framework
python3.11 -m venv /home/USER/venv-shark
source /home/USER/venv-shark/bin/activate

# 2) 의존성 — [shark] extras 가 iree-base-compiler/runtime + iree-turbine
uv pip install -e .[shark]
# 또는: pip install -e .[shark]

# 3) HuggingFace 인증 (gated 모델용)
huggingface-cli login   # accept meta-llama/Llama-3.2-1B-Instruct on hub
```

검증:
```bash
python -c "import iree.turbine; import iree.compiler; print(iree.turbine.__version__)"
iree-compile --version | head -3
pytest tests/test_zoo_ops.py tests/test_iree_turbine_export.py -v
```

---

## 1. 단위 op 1개 lowering (Step 4 recipe)

### 1.1 새 op 추가

```python
# src/torch_mlir_zoo/ops/myop.py
"""MyOp — 한 줄 설명."""
from __future__ import annotations
import torch
from torch import nn


class MyOp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ATen op 만 사용 — custom CUDA / paged / flash attn 금지
        return torch.softmax(x, dim=-1)
```

### 1.2 stages.py 의 loader 에 등록

```python
# src/torch_mlir_zoo/stages.py 의 ModelLoader 안
elif self.type == "myop":
    from .ops.myop import MyOp
    return MyOp(dim=self.config.get("dim", 128))
```

### 1.3 config 두 개 (sibling pair — backend swap 검증)

```yaml
# configs/zoo/myop.yaml — torch_mlir_dialect path
stages:
  - {name: load_model, stage: loader, config: {type: myop, dim: 128}}
  - {name: export, stage: exporter, config: {type: torch_mlir_dialect, out_path: artifacts/zoo/myop.mlir}}
  - {name: analyze, stage: analyzer, config: {type: ir_summary, out_path: artifacts/zoo/myop.summary.json}}
```

```yaml
# configs/zoo/myop_iree_turbine.yaml — iree.turbine path (sibling)
# 위 yaml 의 export.type 만 iree_turbine 으로 변경 + out_path 분리
```

### 1.4 실행 + 검증

```bash
# torch_mlir_dialect path
python scripts/run_zoo_export.py --op myop
cat artifacts/zoo/myop.summary.json | jq .server_side_op_hits   # == {}

# iree_turbine path
python scripts/run_iree_turbine_export.py --op myop
cat artifacts/zoo-iree/myop.summary.json | jq .server_side_op_hits   # == {}
```

성공 기준:
- `server_side_op_hits == {}` (`paged_attention`, `kv_cache`, `vllm.*`, `flash_attn` 0건)
- `has_dynamic_dim == false` (정적 shape)

---

## 2. End-to-End LLM lowering (Step 5 recipe)

### 2.1 기존 Llama-3.2-1B 재현

```bash
mkdir -p artifacts logs

# (a) torch_mlir_dialect path
HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  python scripts/run_llama_export.py \
    --config configs/zoo/llama_on_device.yaml

# (b) iree.turbine path (권장 — AMD-SHARK-AI export pipeline 사용)
HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  python scripts/run_llama_export.py \
    --config configs/zoo/llama_on_device_iree_turbine.yaml
```

예상 산출 (path (b) 기준):
- `artifacts/llama-3.2-1b-on-device-iree-turbine.mlir` (12 GB · 6 218 lines)
- `artifacts/llama-3.2-1b-on-device-iree-turbine.summary.json` (server_side_op_hits = `{}`)

### 2.2 IREE compile to `.vmfb`

```bash
iree-compile artifacts/llama-3.2-1b-on-device-iree-turbine.mlir \
    --iree-hal-target-backends=llvm-cpu \
    -o artifacts/llama-3.2-1b-on-device-iree-turbine.vmfb \
    2>&1 | tee artifacts/iree-compile.log
```

예상: 6.0 GB · ~37 s · exit 0.

**Budget warning 정상 신호**: peak RAM ~12 GB ≫ 2 GB → embedded target 미충족.
Step 6 의 Q8_0 양자화 정당화 근거로 보고서에 인용 (의도된 측정).

### 2.3 새 LLM architecture 추가

자체 nn.Module 작성 + `load_hf_weights` 같은 어댑터:

```python
# src/torch_mlir_zoo/models/myllm_on_device.py
import torch
from torch import nn


class MyLlmOnDevice(nn.Module):
    """on-device 친화 — paged_attention, kv_cache, vllm 종속 0."""
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        # nn.Linear / RMSNorm / SwiGLU / RoPE (precomputed cos/sin) 만 사용
        ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...


def load_hf_weights(model, hf_id: str, token: str | None = None) -> None:
    """HF 상태 dict → 자체 모델 weights mapping (prefix 정리)."""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(hf_id, token=token)
    state = hf.state_dict()
    # model.* prefix 제거 등 mapping
    ...
    model.load_state_dict(state, strict=True)
```

stages.py 의 ModelLoader 에 등록 후 `configs/zoo/myllm_on_device.yaml` +
`..._iree_turbine.yaml` sibling pair 작성.

---

## 3. Backend 선택 — torch_mlir_dialect vs iree_turbine

| 측면 | `torch_mlir_dialect` | `iree_turbine` |
|---|---|---|
| 호출 함수 | `torch_mlir.compile(model, args, output_type=TORCH)` | `iree.turbine.aot.export(FxProgramsBuilder, ...)` |
| 의존 | `torch-mlir` (LLVM project) | `iree-turbine==3.10.*` + `iree-base-{compiler,runtime}==3.11.*` |
| 결과 dialect | top-level torch dialect (순수) | torch.aten.* + torch_c.from_builtin_tensor + util.global.* + func.func 혼합 |
| weights | 별도 미포함 (signature 만) | tensor literal 로 IR 안 inline (큰 모델 시 IR 거대화) |
| `.vmfb` 직결 | 추가 step 필요 | `iree-compile` 가 그대로 받음 |
| 사용처 | IR 분석 / pretty 출력 | downstream `iree-compile` + 실행 |

→ **권장**: 단위 op + IR 분석 = `torch_mlir_dialect`. End-to-end `.vmfb` 까지 = `iree_turbine`.

---

## 4. `aot.externalize_module_parameters` — MLIR 슬림화 recipe

큰 모델 (Llama-1B FP32 = 12 GB MLIR) 의 weights 를 IR 밖 `.irpa` 파일로 분리.

```python
# src/torch_mlir_zoo/exporters/iree_turbine_export.py 에 옵션 추가 (미구현 — 다음 turn 후보)
from iree.turbine import aot
aot.externalize_module_parameters(model)   # ← inline → 외부 참조
exported = aot.export(model, args)
exported.save_mlir(out_path)
# 별도로 .irpa (parameter archive) 저장
```

기대 효과: MLIR text 12 GB → 수 MB (weights ref 만 남음) + `.irpa` 별도 ~2-3 GB.
git 관리 가능 크기 + 협업 향상.

---

## 5. Step 6 — Quantization recipe (보류 중 / 정당화 후 시작)

### 5.1 사전 조건 (사용자 메모리 가드)

**"양자화는 MLIR lowering 후 마지막"** — FP32 path 가 작동하고 budget 위반이
*측정*된 후에만 양자화 적용.

✅ 2026-05-27 시점 충족 — Step 5 `.vmfb` 생성, peak RAM 12.3 GB ≫ 2 GB
(`REPORT-2026-05-27-llama-step5-complete.md` §4).

### 5.2 옵션 (a) Q8_0 GGUF (`unsloth/Llama-3.2-1B-Instruct-GGUF`)

```bash
# huggingface-cli download unsloth/Llama-3.2-1B-Instruct-GGUF llama-3.2-1b-q8_0.gguf
# (계획) src/torch_mlir_zoo/models/llama_gguf.py — GGUF 로더 + Q8_0 dequant on-the-fly
# 또는 llama.cpp 와 binding (Phase 3 virtual platform 단계)
```

### 5.3 옵션 (b) Whisper-tiny-INT8 (`rhasspy/faster-whisper-tiny-int8`)

```bash
# (계획) src/torch_mlir_zoo/models/whisper_on_device.py — encoder/decoder PyTorch-only
# CT2 의 INT8 layer 를 ATen op 로 분해
```

둘 다 미진. 진행 시 본 recipe §1, §2 의 절차를 그대로 따름 + Q8 dequant block 추가.

---

## 6. 검증 — 본 recipe 가 동작함을 보이는 명령

```bash
# 단위 op (Step 4)
pytest tests/test_zoo_ops.py tests/test_zoo_export.py tests/test_iree_turbine_export.py -v

# LLM E2E (Step 5)
pytest tests/test_llama_pipeline.py -v
# heavy real-Llama test 는: RUN_HEAVY=1 HF_TOKEN=... pytest tests/test_llama_pipeline.py -v

# Framework core invariant (Design D4)
git diff main HEAD -- src/npu_harness_framework/   # empty

# server_side_op_hits 정량 (모든 export 결과)
for f in artifacts/**/*.summary.json; do
    jq '{file: input_filename, hits: .server_side_op_hits}' "$f"
done
```

---

## 7. 흔한 함정 (gotcha)

| 증상 | 원인 | 해결 |
|---|---|---|
| `OSError: 401` from HF | gated repo accept 미완 | huggingface.co/meta-llama/Llama-3.2-1B-Instruct 에서 access 요청 |
| Export 중 process kill (OOM) | host RAM < 16 GB | TINY (2-layer) 로 먼저 검증, 또는 host RAM 확장 |
| `iree-compile` 가 generic CPU warning | target CPU 미지정 | `--iree-llvmcpu-target-cpu=host` 추가 |
| `pytest` 가 sm_120 warning | RTX PRO 6000 Blackwell 호환 안 됨 | CPU fallback 으로 자동 진행 — 영향 없음 |
| `server_side_op_hits` 가 비어있지 않음 | 모델이 paged_attention / kv_cache 등 사용 | 자체 nn.Module 로 재구현 ([§2.3](#23-새-llm-architecture-추가)) |
| MLIR 텍스트가 GB 단위 | `iree.turbine.aot.export` 의 weights inline 동작 | `aot.externalize_module_parameters` ([§4](#4-aotexternalize_module_parameters--mlir-슬림화-recipe)) |

---

## 8. 다음 recipe 추가 후보

- [ ] `.vmfb` runtime 실행 — IREE Runtime API 호출 + numerical 비교
- [ ] Whisper-tiny PyTorch-only 재구현 + lowering
- [ ] TTS lowering 후보 평가 (Piper / VITS-lite / ESPnet 유지)
- [ ] Qwen-0.5B lowering 추가 — voice-app default LLM 과 정합
- [ ] CPU-only Dockerfile.zoo — `.vmfb` 실행 컨테이너
