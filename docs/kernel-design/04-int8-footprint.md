# INT8 풋프린트 전수조사 · proj_out tie 수정 (125→103MB)

> "INT8 vmfb가 왜 fp32(150MB)의 1/4(~38MB)이 아니라 125MB인가?"에 대한 전수조사와, 그 과정에서 찾은 **broken-tie 중복 저장 버그**의 수정 기록.
> 근거 산출물: `src/torch_mlir_zoo/kernels/quantized_linear.py`(`quantize_linears_`), `logs/whisper-int8/whisper-tiny-int8.mlir`, `scripts/verify_whisper_int8_iree.py`(WM3), `scripts/run_whisper_int8_export.py`.
> 관련: [03-kernel-survey-and-memory.md](03-kernel-survey-and-memory.md)(임베딩 80MB footprint), `logs/bench-kernel/summary.md`(Rust vs IREE RSS).

---

## 0. 한눈에 (TL;DR)

- **수정 결과: INT8 whisper-tiny vmfb 125.4 → 103.0 MB (−22.4MB).** 수치는 오히려 개선(rel 0.8% → 0.61%, argmax 100% 유지).
- **8bit는 여기서 1/4이 아니다 — 이건 정상.** whisper의 최대 weight인 token embedding(vocab 표, 80MB = 전체의 53%)이 `nn.Embedding`이라 *의도적으로* fp32로 남기 때문. 임베딩 보존 시 현실 바닥은 ~103MB.
- **그런데 125MB는 103MB보다도 컸다 — 이건 버그.** `proj_out`(lm_head)이 `embed_tokens`와 weight를 공유(tied)하는데, `quantize_linears_`가 이를 모르고 `proj_out`만 INT8로 양자화 → **같은 vocab 표가 fp32 80MB + INT8 22MB로 두 벌** 저장됐다.
- **수정: tied weight를 가진 Linear는 양자화에서 제외.** vocab 표가 fp32 한 벌로 복원되고, INT8-proj_out이 더 full precision이라 quant error도 줄었다.

---

## 1. 출발 — 비율이 이상하다

| | fp32 | INT8(수정 전) | 비율 |
|---|---|---|---|
| 전체 params | 37,760,640 | — | — |
| weight 저장 | **151.0 MB** | — | — |
| vmfb | ~150.5 MB | **125.4 MB** | **0.83** |

"8bit면 1/4(≈38MB)"이라는 직관 대비 0.83은 너무 크다. weight가 fp32 그대로 박힌 게 아닌지 의심 → 전수조사.

> **MLIR 텍스트 부풀림 아님:** weight는 `dense_resource` 바이너리 blob으로 저장된다(텍스트 `dense<...>` 아님). `.mlir` 파일이 vmfb의 ~2배인 것은 텍스트 형식이 blob을 hex로 인코딩(1B→2자)하기 때문이고, vmfb(103MB)가 실제 weight 바이트다.

---

## 2. 크기 사다리 (전부 실측)

whisper-tiny를 직접 로드해 모듈 타입별로 분해(`block_size=32`, INT8 = `qs` i8 + per-block `d` fp32 = 1.125 B/param):

| 시나리오 | MB | 비율 |
|---|---|---|
| **S0** fp32 (HF 원본, tie dedup) | 151.0 | 1.00 |
| 순수 1/4 (모든 게 int8이라는 *틀린* 가정) | 37.8 | 0.25 |
| **S1** embedding fp32 + 나머지 int8, tie 유지 (= **정상 8bit 기대**) | 103.4 | 0.68 |
| **S2** embedding fp32 + 나머지 int8 + proj_out int8 *별도* (**수정 전 실제**) | 125.9 | 0.83 |
| **S3** embedding까지 int8 + tie 유지 (한 벌) | 46.2 | 0.31 |

두 개의 갭이 핵심:
- `151 → 103` (S0→S1): **정상.** 임베딩(80MB)을 fp32로 둬서 못 줄인 분 — 의도된 손실.
- `103 → 126` (S1→S2): **버그.** proj_out 중복(+22.4MB).

---

## 3. 왜 1/4이 아닌가 — 두 원인

### 원인 ① embedding / positional이 fp32 (의도된 설계)

`quantize_linears_`는 `nn.Linear`만 교체한다. whisper의 최대 weight는 `nn.Embedding`이라 대상이 아니다:

| weight | 모양 | fp32 | 타입 | 양자화? |
|---|---|---|---|---|
| `decoder.embed_tokens` (vocab 표) | 51865×384 | **79.7 MB** | `nn.Embedding` | ❌ fp32 (의도) |
| positional embedding | — | 0.7 MB | `nn.Embedding` | ❌ fp32 (의도) |
| conv1/conv2 (encoder) | — | 2.1 MB | `nn.Conv1d` | ❌ fp32 |

→ vocab 표 하나가 전체의 **53%**. 임베딩을 full precision으로 두는 한 1/4은 구조적으로 불가능하고, 현실 바닥은 **S1 ≈ 103MB**다.

> **block-scaled q8 바닥은 0.25가 아니라 0.28:** int8 32개(32B)마다 fp32 scale 4B가 붙어 1.125 B/param. 여기에 LayerNorm/bias/conv가 fp32로 남아 S3(임베딩까지 양자화)도 0.31이 바닥이다.

### 원인 ② proj_out broken tie (버그)

> 이 broken-tie 자체는 [03 문서](03-kernel-survey-and-memory.md) §5에서 이미 **진단**돼 있었다("weight-tying 깨져 vocab 이중 저장"). 본 문서의 기여는 그 **수정+검증**(아래 §4)이다.

whisper 원본에서 `proj_out`(출력 투영 = lm_head)은 `decoder.embed_tokens`와 **weight를 공유**한다(tie). fp32 모델은 이 vocab 표를 **한 벌(80MB)만** 저장한다.

그런데 `proj_out`은 `nn.Linear`라 `quantize_linears_`가 이를 보고 **자기만의 INT8 복사본을 새로 만든다.** 결과적으로 INT8 모델엔 같은 표가 두 벌:

| | fp32 모델 | INT8(수정 전) |
|---|---|---|
| `embed_tokens` (fp32, 의도) | 80MB ┐ 공유 | 80MB |
| `proj_out` | ┘ (0 추가) | **+22.4MB (INT8 별도)** |
| vocab 표 소계 | **80MB** | **102MB** |

수정 전 `logs/whisper-int8/whisper-tiny-int8.mlir` grep — 두 표가 공존:

```
3 × tensor<51865x384xf32>      ← embed_tokens, fp32 (의도대로)
7 × tensor<51865x12x32xi8>     ← proj_out, INT8 별도 복사본  ← 범인
7 × tensor<51865x12x1xf32>     ← proj_out 블록 스케일
```

의미상으로도 문제다: embedding은 fp32, proj_out은 INT8로 **더 이상 같은 값이 아니게** 된다(tie가 수치적으로도 깨짐).

---

## 4. 수정 — tied Linear는 양자화 제외

`src/torch_mlir_zoo/kernels/quantized_linear.py`의 `quantize_linears_`에, weight가 다른 모듈과 공유된 Linear를 건너뛰는 가드를 추가:

```python
def _tied_weight_ptrs(module):
    # 중복 보존 시 tied weight는 holder 수만큼 등장 → count>1이면 공유
    seen = Counter(p.data_ptr() for _, p in module.named_parameters(remove_duplicate=False))
    return {ptr for ptr, count in seen.items() if count > 1}

# quantize_linears_: 최상위 호출에서 _tied 1회 계산 후 재귀로 전달
if (isinstance(child, nn.Linear)
        and child.in_features % block_size == 0
        and child.weight.data_ptr() not in _tied):   # ← tie면 skip
    setattr(module, name, BlockScaledQ8Linear.from_linear(child, block_size))
```

`proj_out`은 fp32 `nn.Linear`로 남아 `embed_tokens`와의 tie를 유지하고, export 시 vocab 표는 fp32 한 벌로 dedup된다.

### 검증 (실측)

`scripts/verify_whisper_int8_iree.py` (전체 vmfb 컴파일 + IREE-CPU 실행 + fp32 대조):

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| **vmfb 크기** | 125.4 MB | **103.0 MB** (102,986,823 B) |
| `block_scaled_q8` calls / unique kernels | 65 / 4 | **64 / 3** (N=51865 proj 커널 제거) |
| rel error (INT8-IREE vs FP32) | 0.8% | **0.61%** (개선) |
| argmax 일치 (vs FP32) | 100% | **100%** |
| 단위 테스트 `test_block_scaled_q8.py` | 7/7 | **7/7** |

수정 후 MLIR grep — INT8 vocab 복사본이 사라지고 fp32 표만 남음:

```
5 × tensor<51865x384xf32>      ← embed_tokens/proj_out 공유, fp32 한 벌
(tensor<51865x12x32xi8> 없음)  ← INT8 proj 복사본 제거됨
```

`server_side_op_hits {}` · `iree_linalg_ext` 없음 · opaque SDPA 없음 — clean lowering 유지.

---

## 5. 더 줄이려면 (선택지)

| 목표 | 방법 | 결과 |
|---|---|---|
| **임베딩 보존 (현 수정)** | tied Linear skip | **103 MB** (0.68×) — 의도 일치, 본 문서 |
| **최대 압축** | embedding을 INT8 block-scaled gather로 양자화 + tie 유지 | ~46 MB (0.31×) — [03 문서](03-kernel-survey-and-memory.md) §1 🔴 "임베딩 INT8" |

본 수정은 "embedding/positional은 fp32로 둔다"는 설계 의도를 지키는 선에서의 최선(103MB)이다. 그 이하(46MB)는 임베딩 양자화라는 별개 결정이 필요하며, vocab gather-dequant 커널이 선행돼야 한다.

---

## 6. Rust 풋프린트와의 관계 (맥락)

같은 INT8 커널을 Rust 네이티브로 짠 RK2(`logs/bench-kernel/summary.md`)에서 Rust의 peak RSS가 IREE 대비 3–4.6× 작았다. 이유는 본 문서의 vmfb 크기 문제와 같은 뿌리 — **weight를 몇 벌 들고 있느냐**:
- **Rust**: weight를 힙에 정확히 한 벌, dequant을 per-element fuse(f32 weight 미생성) → RSS ≈ weight 1벌 + 수백 KB.
- **IREE**: 고정 런타임 바닥(~7MB) + weight가 mmap 모듈 이미지 + HAL device buffer로 다중 상주 → RSS ≈ weight ~2–3벌.

단, IREE 바닥은 모델 전체에 1회 상각되므로 단일 커널 비교는 Rust에 유리하게 보인다(상세 [03](03-kernel-survey-and-memory.md), 메모리 노트 `project_rust_vs_iree_footprint`).

---

## 재현

```bash
# venv-shark 안에서
python scripts/run_whisper_int8_export.py      # MLIR 재생성 (calls 64 / unique 3 확인)
python scripts/verify_whisper_int8_iree.py     # vmfb 103MB + rel 0.61% + argmax 100%
python -m pytest tests/test_block_scaled_q8.py -q   # 7/7

# vocab 표가 fp32 한 벌인지 (INT8 복사본 없음) 확인
grep -aoE 'tensor<[0-9x]*51865[0-9x]*x(f32|i8)>' logs/whisper-int8/whisper-tiny-int8.mlir | sort | uniq -c
```
