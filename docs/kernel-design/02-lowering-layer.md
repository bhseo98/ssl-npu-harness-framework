# Lowering 계층 설계 — 내 몫의 "컴파일러"

> [01-guideline-and-plan.md](01-guideline-and-plan.md)가 *무엇을·왜*(커널 구성·SSLNPU 적합·호출 함수·class/마이크로커널 분리)였다면, 이 문서는 그중 **"컴파일러" 부분의 구조**를 못박는다. 스택 최상단 점선 박스에서 내가 맡은 "컴파일러"는 SSLNPU 백엔드 컴파일러(runtime팀)도 IREE(기존 OSS)도 아니라 — **CustomOp `select`(shape 계약) → `generate`(emit) → `.mlir` 마이크로커널 splice 로 이어지는 *컴파일 시점 lowering 계층*** 이다.
> 근거 산출물: whisper-tiny INT8 lowering(`scripts/run_whisper_int8_export.py` → `logs/whisper-int8/whisper-tiny-int8.mlir`), 커널 스캐폴드 `src/torch_mlir_zoo/kernels/`.

---

## 0. 한눈에 (TL;DR)

- 내 "컴파일러" = **compile-time lowering 계층**. 숫자를 계산하지 않고, "모델이 부른 커널 호출"을 "IREE가 컴파일할 MLIR(`util.func` + `util.call`)"로 **번역**한다.
- 방출 타깃은 **표준 linalg 마이크로커널(지금, 포터블 — IREE-CPU에서 검증됨)**. SSLNPU IR 인계 후 **SSLNPU intrinsic**으로 바꾸는 건 **이음새 1곳**(`MICROKERNEL_BACKEND` 상수 + 형제 템플릿 파일 1개)뿐이다. `select`(계약)·eager·모델 클래스는 불변.
- 실제 작동 증거: whisper-tiny INT8의 **65개 양자화 Linear 호출이 shape별로 4개 unique 마이크로커널로 dedup**되어 lowering된다 (`iree_linalg_ext` 0, 불투명 SDPA 0, `server_side_op_hits {}`).

---

## 1. "컴파일러(?)" 해소 — 무엇이 내 lowering 계층인가

스택에는 "컴파일러"라 부를 만한 게 여럿이라 헷갈린다. 경계를 못박으면:

| 레이어 | 이게 "내 lowering 계층"인가 | 담당 | 비고 |
|---|---|---|---|
| **CustomOp `generate()` → `.mlir` splice** | ✅ **그렇다 (이 문서의 대상)** | **나** | compile-time 메타코드. 숫자 계산 X, MLIR emit O |
| Torch-MLIR Compiler 박스(스택 점선) | (수정 가능 영역) | 나 | 지금은 iree.turbine 파이프라인을 그대로 사용(torch-mlir 빌드 안 함) |
| IREE Compiler/Runtime/VM | ❌ | 기존 OSS | 내 MLIR을 받아 CPU/타깃으로 컴파일·실행 |
| SSLNPU Library/Runtime 컴파일러 | ❌ | runtime팀 | IREE HAL 아래, 인계 대기 |

> 즉 내가 "설계"하는 컴파일러는 **CustomOp가 어떤 MLIR을 어떤 모양으로 찍을지 정하는 계층**이다. IREE가 그 MLIR을 *받아서* 진짜 코드를 만든다 — 그 아래는 내 몫이 아니다.

---

## 2. Lowering 파이프라인 — 호출에서 MLIR까지

모델 한 번 forward가 lowering 되는 경로 (한 호출 사이트 기준):

```
모델 forward
  └ BlockScaledQ8Linear.forward(x)               (모델 클래스, quantized_linear.py)
      └ quantized_matmul(a, d, qs)               (호출 함수 wrapper)
          └ block_scaled_q8(a, d, qs)            (등록된 CustomOp, torch.ops.sslnpu.*)
─ torch.export 트레이스 ─────────────────────────────────────────
  └ select(ksel)     : 인자 shape → 결과 shape·dtype, K·qs·d 특수화   (호출마다)
  └ generate(ksel,kb): 템플릿을 채워 util.func 정의 + util.call 삽입   (lowering 시)
─ IREE ──────────────────────────────────────────────────────────
  └ util.func @ssl_mmt_block_scaled_q8_* 를 컴파일 (여기부턴 IREE 몫)
```

**핵심: `generate`는 함수 이름을 `fn_name = f"ssl_mmt_block_scaled_q8_{n}_{k}_{bs}_{a_dtype}"`로 만든다.** 같은 `(N, K, BS, dtype)`이면 같은 이름 → 마이크로커널 정의가 **한 번만** 삽입되고 나머지는 `util.call`로 공유된다 (shape 기반 dedup).

### 워크드 예시 — whisper-tiny INT8 (실측, WM2)

`quantize_linears_(whisper_tiny)`로 모든 `nn.Linear`를 INT8 block-scaled로 바꾸고 export하면, **65개 호출 사이트가 4개 unique 마이크로커널 정의로** 접힌다:

| unique 커널 `@ssl_mmt_block_scaled_q8_{n}_{k}_{bs}_{type}` | (N, K) | `util.call` 수 | whisper-tiny에서 무엇 |
|---|---|---|---|
| `…_384_384_32_f32` | 384×384 | 48 | self/cross-attention 의 q·k·v·out 프로젝션 (d_model=384) |
| `…_1536_384_32_f32` | 1536×384 | 8 | MLP fc1 (up, 384→1536) |
| `…_384_1536_32_f32` | 384×1536 | 8 | MLP fc2 (down, 1536→384) |
| `…_51865_384_32_f32` | 51865×384 | 1 | proj_out (vocab head, 384→51865) |
| **합** | | **65 call → 4 def** | 인코더 4 + 디코더 4 layer |

검증 출력:
```
block_scaled_q8 calls: 65 | unique kernels (dedup): 4
iree_linalg_ext: False | opaque SDPA: False
server_side_op_hits: {}
```
→ lowering 계층이 모델 전체를 **포터블 표준 linalg 마이크로커널만으로** 내려보낸다(백엔드 전용 op·불투명 SDPA·서버측 우회 0).

---

## 3. 안정적 계약 = `select` (runtime팀에 넘기는 인터페이스)

`select`는 lowering 계층에서 **유일하게 외부와 약속하는 면**이다. 인자 shape/dtype을 검사·특수화하고 결과 텐서를 정한다 — 그래서 백엔드가 바뀌어도 *이 면은 고정*이고, runtime팀은 이 계약만 보고 그 아래 마이크로커널 본체를 채운다.

```
a:   [B, M, K]        float   (활성, LHS)         K 특수화, B·M 동적
qs:  [N, K//BS, BS]   int8    (양자화 weight)      전 차원 특수화
d:   [N, K//BS, 1]    float   (per-block scale)   전 차원 특수화
out: [B, M, N]        = a @ dequant(qs, d)^T       N 특수화
```

(`src/torch_mlir_zoo/kernels/block_scaled_q8.py:select` — `torch._check`로 `K == qs_group0 * BS`, `d`가 `qs`와 정합인지 강제. 어긋나면 조용한 오산이 아니라 거부.)

---

## 4. Swap 이음새 — 지금(표준 linalg) → 나중(SSLNPU intrinsic)

가이드라인의 "generate 한 곳만 바꾸면 된다"를 **문자 그대로 참인 구조**로 만든 게 이 이음새다.

```python
# block_scaled_q8.py
MICROKERNEL_BACKEND = "standard"          # ← 유일한 swap 지점
...
template_name = f"block_scaled_q8_{MICROKERNEL_BACKEND}"   # → templates/block_scaled_q8_standard.mlir
```

| | swap 시 변하는 것 | 불변 |
|---|---|---|
| 코드 | `MICROKERNEL_BACKEND = "standard"` → `"sslnpu"` (1줄) | `signature` / `select` / `eager_execute` |
| 템플릿 | `templates/block_scaled_q8_sslnpu.mlir` 1개 추가 | `block_scaled_q8_standard.mlir`(그대로 둠) |
| 상위 | — | `BlockScaledQ8Linear`·`quantize_linears_`(모델 클래스), `generate`의 fn_name·shape 추출 로직 |

> 그래서 SSLNPU IR이 와도 **임계경로가 안 늘어난다** — 모델·계약·테스트는 그대로 두고 마이크로커널 본체만 새 템플릿에 채운다.

---

## 5. 마이크로커널 해부 (`block_scaled_q8_standard.mlir`)

`.mlir`이 lowering 계층이 *방출하는* 실제 계산이다. 두 `linalg.generic`으로 구성:

1. **dequant** — `i8 qs` → `extsi`(i32) → `sitofp` → per-block `d` 곱. weight를 `[N, group0, BS]` fp로.
2. **grouped batch matmul** — `(group0, BS)` 2축 reduction, f32 accum. **dequant(1)이 이 타일 루프로 fuse**되어 *fp weight 전체가 DRAM에 적재되지 않는다* — weight는 int8로 머문다(easy8-qna의 핵심: "정확한 dequant이 아니라 fusion이 일감").

`linalg`/`tensor`/`arith`/`util`만 쓰고 `iree_linalg_ext`를 안 써서 IREE-CPU로 포터블하게 내려간다. SSLNPU판은 이 두 generic을 SSLNPU intrinsic으로 바꾸되, **입출력 시그니처(`!a_tensor_type` …)는 `select` 계약을 따라 동일**하다.

---

## 6. 검증 — 지금 green인 게이트 / 다음

| 게이트 | 무엇을 증명 | 상태 |
|---|---|---|
| eager == dequant+matmul (exact) | 등록·계약·수치 (IREE 없이) | ✅ `tests/test_block_scaled_q8.py` |
| export smoke | `generate`가 표준 linalg 마이크로커널 splice (`iree_linalg_ext` 0, srv_hits {}) | ✅ |
| **M1 — IREE-CPU 실행** | 마이크로커널이 *실제로 컴파일+실행*되고 torch 레퍼런스와 일치 (max|iree−eager|≈7.6e-6) | ✅ |
| **WM2 — 모델 lowering** | whisper INT8 전체가 4 unique 커널로 내려감 (§2) | ✅ |
| WM3 — 모델 E2E 실행 | whisper INT8 *전체* vmfb를 IREE-CPU에서 실행, FP32 logits 일치 | ⏭ 다음 마일스톤 |

> lowering 계층의 천장 = **CPU-via-IREE에서 end-to-end 컴파일·실행·수치 검증**. 그 위(실제 SSLNPU HW)는 IREE HAL을 접점으로 runtime팀과 합쳐야 완성된다([01 §1](01-guideline-and-plan.md)).

---

## 7. 그림

```mermaid
flowchart TB
  call["block_scaled_q8(a, d, qs)<br/>(등록된 CustomOp 호출)"] --> sel
  subgraph LOW["lowering 계층 = 내 '컴파일러' (compile-time)"]
    direction TB
    sel["select<br/>shape 계약 · 특수화 (안정적 계약)"] --> gen["generate<br/>fn_name = ssl_mmt_block_scaled_q8_N_K_BS_dtype<br/>→ util.func 정의 + util.call (shape별 dedup)"]
    gen -->|"template = block_scaled_q8_{MICROKERNEL_BACKEND}"| seam{{"swap 이음새"}}
    seam -->|"\"standard\" (지금)"| std["block_scaled_q8_standard.mlir<br/>표준 linalg — 포터블"]
    seam -->|"\"sslnpu\" (인계 후)"| ssl["block_scaled_q8_sslnpu.mlir<br/>SSLNPU intrinsic"]
  end
  std --> iree["IREE (기존 OSS)<br/>CPU 컴파일·실행 ✅"]
  ssl --> hal["IREE HAL → SSLNPU Runtime<br/>(runtime팀, 인계 대기)"]
  classDef mine fill:#e7f6ec,stroke:#2e7d32;
  classDef stable fill:#e3f2fd,stroke:#1565c0;
  classDef wait fill:#fdecea,stroke:#c62828;
  class sel,gen,std mine;
  class seam,iree stable;
  class ssl,hal wait;
```

---

## 8. 참고

- 커널 4-파트·우회 분석: [01-guideline-and-plan.md](01-guideline-and-plan.md) · [../easy-series/easy8-qna.md](../easy-series/easy8-qna.md)
- 코드: [`../../src/torch_mlir_zoo/kernels/block_scaled_q8.py`](../../src/torch_mlir_zoo/kernels/block_scaled_q8.py) · 템플릿 [`templates/block_scaled_q8_standard.mlir`](../../src/torch_mlir_zoo/kernels/templates/block_scaled_q8_standard.mlir) · 모델측 [`quantized_linear.py`](../../src/torch_mlir_zoo/kernels/quantized_linear.py)
- 검증: [`../../tests/test_block_scaled_q8.py`](../../tests/test_block_scaled_q8.py) · WM2 스크립트 [`../../scripts/run_whisper_int8_export.py`](../../scripts/run_whisper_int8_export.py)
