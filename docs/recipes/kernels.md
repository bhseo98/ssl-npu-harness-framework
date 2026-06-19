# Recipes — SSLNPU 커널 / INT8 Lowering (Phase A)

> `block_scaled_q8` CustomOp + 마이크로커널을 **직접 저작**하는 단계의 재현 recipe.
> [RECIPES-zoo.md](zoo.md)가 *PyTorch→MLIR→.vmfb* 일반 lowering recipe라면,
> 본 문서는 그 위에서 **whisper-tiny INT8 end-to-end 실행 + Rust 네이티브 커널 + riscv64 prep**의 recipe.
> 설계 배경은 [kernel-design/](../kernel-design/), 개인 학습 노트는 [easy-series/](../easy-series/)(별도 트랙).
>
> 모든 게이트는 **실제 실행 출력으로 검증**(string-match 아님). Last verified: **2026-06-10**.
> 📊 파이프라인 그림: [../diagrams/lowering-pipeline.drawio](../diagrams/lowering-pipeline.drawio)

---

## 0. 사전 준비

```bash
cd /home/bohyun/npu-harness-framework
source /home/bohyun/venv-shark/bin/activate          # iree-turbine + iree-base-{compiler,runtime}
python -c "import iree.turbine, iree.compiler; print('iree ok')"

# Rust (RK1/RK2) — 유저로컬 rustup, 시스템 오염 금지
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y   # ~/.cargo
source ~/.cargo/env && rustc --version
```

---

## 1. WM3 — whisper-tiny INT8 *전체*를 IREE-CPU에서 실행

INT8 양자화한 whisper-tiny 전체를 vmfb로 컴파일하고 IREE-CPU로 단일 forward 실행, FP32와 수치 비교.

```bash
# (a) 프레임워크 lowering — MLIR + op 요약 덤프 (WM2)
python scripts/run_whisper_int8_export.py
#   → logs/whisper-int8/whisper-tiny-int8.mlir
#   → logs/whisper-int8/whisper-tiny-int8.summary.json

# (b) vmfb 컴파일 + IREE-CPU 실행 + FP32 대조 (WM3)
python scripts/verify_whisper_int8_iree.py
```

**성공 기준(실측):**

| 게이트 | 기대값 | 근거 |
|---|---|---|
| 불투명 op 0 | `server_side_op_hits == {}` | `summary.json` (paged_attention/kv_cache/flash_attn 0건) |
| custom call splice | 65 weight matmul → `block_scaled_q8` 4 unique 커널(shape별) | 384×384 / 1536×384 / 384×1536 / 51865×384 |
| IREE-CPU 실행 | 전체 vmfb 컴파일·실행 성공 | WM3 |
| 정확도 | logits **rel < 5%**(실측 0.8%), argmax 일치 ≈ **100%** | INT8 vs FP32 |

> 양자화 범위 = `nn.Linear` 한정(`quantize_linears_`). 임베딩 테이블(`51865×384` f32 ≈ 80MB)은
> 잔존 → vmfb 크기를 지배. weight-tying(proj_out↔embed_tokens)은 proj_out 양자화로 분리됨(동작 무해, 메모리 중복).

---

## 2. INT8 vs FP32 — 정적/지연 비교

```bash
/usr/bin/time -v python scripts/bench_whisper_int8_vs_fp32.py --variant fp32 --iters 20
/usr/bin/time -v python scripts/bench_whisper_int8_vs_fp32.py --variant int8 --iters 20
```

**실측(llvm-cpu, 단일 forward = mel 30s + 디코더 4토큰):**

| 지표 | FP32 | INT8 | 차이 |
|---|---|---|---|
| vmfb 크기 | 150.5 MB | **125.4 MB** | **−17%** |
| forward 지연(x86 CPU 중앙값) | 25.9 s | 27.6 s | ≈ 동일 |
| 정확도 | 기준 | rel 0.8% / argmax 100% | 무손실 |

해석: −17%뿐인 이유는 80MB f32 임베딩 잔존 + weight-tying 분리. **임베딩 INT8 + re-tie 시 ~45MB(≈3×)**.
CPU는 dequant 후 f32 matmul이라 속도 이득 없음(int8 MAC 이득은 NPU). 상세 [kernel-design/03 §5](../kernel-design/03-kernel-survey-and-memory.md#5-실측--whisper-tiny-fp32-vs-int8-iree-cpu).

---

## 3. RK1 — Rust 네이티브 block-scaled INT8 matmul 커널

`select` 계약과 **동일 레이아웃**(`a[B,M,K]` f32, `qs[N,K/BS,BS]` i8, `d[N,K/BS,1]` f32 → `out[B,M,N]` f32)으로
dequant을 matmul 루프에 fuse(weight i8 유지). 의존성 0(naive scalar), `panic=abort`+`lto`+`strip`.

```bash
# 레퍼런스 fixture(torch) 덤프 → tests/fixtures/
python scripts/dump_block_scaled_q8_fixture.py

# 패리티 테스트
cd rust/block_scaled_q8 && source ~/.cargo/env
cargo test --release          # parity.rs
cargo build --release         # → target/release (self-contained 바이너리)
```

**성공 기준:** Rust 커널 == torch 레퍼런스 **max abs err < 1e-4**(실측 rel **3.5e-7**).

---

## 4. RK2 — 커널 풋프린트/지연: IREE vmfb vs Rust 네이티브

```bash
python scripts/bench_kernel_iree_vs_rust.py     # → logs/bench-kernel/summary.{json,md}
```

동일 4 shape(BS=32, M=4, f32). IREE = native `iree-benchmark-module`(local-task), Rust = self-contained 바이너리.
peak RSS는 `/usr/bin/time -v`.

| shape N×K | IREE ms / RSS(MiB) / vmfb(KiB) | Rust ms / RSS(MiB) |
|---|---|---|
| 384×384 | 0.49 / 7.4 / 173 | 0.36 / **1.6** |
| 1536×384 | 1.43 / 7.7 / 659 | 2.44 / **2.1** |
| 384×1536 | 1.42 / 8.9 / 659 | 1.40 / **2.6** |
| 51865×384 | 48.5 / 70.9 / 21891 | 49.4 / **23.6** |

**결론(1줄):** Rust 네이티브 커널은 peak RSS **3–4.6× 작고** 지연은 비등 — IREE HAL/VM/threadpool/driver-registration을
안 지고 가는 게 이득의 원천(언어 차이 아님). 온디바이스/임베디드엔 Rust 경로가 가볍다.
artifact: IREE 런타임 1214 KiB + vmfb vs Rust self-contained **344 KiB**.

> caveat: Rust는 naive scalar single-thread, IREE는 vectorized/threaded — 지연은 IREE 유리, 풋프린트는 Rust 유리.
> 두 경로 수치 일치(Rust↔torch 3.5e-7, IREE↔torch 7.6e-6) = 같은 커널.

---

## 5. VP0 (prep) — riscv64 vmfb 컴파일 가능 여부

```bash
python scripts/probe_riscv64_compile.py
#   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu (hard-float lp64d)
```

**성공 기준(실측):** 단일 커널 → riscv64 vmfb **178KB**, whisper INT8 전체 → riscv64 vmfb **125MB** 생성 성공.
(※ iree-compile는 riscv64 *컴파일* 가능. riscv64 IREE *런타임*은 미빌드 — VP 단계에서 크로스빌드. [RECIPES-vp.md](vp.md))

---

## 6. 마이크로커널 백엔드 swap (지금 → 나중)

`block_scaled_q8.py`의 단일 이음새:

```python
MICROKERNEL_BACKEND = "standard"   # → "sslnpu" (IR 인계 후)
# generate(): template_name = f"block_scaled_q8_{MICROKERNEL_BACKEND}"
```

`select`(shape 계약)·dispatch·모델 클래스는 백엔드 무관 → 지금 전부 검증 가능.
**마이크로커널 본체(.mlir)만** SSLNPU Runtime/Compiler 인계 후 `templates/block_scaled_q8_sslnpu.mlir`로 채우고 상수 1줄 swap.
설계 근거 [kernel-design/02](../kernel-design/02-lowering-layer.md).

---

## 7. gotcha

| 증상 | 원인 | 해결 |
|---|---|---|
| `server_side_op_hits` 비어있지 않음 | 모델이 paged/kv_cache 사용 | eager attn 강제(`attn_implementation="eager"`) / 자체 nn.Module |
| INT8 vmfb가 FP32 대비 17%만 작음 | 임베딩 f32 잔존 + tying 분리 | 임베딩 INT8 + re-tie(다음 마일스톤) |
| riscv64 vmfb는 되는데 실행 불가 | riscv64 IREE 런타임 미빌드 | VP 단계 크로스빌드 |
| `cargo`가 시스템에 없음 | rustup 미설치 | `~/.cargo` 유저로컬 설치(§0) |
