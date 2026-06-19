# 커널 전수 점검 · whisper lowering 제약 · 메모리 최적화

> [01](01-guideline-and-plan.md)(가이드라인) · [02](02-lowering-layer.md)(lowering 계층)에 이어, 우리가 직접 저작한 `block_scaled_q8` *외의* 커널들을 whisper-tiny INT8 lowering 실측을 근거로 전수 점검하고, whisper lowering 제약을 정리하고, 시스템 관점의 메모리 최적화 기법과 기대효과를 모은다.
> 근거 산출물: `logs/whisper-int8/whisper-tiny-int8.mlir`(WM2/WM3), `src/torch_mlir_zoo/{ops,models,kernels}/`, amdsharktank 커널 우회 분석([../easy-series/easy8.md](../easy-series/easy8.md)).

---

## 0. 한눈에 (TL;DR)

- **막힌 커널은 없다.** whisper INT8 전체가 clean lowering(`server_side_op_hits {}`, 불투명 op 0, `iree_linalg_ext` 0)되고 IREE-CPU에서 실행됨(WM3, rel 0.8%). 65개 weight matmul은 전부 `block_scaled_q8`로 잡힘.
- 따라서 나머지 커널의 "제약"은 **성능/메모리를 흘리는 지점**, "개선"은 **추가 커스텀 커널 후보**다.
- **footprint 범인 2개**: f32 임베딩 테이블 **80MB**(125MB vmfb의 대부분) + attention **O(T²)** 중간텐서.
- **다음에 칠 2개**: ①weight 외부화+mmap / 임베딩 INT8(정적 메모리) ②flash-attention(동적 메모리).

---

## 1. 커널 전수 점검 (block_scaled_q8 외)

이미 다룬 것: weight matmul → `block_scaled_q8` ✅(구현) · RoPE/KV-gather/flash-attn → easy8 분석(미구현) · zoo: SDPA/RMSNorm/SwiGLU/TopK(표준 분해).

whisper-tiny INT8 실측 op 분포 기준, 나머지 커널:

| 커널 | 출현(실측) | 현재 lowering | 제약 | 개선 후보 | 우선 |
|---|---|---|---|---|---|
| **임베딩 테이블** | `embedding` 1, `51865×384xf32` 잔존 | `nn.Embedding` f32 그대로 | **79.6MB f32 = vmfb 125MB의 대부분** (llama면 ~1GB) | **INT8 block-scaled 임베딩 + gather-dequant** → ~4× 축소 | 🔴 |
| **attention score** | `bmm` 24 | `matmul(q,kᵀ)`+softmax+`matmul(attn,v)` | **O(T²) score materialize** (enc T=1500 → head당 ~9MB), activation×activation라 block_scaled_q8 미적용 | **flash-attention**(score 미생성, 타일+online softmax+mask fuse) | 🔴 |
| **LayerNorm / RMSNorm** | `var_mean` 22(LN), llama RMSNorm | sub/rsqrt/mul/add 분해 | 행 위 2–3 pass, fp32 | **fused norm**(1-pass) + 뒤 matmul dequant epilogue 융합 | 🟡 |
| **softmax** | `_softmax` 12 | max/exp/sum/div 분해 | T축 3 pass, 안정화 | flash-attention에 **흡수**(online softmax) | 🟡 |
| **causal mask** | `triu` 1 + masked_fill | `[T,T]` bool materialize | T=1500 → 2.25M bool | flash-attention 타일 내 on-the-fly | 🟡 |
| **RoPE**(llama) | `_apply_rope` chunk+cat | mul/add + `cat` | `cat`가 fusion barrier(easy8) | `rope_select_concat`(generic+select) | 🟡 |
| **GELU/SiLU** | `gelu` 10 / silu | pointwise | 거의 없음(IREE epilogue 융합) | 앞 matmul epilogue 융합 확인/HW LUT | 🟢 |
| **conv1d 프론트엔드** | `convolution` 2 | fp32 conv | 1회·작음, matmul과 다른 커널 패밀리 | SSLNPU에 conv 없으면 im2col→matmul | 🟢 |
| **layout** | `clone` 99·`view` 240·`transpose` 60·`expand` 52 | reshape/copy | clone=DRAM 복사, GQA `repeat_interleave` materialize | layout 전파 / GQA broadcast화(대부분 컴파일러측) | 🟢 |
| **TopK**(sampling) | zoo op | `torch.topk` | vocab 정렬 비쌈 | 별도 sampling 커널(forward 범위 밖) | 🟢 |

---

## 2. whisper lowering 제약 — 이전(easy7) 대비

**새로운 하드 블로커 없음.** 이전 제약이 그대로 재현됐고, INT8이 *추가한* 건 블로커가 아니라 메모리/배포 성격.

**easy7과 동일하게 재현:**
- **불투명 SDPA** — `attn_implementation="eager"`로 강제해 `_softmax`/`bmm`으로 분해(실측 `_softmax` 12, `bmm` 24). sdpa 구현이면 불투명 `aten.scaled_dot_product_attention`이 안 내려감. (핵심 제약, 그대로)
- **RoPE 없음** — whisper는 learned positional embedding이라 cat-barrier 이슈가 애초에 없음(llama와 다름).

**INT8에서 추가 관찰(블로커 아님, 고려사항):**
- **양자화 범위 = `nn.Linear` 한정** — `quantize_linears_`가 `nn.Embedding`·conv는 안 건드림 → 임베딩 80MB f32 잔존. (개선 훅 미구현 성격)
- **weight-tying 깨짐** — proj_out↔embed_tokens 공유가 proj_out 양자화로 분리(proj_out i8, embed f32). 동작 무해, 메모리 중복.
- **forward-only / teacher-forced** — WM3은 단일 forward만 검증. 자기회귀 생성 루프(KV cache·sampling)는 torch.export 트레이싱 불가 → llama on-device와 같은 제약. (실오디오 전사는 stretch)
- **vmfb 125MB** — weight가 상수로 박혀 큰 아티팩트(§3 mmap/양자화로 해결).

**매끄러웠던 점(제약 아님):** `block_scaled_q8`가 65군데 깨끗이 splice, 미지원 op 0, 전체 IREE-CPU 컴파일·실행. **INT8 전환이 lowering 블로커를 추가하지 않았다.**

---

## 3. 메모리 최적화 (시스템 관점)

원칙: **(1) 측정해 큰 덩이부터 → (2) compute↔memory 트레이드오프 의식 → (3) 메모리 계층(SRAM↔DRAM↔storage) 이용.** 정적 메모리는 "양자화+외부화/mmap", 동적 메모리는 "fusion+flash-attn+paged/quant KV"로 친다 — 둘은 독립이라 효과가 곱해진다.

### A. 정적 weight 메모리

| 기법 | 무엇 | 기대효과(우리 수치) | 사례 |
|---|---|---|---|
| **weight 외부화 + mmap** | 가중치를 vmfb에 굽지 말고 `flow.parameter.named` 외부 파일로 → mmap, OS demand-paging | vmfb 125MB→**코드만(~수MB)**, peak RSS=작업셋만, 프로세스 공유, 즉시 기동 | llama.cpp(GGUF mmap) |
| **임베딩 INT8 + re-tie** | `nn.Embedding`을 block-scaled INT8 + proj_out과 가중치 공유 복원 | 80→**20MB**, 중복 proj_out 20MB 제거 → vmfb ~**45MB** | 모든 LLM |
| **INT4 / 혼합정밀** | 둔감층 INT4(Q4_K), 민감층(첫·끝·attn-out) INT8 | matmul weight **~2× 더**, 정확도 손실 최소 | GPTQ/AWQ, llama.cpp Q4_K |
| **엔트로피 압축 weight** | DRAM/스토리지 압축 저장, 로드/타일 시 복원 | 저장 footprint↓ (단 우리 ESC 실측 46%, 분포의존) | C-Transformer implicit weight gen |

> 가장 큰 "공짜" 구조 개선 = **weight를 vmfb에서 빼서 mmap** (정확도 0손실, RSS "상주"→"필요 페이지만"). README의 `dense_resource → flow.parameter.named` 마일스톤이 이것.

### B. 동적 activation / KV 메모리

| 기법 | 무엇 | 기대효과 | 사례 |
|---|---|---|---|
| **flash-attention** | score 미생성, 타일+online softmax | whisper enc(T=1500): head당 9MB→타일 ~0.8MB, **O(T²)→O(T)** → 긴 오디오 컨텍스트 | FlashAttention |
| **KV cache** | decode 시 과거 K/V 저장(재계산 제거) | step당 **O(T²)→O(T)**, 긴 컨텍스트 생성(현재 forward-only는 매 step 전체 재계산) | 모든 추론엔진 |
| **KV 양자화(INT8/4)** | KV cache 저비트 | llama-1B INT8 KV ≈16KB/token → 2GB로 ~128K token(f32는 32K) = **4× 컨텍스트** | KVQuant |
| **paged KV(PagedAttention)** | KV 블록 비연속 할당, 예약낭비 제거 | 같은 예산에 **~2–4× 더 긴 시퀀스**(단편화 0) | vLLM, amdsharktank paged |
| **fusion(중간텐서 회피)** | dequant+matmul, LN+matmul, gelu epilogue 융합 | DRAM 왕복↓, peak 작업셋↓ (우리 커널이 이미 dequant fuse) | 모든 컴파일러 |
| **buffer 재사용 / ping-pong** | 순차 transformer는 레이어 버퍼 2개면 충분 | activation peak **O(전체)→O(1 레이어)** | 정적 할당기 |
| **streaming / chunk** | 오디오 30s 윈도우·토큰 청크 | 입력 길이 무관 메모리 상한 고정 | streaming whisper |

### C. 아키텍처/시스템 수준 (RISC-V VP · SSLNPU)

- **SRAM 타일링(big-little)** — 타일을 온칩 SRAM 유지, DRAM 스트리밍. SSLNPU 존재 이유. → DRAM 대역폭·에너지↓.
- **weight streaming** — RAM보다 큰 모델을 레이어별 스토리지 스트림. → RAM 초과 모델 실행(대역폭 비용).
- **sparsity 활용** — 0 저장·계산 회피(prune). → weight·compute 비례 절감.

---

## 4. 우선순위 (시니어 관점, 우리 케이스)

```
1. 측정              → 이미 함 (임베딩 80MB가 범인)
2. weight mmap 외부화 → 정확도 0손실, RSS 구조적 절감 (먼저)
3. 임베딩 INT8 + re-tie → -60MB (vmfb 125→~45MB)
4. INT4 혼합정밀     → matmul weight 추가 ~2×, 정확도 검증 게이트
5. flash-attention   → activation O(T²)→O(T), 긴 오디오
6. (생성형/llama) paged + 양자화 KV → 4× 컨텍스트
```

정적(2·3·4)과 동적(5·6)은 독립이라 곱해진다 — 임베딩 INT8 + mmap + flash-attn → vmfb 45MB + 낮은 RSS + 긴 컨텍스트 동시.

---

## 5. 실측 — whisper-tiny FP32 vs INT8 (IREE-CPU)

`scripts/bench_whisper_int8_vs_fp32.py` (llvm-cpu, 단일 forward = mel 30s + 디코더 4토큰, `/usr/bin/time -v`):

| 지표 | FP32(기본) | INT8(현재) | 차이 |
|---|---|---|---|
| **vmfb 크기** (정적 footprint) | 150.5 MB | **125.4 MB** | **−17%** |
| **forward 지연** (x86 CPU, 중앙값) | 25.9 s | 27.6 s | ≈ 동일(약간↑) |
| **정확도** (WM3) | 기준 | rel 0.8%, argmax 100% | 사실상 무손실 |

해석(우선순위 §4 정량 확인):

1. **−17%뿐인 이유** — 80MB f32 임베딩이 그대로 + weight-tying 깨져 vocab 가중치 이중 저장(embed f32 80MB + proj_out i8 20MB). **임베딩 INT8 + re-tie 시 125 → ~45MB (~3×)**.
2. **CPU에선 속도 이득 없음** — 커널이 dequant 후 f32 matmul이라 연산량 동일 + dequant 추가. 속도 이득은 int8 MAC 유닛(NPU)에서 발생, CPU에선 메모리 이득만.
3. 26s 지연 자체는 encoder O(T²) attention 지배 → flash-attention(§4-5) 후보.
4. peak RSS는 측정 하네스(PyTorch+컴파일)가 지배해 배포 footprint로 의미 없음 → 배포 RSS는 최소 런타임 + mmap에서 별도 측정.

---

## 6. 참고

- 커널 우회·fusion: [../easy-series/easy8.md](../easy-series/easy8.md) · CustomOp 동작: [../easy-series/easy8-qna.md](../easy-series/easy8-qna.md)
- 우리 커널: [`../../src/torch_mlir_zoo/kernels/block_scaled_q8.py`](../../src/torch_mlir_zoo/kernels/block_scaled_q8.py) · 모델: [`../../src/torch_mlir_zoo/models/llama_on_device.py`](../../src/torch_mlir_zoo/models/llama_on_device.py) · zoo ops: [`../../src/torch_mlir_zoo/ops/`](../../src/torch_mlir_zoo/ops/)
- whisper INT8 lowering: [`../../scripts/run_whisper_int8_export.py`](../../scripts/run_whisper_int8_export.py) · E2E 실행: [`../../scripts/verify_whisper_int8_iree.py`](../../scripts/verify_whisper_int8_iree.py)
- 풋프린트 벤치(IREE vs Rust): [02 §6](02-lowering-layer.md) / `logs/bench-kernel/summary.md`
