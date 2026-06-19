# easy8-qna — easy8 후속 질문 모음

> [easy8.md](easy8.md) / [easy8-detail.md](easy8-detail.md) 를 보다 나온 실제 질문들을 정리한 노트.
> ④ flash attention 의 softmax 교체 가능성, ③ `mmt_block_scaled_q8` 의 CustomOp 동작 원리 두 갈래.
> 소스 근거: `/home/bohyun/amd-shark-ai/amdsharktank/amdsharktank/` (읽기 전용) + iree.turbine `op_reg`.

---

## Q1. ④ flash attention 에서 softmax 대신 경량화를 위해 LayerNorm 을 쓰면 안 되나?

결론부터: **그대로 대체는 안 된다.** softmax 와 LayerNorm 은 "둘 다 정규화"라는 점만 같을 뿐, *무엇을 어느 축으로 정규화하는지*와 *출력의 의미*가 완전히 다르다. 게다가 우리 lowering 맥락에선 softmax 가 NPU 를 막는 원인이 아니라서, 바꿔도 경량화 효과가 사실상 없다.

### 둘은 같은 자리에 들어갈 연산이 아니다

attention 의 `softmax(QKᵀ·scale) @ V` 에서 softmax 가 하는 일은 "이 query 가 각 key 에 **얼마씩 주목할지**"를 정하는 것이다. 핵심은 출력이 **확률분포**라는 점.

| | Softmax (attention 안) | LayerNorm |
|---|---|---|
| 정규화 축 | **key 축** (K2, 시퀀스 길이) — query 마다 key 들에 대해 | **feature 축** (고정 길이) |
| 출력 성질 | 전부 ≥ 0, **합 = 1** | 부호 있음(음수 가능), 합 ≠ 1 |
| 의미 | V 들의 **볼록결합**(가중평균) | 벡터 표준화 후 γ·z+β |
| 학습 파라미터 | 없음 | γ, β 있음 |

softmax 를 LayerNorm 으로 바꾸면 "attention 가중치"가 음수도 되고 합도 1이 아니게 된다. 그러면 `weights @ V` 가 더 이상 V 들의 가중평균이 아니라 **아무 선형결합**이 되어, attention 을 작동하게 만든 핵심 inductive bias(볼록결합)가 깨진다. 같은 모델이 아니라 *다른 연산*이 되는 것이고, 당연히 재학습해야 하며 보통 불안정해진다.

특히 **마스킹**에서 바로 티가 난다. ④ 에 `masked_flash_attention` 이 따로 있는데, softmax 는 마스크 위치에 `-inf` 를 넣으면 `exp(-inf)=0` 으로 **깔끔하게 0 기여**가 된다. LayerNorm 은 평균·분산을 그 길이 전체로 계산하므로 마스크 위치를 빼는 것도 어색하고 0 기여를 보장할 방법이 없다 — causal LM 에선 치명적.

### 우리 맥락에선 softmax 가 "막는 원인"이 아니다

이게 더 중요하다. easy8 에서 ④ 가 문제 커널인 이유는 **softmax 가 무거워서가 아니라**, AMD 가 attention 전체를 `iree_linalg_ext.attention` 이라는 IREE 전용 op 으로 감싸 O(seq²) score matrix 를 안 만들게(fusion) 했기 때문이다. 막는 건 그 **fused flash op / IREE 전용 방언**이지 softmax 자체가 아니다.

on-device 에서 우리가 하려는 건 이 fused op 를 **표준 op 로 분해**하는 것이고, 그러면 softmax 는 그냥 `aten.softmax → aten._softmax` 로 표준 lowering 된다. NPU 가 막지 않는다. 즉 **softmax 를 LayerNorm 으로 바꿔도 lowering 문제는 그대로** — 둘 다 표준 op 로 잘 내려가니까.

### exp() 제거가 진짜 목적이면

목적이 softmax 의 `exp()` 를 없애 더 싼 커널을 쓰는 거라면, 실제 후보는 **LayerNorm 이 아니라** 이쪽이다.

- **ReLU attention** — Google "Replacing softmax with ReLU"(2023, ViT)는 `ReLU(QKᵀ)/seq_len` 으로 softmax 근사. Primer 는 squared ReLU.
- **Linear attention** (Performer, cosFormer, linear transformer) — softmax 를 feature map φ 로 바꿔 `φ(Q)(φ(K)ᵀV)` 로 결합법칙을 써서 O(N²)→O(N). 단 계산 구조 자체가 바뀐다.

이것들은 전부 **재학습이 필요**하고, exp() 는 보통 NPU activation 유닛에 이미 있어서 경량화 이득도 생각보다 작다. softmax(max+exp+sum+div)나 LayerNorm(mean+var+norm+affine)이나 둘 다 O(n) reduction 이라 비용 차이도 크지 않다.

### LayerNorm 이 attention 에서 정당하게 쓰이는 자리

대체가 아니라 **보완**으로는 쓴다.

- **QK-norm**: Q, K 를 내적 전에 정규화(ViT-22B 등 학습 안정화) — softmax 는 그대로 두고 *앞에* 추가.
- **pre/post-LN**: attention 블록 바깥의 표준 LayerNorm.

> 요약: softmax↔LayerNorm 은 축·의미가 달라 그대로 못 바꾸고(마스킹·확률분포·볼록결합이 깨짐), 우리 NPU 문제는 softmax 가 아니라 fused flash op 이라 바꿔도 소용없다. exp() 제거가 목적이면 ReLU/linear attention 이 정답이지만 재학습이 든다.

---

## Q2. ③ `mmt_block_scaled_q8` — CustomOp 는 실제로 어떻게 동작하나

```python
@CustomOp.register(library=LIBRARY)
class mmt_block_scaled_q8(CustomOp):
    """Generic block scaled matmul with transposed RHS.
    * d:  [N, K // 32, 1]      (per-block scale)
    * qs: [N, K // 32, 32]     (int8 weight)"""
    signature = "mmt_block_scaled_q8(Tensor a, Tensor d, Tensor qs) -> (Tensor)"
```

핵심 사실 하나부터. `CustomOp` 은 amdsharktank 게 아니라 **iree.turbine** (shark 런타임) 인프라다(kernels/base.py:29 에서 `from iree.turbine.runtime.op_reg import CustomOp`). 그리고 이 클래스의 Python 은 **런타임 커널이 아니라 "컴파일 시점에 MLIR 을 만들어내는 메타코드"** 다. 이 구분이 아래 6문답 전부의 답을 가른다.

### 먼저: 한 op 에 "두 종류"가 들어있다

```
mmt_block_scaled_q8 클래스
├─ signature / select  ← Python, 트레이스·컴파일 시점 실행 (shape·dtype 계산, 특수화)
├─ generate            ← Python, lowering 시점 실행 (.mlir 템플릿을 채워 emit)
└─ templates/mmt_block_scaled_q8_3d.mlir  ← 진짜 계산(마이크로커널). IREE 가 컴파일해 device 에서 돈다
```

`select`/`generate` 의 Python 은 **숫자를 계산하지 않는다.** 숫자를 계산하는 건 `.mlir` 이고, Python 은 "어떤 MLIR 을 어떤 모양으로 찍어낼지"만 정한다. (turbine op_reg/base.py 의 `generate` docstring: *"This method should generate IR into the given KernelBuilder"* — IR 을 만드는 거지 실행하는 게 아님.)

### 1) 커스텀 커널에서 Python 코드를 그대로 쓸 수 있나?

**부분적으로만.** `select`/`generate` 안의 Python 은 그대로 쓰이지만 그건 **컴파일 시점 오케스트레이션**이다. device 에서 도는 커널 본체는 반드시 MLIR 이어야 한다. 임의의 Python 수식을 커널 몸통으로 넣어 NPU/GPU 에서 돌릴 수는 없다.

예외가 하나 — `eager_execute(*args)` (op_reg/base.py:221). 여기에 순수 Python 구현을 넣으면 **eager 모드에서만** Python 으로 돌 수 있다. 하지만 (a) AOT/export 모드에선 호출 안 되고, (b) `mmt_block_scaled_q8` 은 이걸 override 안 했다(기본 `return NotImplemented`). 그래서 이 커널은 eager 에서도 Python 이 아니라 **MLIR 을 즉석 컴파일해서** 돈다.

### 2) 나중에 커스텀할 때 저 Class 만 바꾸면 되나?

바꾸려는 게 뭐냐에 따라 다르다.

| 바꾸려는 것 | 손볼 곳 |
|---|---|
| 계산 알고리즘 (같은 입출력 shape) | **`.mlir` 템플릿만** (class 는 템플릿 이름·Jinja 변수만) |
| 입출력 signature·shape·특수화 | `signature` + `select` + `generate` (class) |
| 어떤 모델이 이걸 쓰게 | 디스패치 배선(`@matmul.override`) 또는 명시적 호출 |

class 는 "인터페이스 + shape 계약 + 어떤 MLIR 찍을지"이고, **실제 무거운 수학은 `.mlir` 에 있다.** class 만 바꿔선 계산 내용이 안 바뀌고, class 를 안 바꾸면 op 등록·shape 추론이 없어 트레이스가 안 된다. 보통 둘 다 건드리되 비중은 `.mlir` 이 크다.

### 3) 그대로 갖다 쓰는 사례가 있나?

**amdsharktank 자기 자신이 이미 그렇게 쓴다.** ops/custom_impls.py:77-94 에서:

```python
@matmul.override(Tensor, QuantizedTensor, impl_name="amdsharktank")
def matmul_generic_tensor_block_scaled(lhs, rhs, *, transpose_rhs):
    ...
    rhs_unpacked = rhs.unpack()
    return mmt_block_scaled_q8(lhs, rhs_unpacked.d, rhs_unpacked.qs)   # ← 그대로 호출
```

모델이 `ops.matmul(x, weight)` 를 부를 때 weight 가 `BlockScaledLayout`(int8 블록 양자화)이면 **자동으로** 이 구현으로 디스패치되어 호출된다. "갖다 쓰기" = (a) weight 를 그 레이아웃으로 두고 (b) `ops.matmul`/linear 를 부르면 끝.

### 4) CustomOp 가 어느 단에서 made 되나?

세 단계로 나눠 일어난다 (op_reg/base.py:100-168).

1. **import 시점 — 정의/등록**: `@CustomOp.register(library=LIBRARY)` 실행 →
   - torch.library 에 op 정의 → `torch.ops.amdsharktank.mmt_block_scaled_q8` 생성
   - `register_meta=True` 로 **Meta impl**(트레이스용 shape 추론) 등록
   - 실제 디스패치 키(CPU/CUDA)에 **impl 트램펄린** 등록
   - 데코레이터가 **클래스를 호출 가능한 op 으로 치환** (`return instance.op`, :138) — 그래서 `mmt_block_scaled_q8` 은 이제 클래스가 아니라 호출가능 함수
2. **호출(trace/meta) 시점 — 선택**: 매 호출마다 `select()` 실행 → 인자 shape 에서 결과 shape/dtype 산출, 특수화 결정
3. **lowering/실행 시점 — 생성**: `generate()` 가 `.mlir` 을 KernelBuilder 에 emit. eager 면 standalone 커널로 IREE JIT 컴파일·실행, AOT export 면 `util.func` 로 모듈에 splice

### 5) `.mlir` 만 바꾸면 되나? 등록하고 Python class 쓰면 되나?

**계약(입출력 shape/signature)이 그대로면 → `.mlir` 만 바꾸면 된다.** `generate` 가 이미 "템플릿 → call_function" 배선을 해두므로, class 는 템플릿 이름·Jinja 치환값만 맞으면 된다.

다만 class 는 **반드시 있어야** 한다 — op 을 torch.library 에 등록하고(없으면 `torch.ops.*` 가 안 생김), `select` 로 트레이스용 shape 를 계산하니까. 정리하면 `.mlir` = 계산 본체(여길 바꿔 새 마이크로커널), class = 등록 + shape 계약 + 어떤 템플릿 쓸지(유지/조정).

### 6) 함수 정의해놓고 호출만 하면 되나? 모델 호출이 실제로 일어나나?

**네, 실제로 일어난다.** 등록 후엔 `mmt_block_scaled_q8(a, d, qs)` 처럼 평범한 함수로 부르면 된다(데코레이터가 callable op 으로 치환했으니).

- **eager(PyTorch 런타임)**: `eager_execute` 가 기본 `NotImplemented` 라서 → `generate` 가 만든 **MLIR 마이크로커널을 IREE 로 즉석 컴파일해 진짜로 실행**한다. 계산이 실제로 돈다.
- **export(AOT)**: eager 실행 대신, 트레이스엔 op 노드로 남고 lowering 때 그 MLIR 이 모듈에 splice 된다.
- 모델 코드에선 보통 직접 안 부르고 `ops.matmul` 이 weight 타입을 보고 **자동 디스패치**로 부른다(3번).

### 한 줄 요약 + 우리 NPU 로의 함의

> Python class = **op 등록 + shape 계약 + 어떤 MLIR 을 찍을지**(컴파일 시점). `.mlir` = **실제 계산**(IREE 가 컴파일해 device 에서 실행). 계산만 바꾸려면 `.mlir`, 인터페이스/모양 바꾸려면 class. 모델에선 `ops.matmul` 자동 디스패치로 실제 호출이 일어나고, eager 든 export 든 MLIR 이 진짜로 컴파일·실행/splice 된다.

우리 NPU 로 가져갈 때 핵심도 여기서 나온다 — IREE 마이크로커널(`.mlir`)을 그대로 못 받으면, **class 의 등록·shape 계약은 재사용하되 `.mlir` 을 우리 backend 가 받는 형태(표준 linalg 나 NPU intrinsic)로 갈아끼우는** 게 일이다.

---

## 더 볼 것

- [easy8.md](easy8.md) — 4커널 우회 왜·어떻게 요약 · [easy8-detail.md](easy8-detail.md) — 소스 정독 + fusion/AOTInductor 그림
- ③ q8 dataflow 그림: [../diagrams/kernel-bypass.drawio](../diagrams/kernel-bypass.drawio) Page 2
- amdsharktank: kernels/mmt_block_scaled_q8.py · kernels/templates/mmt_block_scaled_q8_3d.mlir · ops/custom_impls.py · iree.turbine runtime/op_reg/base.py
