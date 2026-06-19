# Recipes — RISC-V64 Virtual Platform (Deliverable ③)

> CPU-only SoC Virtual Platform(Qbox = QEMU+SystemC) 위에 riscv64 네이티브 엔진을 올리고
> 호스트가 오케스트레이션하는 단계의 재현 recipe. riscv64 *컴파일* prep은 [RECIPES-kernels.md §5](RECIPES-kernels.md#5-vp0-prep--riscv64-vmfb-컴파일-가능-여부).
>
> 접속/레이아웃 구체값(IP·docker id)은 **로컬 메모리 `reference_vp_server4`**에 둔다(여기엔 비밀정보·구체 주소 미기재).
> Last verified: **2026-06-10** (ssl-soc-gen2 부팅 + 게스트 llama-cli 실행).

---

## 0. 원칙

- **엔진 고정 + 모델 볼륨 + 호스트 오케스트레이션**: riscv64 네이티브 엔진은 rootfs에 고정, 모델은 swap 가능한 볼륨(sdcard.img), 오케스트레이션(audio→STT→LLM→TTS)은 호스트 harness가 콘솔로.
- **빌드는 격리 docker(server5)** → 바이너리만 server4로. server4의 공유 컨테이너 기존 종속성을 **건드리지 않는다**(읽기·부팅 검증만).
- CPU-only · 합 ≤2GB · per-stage latency 기록.

---

## 1. 접속 (server4)

```bash
# tailscale 노드 [runtime-host], 포트22 키 인증. 비밀번호는 채팅/문서에 쓰지 않는다.
ssh -o BatchMode=yes -o ConnectTimeout=12 bohyun@<[runtime-host]>   # 구체 IP는 로컬 메모리
# 원격 출력 잡음(post-quantum/docker emulate)은 grep 필터:
#   ... | grep -v -E "post-quantum|store now|server may need|Emulate Docker|nodocker"
```

- 호스트: x86_64, **podman**(docker alias). passwordless sudo 아님 → root 작업은 사용자가.
- VP 워크스페이스: 공유 컨테이너 안 `/projects/ssl-npu/ssl-virtual-platform`.

---

## 2. VP 부팅 — `ssl-soc-gen2`

> **`ssl-soc-gen2`를 쓴다.** `ssl-soc-qbox`는 부팅 자산(`simple_jump_boot.elf`/`simple_bootrom.elf`/`fw_payload.bin`)이
> 미빌드라 abort한다. gen2는 자산이 prebuilt이고 **DDR3 2GB**다.

```bash
cd /projects/ssl-npu/ssl-virtual-platform
./build/platforms/platforms-vp --gs_luafile platforms/ssl-soc-gen2/conf_ssl-soc-gen2.lua
```

**부팅 체인(실측):**
```
Reset(0x1000) → ZeroStageROM(0x20000000, simple_bootrom.elf)
  → BootRAM(0x20020000, simple_jump_boot.elf)
  → OpenSBI v1.5 (0x80000000, M-mode, fw_payload.bin 27M)
  → Linux 6.16 (0x80200000, S-mode)
  → initramfs(rootfs.cpio 20M) → /sbin/init → login: ttySIF0,115200
```
기대 출력: `Welcome to SSL-SoC Linux (RISC-V64)` → `ssl-soc login:`. (자동 로그인 또는 `root`)

---

## 3. 콘솔 자동화 (GUI 없음 → stdin 주입)

호스트 오케스트레이션(VP-D)의 토대. 로그인 후 명령 실행 → 출력 회수:

```bash
{ sleep 150; printf "root\n"; sleep 10; \
  printf "<command>; echo EXIT=\$?\n"; sleep 30; } \
| timeout 220 ./build/platforms/platforms-vp \
    --gs_luafile platforms/ssl-soc-gen2/conf_ssl-soc-gen2.lua
```

**검증(실측):** `llama-cli --version` → `version: 1 (0dedb9e)`, **EXIT=0**.
rootfs.cpio(20M)에 `usr/bin/llama-cli` 이미 포함.

---

## 4. riscv64 엔진 크로스컴파일 (격리 docker, server5)

```bash
# llama.cpp — riscv64 (rv64gc, hard-float lp64d), static
cmake -B build-rv64 \
  -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++ \
  -DCMAKE_C_FLAGS="-march=rv64gc -mabi=lp64d" \
  -DCMAKE_CXX_FLAGS="-march=rv64gc -mabi=lp64d" \
  -DGGML_RV_ZICBOP=OFF -DGGML_RV_ZFH=OFF -DGGML_RV_ZIHINTPAUSE=OFF \
  -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build-rv64 -j   # → llama-cli ~8.3MB (static)
```

- IREE 런타임 riscv64도 같은 triple로 크로스빌드(호스트엔 x86 런타임만 빌드돼 있음).
- 빌드 산출 바이너리만 server4로 복사 → rootfs/볼륨에 배치.

---

## 5. 배포 — 엔진은 rootfs, 모델은 볼륨

| 구성요소 | 위치 | 이유 |
|---|---|---|
| riscv64 엔진(llama.cpp/whisper.cpp) | `rootfs.cpio`에 고정 | 자주 안 바뀜 |
| 모델(.gguf) | swap 가능한 `sdcard.img`/블록 디바이스 | **주기적으로 lowering·경량 모델로 교체** |
| 오케스트레이션 | 호스트 harness(콘솔) | Python+torch는 에뮬 riscv64에서 못 돎 → 네이티브 엔진 + 호스트 제어 |

---

## 6. 마일스톤 + 남은 갭

| 단계 | 상태 |
|---|---|
| **VP-0** 부팅 + 셸 + `llama-cli --version` EXIT=0 | ✅ (2026-06-10) |
| **VP-A** 실제 토큰 생성 | ⏳ 게스트에 runnable GGUF 필요 |
| **VP-B** whisper.cpp(STT) | 미착수 |
| **VP-C** TTS riscv64 | 미착수 |
| **VP-D** 호스트 오케스트레이션 1턴 | 미착수 |

**VP-A 다음 1수:** 게스트에 모델이 없다(vocab-only만). 1.3GB Llama-3.2-1B Q8은 2GB RAM엔 맞지만 20MB initramfs엔 안 들어감 →
**작은 GGUF**(예: SmolLM2-135M Q8 ~145MB)로 initramfs 확장해 첫 토큰 생성 증명 → 1.3GB Llama는 **모델 볼륨**(sdcard.img)으로.

```bash
# (VP 안에서) 모델 배치 후
llama-cli -m /mnt/models/<small>.gguf -p "<프롬프트>" -n 16
```

---

## 7. gotcha

| 증상 | 원인 | 해결 |
|---|---|---|
| `cannot open elf file simple_jump_boot.elf` | `ssl-soc-qbox`는 부팅 자산 미빌드 | **`ssl-soc-gen2`** 사용 |
| 포트 40022 연결 안 됨 | tailscale 인터페이스에서 방화벽 차단 | 포트 22 + 키 인증 |
| llama-cli 실행되나 토큰 생성 0 | 게스트에 runnable 모델 없음 | 작은 GGUF / 모델 볼륨(§6) |
| 공유 컨테이너 deps 깨짐 우려 | server4 컨테이너는 기존 작업과 공유 | 빌드는 server5 격리 docker, server4는 읽기·부팅만 |
