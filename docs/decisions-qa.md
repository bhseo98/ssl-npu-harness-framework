# Decisions & Q&A — 세션 질문 정리 (검증 인덱스)

> 진행 중 주고받은 **질문 → 답/결정 → 근거·문서 위치**를 한곳에 모은 인덱스. "내가 물어본 게 다 정리됐나"를 여기서 확인한다.
> 상세는 각 링크 문서가 단일 소스. Last updated: **2026-06-11**.

---

## 0. 한눈에 (상태표)

| # | 질문(요약) | 답/결정 | 위치 | 상태 |
|---|---|---|---|---|
| Q-A1 | 에이전트 팀이 목표 달성했는지 더블체크 | ① 달성(TTS만 PENDING) · ② 부분(target 정렬은 해소) · ③ 부팅까지 | [STATUS §1](status.md) | ✅ |
| Q-A2 | VP 에이전트 만들기 | 기존 `vp-embedded-engineer` 갱신(운영 지식 반영) | `.claude/agents/vp-embedded-engineer.md` | ✅ |
| Q-A3 | VSCode Remote-SSH에 gpu4-lab 추가 | 맥북 `~/.ssh/config` 블록 제공(§2) | 본 문서 §2 | ✅ 제공 |
| Q-A4 | 모든 가이드라인·레시피 포멀 작성(easy는 별도) | GUIDELINES + RECIPES 3종, easy-series는 별도 트랙 | [GUIDELINES](guidelines.md)·[RECIPES](recipes/kernels.md) | ✅ |
| Q-A5 | 작업물 문서화·push | feature/sslnpu-kernels 6889293 등 | git | ✅ |
| Q-B1 | server4 컨테이너 안에서 branch 작업 가능? | 비권장 — 우리 repo 없음·신원/deps 다름 → server5 dev + 격리 워크스페이스 | §3 · [RECIPES-vp §0.5](recipes/vp.md#05-런타임-아키텍처--호스트-오케스트레이션--vp-게스트-엔진) | ✅ |
| Q-B1b | 2달+ 구버전 정리 | 버킷 B/C/D/E 승인 → 분류기 차단으로 사용자 실행(스크립트 제공) | §3 | ⏳ 사용자 실행 |
| Q-B2 | "voice app"→"speech" | `voice-app-engineer`→`speech-engineer` 개명 + 용어 통일 | git e13e3ed | ✅ |
| **Q3** | **4번 VP 위에서 5번 모델이 뭐가 어떻게 되나** | 2-plane(호스트 코어+게스트 엔진), CPU=GGUF/ggml·NPU=vmfb(미래) | [RECIPES-vp §0.5](recipes/vp.md#05-런타임-아키텍처--호스트-오케스트레이션--vp-게스트-엔진) · [그림](diagrams/vp-runtime-architecture.drawio) | ✅ |
| Q-C1 | NPU/RV/MC(DRAM) 버스 width(동료 확인) | NPU 64-bit AXI(flit 34) · RV 32 · MC 32 | [RECIPES-vp §8](recipes/vp.md#8-soc-스펙--버스-width--메모리-맵-ssl-soc-gen2-실측) | ✅ |
| Q-D1 | harness가 VP 위 동작방식·아키텍처 | 런타임 아키텍처 + SoC 맵 문서화 | [RECIPES-vp §0.5·§8](recipes/vp.md) · drawio | ✅ |
| Q-E1 | 실마이크 WebUI 테스트 방법 | ① 서버 디딤돌 recipe(지금 실행 가능) | [RECIPES-webui-mic §1](recipes/webui-mic.md) | ✅(①만) |
| Q-E3 | WebUI가 **VP에서** 돌게 — 원래 목적 정렬 | **Option A**: 호스트 WebUI + VP 게스트 추론. compute=VP. ※ 추론이 서버서 돌던 ① 레시피는 디딤돌로 재표기 | [RECIPES-webui-mic §0](recipes/webui-mic.md) · drawio | ⏳ 설계확정·선행조건 미완 |
| Q-E2 | 그림 draw.io 정리 | VP 런타임 아키텍처 다이어그램 | [vp-runtime-architecture.drawio](diagrams/vp-runtime-architecture.drawio) | ✅ |
| (확정) | target 파이프라인 | WebUI push-to-talk → whisper-tiny INT8 → Llama-3.2-1B → TTS(KO, 선정 PENDING) | [STATUS §3](status.md) | ✅ |
| (확정) | 출처 신선도 | 5번 서버 = canonical, server4 컨테이너 = 2달+ 구버전 | 에이전트/메모리 | ✅ |

남은 결정(사용자 몫): **TTS 모델 선정** · **2GB 배포 RSS 분리 측정** · **VP-A 토큰 생성** · **컨테이너 정리 실행**.

---

## 1. Q3 — 4번 VP 위에서 5번 모델이 "뭐가 어떻게" (핵심)

VP의 에뮬 riscv64 Linux엔 Python+torch를 못 올린다(2GB·휠 부재·에뮬 속도). 그래서 **두 평면**으로 나뉜다:

- **프레임워크 코어(BaseStage/registry/pipeline/profiler)는 호스트(server4)에 잔류**, 각 stage가 "VP 엔진 어댑터"가 된다(모델 swap 철학 유지).
- **제어 평면 = 콘솔**: 호스트 harness가 stdin 주입으로 게스트 명령 실행·출력 파싱.
- **데이터 평면 = 모델 볼륨(sdcard.img)**: 오디오·모델 가중치·출력 wav. 모델 교체 = 볼륨 내용 교체(엔진은 rootfs 고정).
- **CPU 트랙(지금) = GGUF/ggml + llama.cpp/whisper.cpp** vs **NPU 트랙(미래) = IREE vmfb(②) → IREE HAL → SSLNPU**. 즉 "5번 모델이 VP에 올라간다" = **양자화 GGUF/ggml 가중치 + riscv64 엔진**.

전체 그림: [diagrams/vp-runtime-architecture.drawio](diagrams/vp-runtime-architecture.drawio) · 상세: [RECIPES-vp.md §0.5](recipes/vp.md#05-런타임-아키텍처--호스트-오케스트레이션--vp-게스트-엔진).

---

## 2. Q-A3 — VSCode Remote-SSH에 gpu4-lab 추가

VSCode Remote-SSH config는 **맥북** `~/.ssh/config`에 있어 server5에서 직접 못 고친다. 맥북에 추가:

```sshconfig
Host gpu4-lab
    HostName <gpu4-lab tailscale IP>    # tailscale status, 또는 local memory reference_vp_server4
    User bohyun
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

- MagicDNS(`gpu4-lab.<tailnet>.ts.net`)도 가능하나 IP 권장.
- **전제**: 맥북 공개키가 `gpu4-lab:~/.ssh/authorized_keys`에 등록돼야 함(`ssh-copy-id gpu4-lab`). 현재 키 등록은 gpu5-lab 것.

---

## 3. Q-B1 — 컨테이너 branch 작업 + 정리

**branch 작업**: 비권장. server4 `ssl-npu-dev`엔 **우리 repo가 없고**(VP 인프라 repo만), git 신원이 별도 랩 계정, `gh` 미설치, 공유 deps. → 개발·git은 **server5(canonical)**, server4엔 **격리 워크스페이스 dir**(우리 소유)만 두고 배포 산출물 + 얇은 orchestrator. 공유 컨테이너 repo에 branch 금지.

**정리(승인 B/C/D/E, 약 7.9GB)**: KEEP = VP 인프라(ssl-virtual-platform·qbox·llama.cpp·amd-shark-ai·venv)·`Llama-3.2-1B-Instruct-Q8.gguf`·PDF. 삭제 = 중복 Llama Q8 변형(~7.2GB)·옛 중간 MLIR/스크립트(~510MB)·옛 리포트/zip(~5MB)·git-export+/tmp(~244MB). `/tmp/opensbi_test` 제외(부팅자산 재빌드 소스 가능). 대량 원격 `rm`은 안전 분류기가 차단 → 검증 스크립트를 사용자가 실행(존재/크기 출력 → 삭제 → KEEP 무결성 검증 포함).

---

## 4. 문서 지도

- 결정/Q&A(본 문서) · [STATUS](status.md)(현황 단일 소스) · [GUIDELINES](guidelines.md)
- 레시피: [RECIPES-zoo](recipes/zoo.md) · [RECIPES-kernels](recipes/kernels.md) · [RECIPES-vp](recipes/vp.md) · [RECIPES-webui-mic](recipes/webui-mic.md)
- 그림: [vp-runtime-architecture.drawio](diagrams/vp-runtime-architecture.drawio)
- 설계: [kernel-design/](kernel-design/) · 학습 노트(개인, 별도): [easy-series/](easy-series/)
