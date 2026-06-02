# configs/variants/

> **Status**: variants 는 *모델 비교용 reference only*. 설계 §2 *"알고리즘이나
> 모델을 확정한 다음에"* 에 따라 **final selection 은 [`../default.yaml`](../default.yaml)
> (Whisper-small / Qwen2.5-0.5B / espnet_kss) 로 locked**. variants 는 향후 lowering /
> VP 단계에서 비교 baseline 으로만 사용. 자세히는 (내부 문서) §"Phase 1 선정 모델".

`default.yaml`은 그대로 두고, 모델 스왑 실험은 여기 별도 YAML로 관리.

로더(`scripts/run_demo.py`, `scripts/web_demo.py`)는 부분 머지를 지원하지 않으므로
각 variant 파일은 `default.yaml`의 **완전한 복사본**에서 바꾸려는 섹션만 수정한 형태.

## 사용

```bash
python scripts/run_demo.py -c configs/variants/<file>.yaml
# 또는
TARGET_APP_CONFIG=configs/variants/<file>.yaml python scripts/web_demo.py
```

## 현재 variant

| 파일 | 변경점 | 메모 |
|---|---|---|
| `llm-hyperclovax-1.5b.yaml` | LLM → HyperCLOVAX-1.5B (KSS JETS TTS 유지) | 네이버 한국어 네이티브 SLM. fp16 ~3GB, 2GB 예산 초과(경고만). **HF gated** — 컨테이너에서 `huggingface-cli login` 1회 필요 |
| `llm-hyperclovax-1.5b-edge-tts.yaml` | 위 + TTS → Edge-TTS (Microsoft Neural) | 음성 자연스러움 ↑↑. **클라우드 호출** → 네트워크 필요. GPU/모델 가중치 0. 임베디드 타겟 가서는 다시 로컬 TTS로 교체 예정 |

## 새 variant 추가 절차

1. `cp configs/default.yaml configs/variants/<name>.yaml`
2. 바꿀 섹션(`llm:` / `tts:` / `stt:`)만 수정
3. 상단 주석에 변경 이유·메모리·실행법 한 줄씩 기록
4. 이 README의 표에 한 줄 추가
