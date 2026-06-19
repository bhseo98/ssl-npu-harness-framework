1. 실험제목

한국어 Voice AI Pipeline (STT-LLM-TTS) End-to-End 기능 검증 · GPU Profiler 측정 · 2GB Budget 해석 분석 (Naive Application Code 완료)

2. 실험목적

지난 주 framework contract 검증을 마친 voice-app pipeline 을, 실제 GPU 서버 (RTX Pro 6000) 의 Docker (CUDA 12.8) 위에서 한국어 마이크 입력 → STT → LLM → TTS → 한국어 음성 출력 한 turn 으로 동작시키고, 각 stage 의 latency · GPU peak · RAM(RSS) · audio duration 을 자동 기록하는 것이 1차 목적이다. 설계 검토 §2 의 8 가지 명시 요구 중 "2GB 넘지 않는 모델" 조항은 (a) weight 합 / (b) 동시 load peak / (c) 전체 RSS 세 시나리오로 어떤 해석에서 합격/불합격인지 정량 판정하고, (c) 해석에 대비한 embedded simulation mode (CPU-only, sequential per-turn rebuild) path 까지 사전 구축하여 Phase 2 (torch-mlir lowering) 진입 조건을 충족시킨다. 또한 BaseStage ABC 의 registry 호환 (Gap B) 과 SequentialVoicePipeline (Gap F) 를 framework 본체에 lock 하여 "한번 만들고 모델만 갈아끼우는" 약속이 실제 모델 (Whisper/Qwen/ESPnet) 로딩 상태에서도 유효한지 e2e 로 검증한다.

3. 실험 내용

- 구성 (Phase 1 final, configs/default.yaml @ commit d5f82f0):
  · STT : faster-whisper small (CT2 INT8) ~244 MB · 한국어 인식 + 안정성
  · LLM : Qwen/Qwen2.5-0.5B-Instruct ~988 MB · 한국어 instruction-tuned
  · TTS : ESPnet KSS JETS (imdanboy) ~120 MB · 한국어 native voice (KSS)
  · 모델 합 ≈ 1.35 GB (< 2 GB), Container = Dockerfile (CUDA 12.8 devel)
  · Stage ABC = BaseStage marker + BaseSTT/LLM/TTS compat (Gap B, commit de29201)
  · Embedded path = Dockerfile.cpu + configs/embedded.yaml + SequentialVoicePipeline (Gap F, commit 2227745)
  · Profiler = measure(stage, budget_mb=2048) — latency · GPU peak · RAM RSS · audio duration 자동, 초과 시 stdout 경고
  · 모델 swap = YAML 한 줄 (stt.type/llm.type/tts.type) + @register decorator, configs/variants/ 7 종 (Whisper-tiny/base, Qwen-1.5B, HyperCLOVAX-1.5B, Llama-3.2-1B, Edge-TTS)
  · Test framework = pytest 22+ cases / 3 파일 (test_pipeline · test_base_stage_compat · test_sequential_pipeline)

- 기능 검증 (한국어 web demo, capture/2.png):
  · MacBook 마이크 입력 "확률이 뭐야?" → STT (Whisper-small) → LLM (Qwen-0.5B 한국어 응답) → TTS (KSS JETS 15.42 s @ 24 kHz, 즉시 재생) 한 turn end-to-end 정상 동작.
  · 동일 세션 내 multi-turn 3 회 이상 누적 RAM Δ +0~+2 MB → 누수 없음.
  · 대화 history 가 VoicePipeline.history 에 누적되어 LLM context 로 전달됨을 확인.

- Profiler 측정 (3 turn 발췌, capture/1.png · capture/3.png):
  · STT : latency 325~1211 ms, GPU peak 1446 MB
  · LLM : latency 1002~1337 ms, GPU peak 1684 MB
  · TTS : latency 75~82 ms, GPU peak 1725~1825 MB, audio 12~16 s
  · 전체 RAM (RSS) : 4518~4522 MB (turn 누적 안정, Δ +0~+2 MB)

- 설계 §2 "2GB 안" 해석별 판정 (정량화):
  · (a) weight 합 ≤ 2 GB : 1.35 GB ✅
  · (b) GPU VRAM peak (model only) ≤ 2 GB : 1.8 GB ✅
  · (c) 전체 RSS ≤ 2 GB : 4.5 GB ❌ (GPU mode)
  · (c) 대응 path : Dockerfile.cpu + configs/embedded.yaml + SequentialVoicePipeline (per-turn build→run→del→gc→empty_cache) 사전 구축, 프로젝트 리드 Q4 답변 즉시 RSS 재측정 가능 상태.

- 모델 swap 약속 실증:
  · configs/variants/ 7 종 (Whisper-tiny/base, Qwen-1.5B, HyperCLOVAX-1.5B, Llama-3.2-1B, Edge-TTS) 모두 YAML 한 줄 변경 + @register decorator 만으로 교체 가능, framework 본체 코드 0 라인 변경.
  · BaseStage ABC ↔ BaseSTT/LLM/TTS 호환 11 test (test_base_stage_compat.py) 통과.

- Phase 2 진입 조건 7/7 충족:
  · 모델 lock (d5f82f0) ✅
  · Demo 동작 ✅
  · Profiler 작동 ✅
  · Stage ABC 호환 (de29201) ✅
  · Budget (a)(b) ✅
  · Budget (c) path 준비 (2227745) ✅
  · Test framework 22+ tests ✅
  → torch-mlir-zoo branch 의 Step 3~5 (sharktank 분석 + 단위 op + Llama IR dump) 1차 milestone 진행 가능.

- 산출물:
  · REPORT-2026-05-21-phase1-final.md (6 Mermaid diagram + 8 요구 매트릭스 + Q4 해석 분기 + Phase 2 진입 조건 + 부록 A~D)
  · docs/reports/2026-05-21-phase1-demo.md (프로젝트 리드 제출용 단축본)
  · capture/{1,2,3}.png (web demo 골든 패스 + profiler stdout)

- 한계 / 후속:
  · RSS 측정 상한 가설 : psutil.Process.memory_info().rss 는 swap 미포함 + shared lib 일부 공유 가능으로 *상한* 해석. CUDA stack 1.4 GB 기여분은 분해 *추정* → pmap / /proc/<pid>/smaps 정밀 분해는 후속.
  · LLM 응답 수치 정확도 trade-off : Qwen-0.5B 의 "확률" 정의 수치 (50/25 등) 부정확. 설계 §2 "기능 위주" scope 내 의도적 보류. Phase 2 lowering 대상 Llama-3.2-1B + INT8 으로 quality-budget 재평가 예정 (Q1 답변 대기).
  · Embedded mode 실측 RSS 부재 : path 는 구축·테스트 완료 (test_sequential_pipeline.py 6 test), 프로젝트 리드 Q4 답변 수령 후 한 번에 측정.
  · 단일 입력 검증 한계 : 짧은 한국어 질문 위주. WER / MOS / response quality 의 정량 benchmark 는 Phase 2 "유의미한 결과" (Q5) 정의 확정 후 추가 round.
  · Phase 2 잔여 (agents/roadmap.md Gap C) : C-1 Whisper-small lowering, C-2 TTS lowering, C-3 voice e2e MLIR → Q1/Q2 답변이 unblocking 조건.

기록일자 Date

2026.05.22

