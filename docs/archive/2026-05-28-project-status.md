> ⚠ **ARCHIVED (구버전, ~2026-05, superseded).** 최신 현황은 [status](../status.md) · 문서 지도 [docs/README](../README.md). 본문 링크는 원래 위치 기준일 수 있음.

# Project Status — 2026-05-28

> 본 세션의 작업 결과 + 앞으로의 작업 정리. TodoWrite 의 시각화 보완.

---

## 1. 이미 완료한 작업 (이번 세션 누적, Phase A→E + Step 5)

```mermaid
flowchart LR
    classDef done fill:#bbf7d0,stroke:#16a34a,stroke-width:2px,color:#052e16

    A["<b>Phase A</b><br/>메모리 규칙 3종<br/>self-owned narrative<br/>no-push-sensitive<br/>destructive-actions-guard"]:::done
    B["<b>Phase B</b><br/>torch-mlir-zoo sanitize<br/>RECIPES-zoo.md 신규<br/>artifacts 17 GB rm<br/>fed8dcb"]:::done
    C["<b>Phase C</b><br/>3 branch sanitize + cross-ref<br/>vision/voice/release<br/>15946ac · 20199e6 · 7144329"]:::done
    D["<b>Phase D</b><br/>archive 3 branch → tag<br/>v0.1-mvp-baseline<br/>exp-hyperclovax-edge-tts<br/>v0.1.0"]:::done
    E["<b>Phase E</b><br/>OSS 표준 (LICENSE/CONTRIBUTING/SECURITY)<br/>main README + repo meta<br/>3 app README + 4 draft PR<br/>f4599ec · 3057623 · 6c8024c · 5c2831e"]:::done
    S5["<b>Step 5 (별도 turn)</b><br/>real Llama-3.2-1B → iree.turbine<br/>→ iree-compile → .vmfb 6 GB<br/>0d62acc"]:::done

    A --> B --> C --> D --> E
    S5 -.선행.-> B
```

---

## 2. 앞으로 할 작업 (우선순위별)

```mermaid
flowchart TB
    classDef now fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef high fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef med fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef low fill:#e0e7ff,stroke:#4f46e5,stroke-width:1px,color:#312e81
    classDef block fill:#e5e7eb,stroke:#6b7280,stroke-width:1px,color:#374151,stroke-dasharray: 5 3

    subgraph IMMEDIATE ["⏰ 즉시 후속 (사용자 영역, 코딩 0)"]
        T1["<b>.github/workflows/ci.yml push</b><br/>PAT workflow scope 부여<br/>또는 GitHub web UI"]:::now
        T2["<b>PR #1 머지 검토</b><br/>main 의 LICENSE/CONTRIBUTING/<br/>SECURITY 활성화"]:::now
    end

    subgraph HIGH ["🔥 High priority — 다음 turn"]
        T3["<b>Step 6 (a) Q8_0 GGUF</b><br/>unsloth/Llama-3.2-1B-Instruct-GGUF<br/>budget 위반 12.3 GB 정량 정당화"]:::high
        T4["<b>Step 6 (b) Whisper-tiny-INT8</b><br/>rhasspy/faster-whisper-tiny-int8<br/>Deliverable 2 STT 부합도 0→1"]:::high
    end

    subgraph MEDIUM ["⚙ Medium priority"]
        T5["<b>.vmfb runtime numerical</b><br/>PyTorch eager vs IREE Runtime<br/>출력 비교"]:::med
        T6["<b>MLIR 슬림화</b><br/>aot.externalize_module_parameters<br/>12 GB → 수 MB + .irpa"]:::med
        T7["<b>Qwen-0.5B lowering</b><br/>voice-app default LLM 정합"]:::med
    end

    subgraph LOW ["✨ Low / Optional"]
        T8["<b>target-cpu=host</b><br/>iree-compile generic warning 해소"]:::low
        T9["<b>TTS lowering</b> (Gap C-2)<br/>open question 답변 후"]:::low
    end

    subgraph BLOCKED ["🚫 Blocked"]
        T10["<b>Deliverable 3 (VP 진입)</b><br/>VP / driver / SoC 모델 수령 대기"]:::block
    end
```

---

## 3. 본 세션 작업 timeline (시간순)

```mermaid
gantt
    title 본 세션 누적 (Phase A → E)
    dateFormat YYYY-MM-DD
    axisFormat %m-%d

    section Phase A (메모리)
    self-owned / no-push / destructive guard  :done, ph_a, 2026-05-27, 1h

    section Phase B (torch-mlir-zoo)
    sanitize 10 파일 + RECIPES-zoo + artifacts rm   :done, ph_b, after ph_a, 2h

    section Phase C (3 branch)
    vision-app cross-ref + .gitignore        :done, ph_c1, after ph_b, 30m
    voice-app sanitize 11 + cross-ref        :done, ph_c2, after ph_c1, 1h
    release/v1-wrap-up merge + cross-ref     :done, ph_c3, after ph_c2, 30m

    section Phase D (archive)
    tag v0.1-mvp-baseline / exp-* / v0.1.0    :done, ph_d, after ph_c3, 30m

    section Phase E (OSS / PR)
    LICENSE+CONTRIBUTING+SECURITY+main README  :done, ph_e1, 2026-05-28, 1h
    repo description / topics / homepage       :done, ph_e2, after ph_e1, 10m
    3 app README badge/nav/citation           :done, ph_e3, after ph_e2, 1h
    4 draft PR (#1-#4)                          :done, ph_e4, after ph_e3, 20m
```

```mermaid
gantt
    title 앞으로 (제안)
    dateFormat YYYY-MM-DD
    axisFormat %m-%d

    section 즉시 (~1 day)
    ci.yml push (사용자)         :crit, t1, 2026-05-29, 1d
    PR #1 머지                   :t2, 2026-05-29, 1d

    section Step 6 (1-2 weeks)
    Q8_0 GGUF lowering          :crit, t3, after t2, 5d
    Whisper-tiny-INT8 lowering  :crit, t4, after t3, 5d

    section Polish (1 week)
    .vmfb numerical 검증         :t5, after t4, 2d
    MLIR 슬림화 (.irpa)          :t6, after t5, 2d
    Qwen-0.5B lowering          :t7, after t6, 3d

    section Phase 3
    VP 진입 (외부 대기)          :milestone, m, 2026-06-15, 0d
```

---

## 4. 작업 비중 (완료 vs 대기)

```mermaid
pie title 본 세션 작업 분포 (15 핵심 항목)
    "✅ 완료 (6 phase)" : 6
    "⏰ 즉시 후속 (2 항목)" : 2
    "🔥 High (2 항목)" : 2
    "⚙ Medium (3 항목)" : 3
    "✨ Low (2 항목)" : 2
"#$ 0
```

> 본 세션은 *인프라/배포 정비* 위주. 다음 세션부터는 *실제 lowering 코드* (Step 6) 비중이 커짐.

---

## 5. 현재 origin 상태 매트릭스

```mermaid
flowchart LR
    classDef branch fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef tag fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef pr fill:#dcfce7,stroke:#15803d,color:#14532d

    subgraph BRANCHES ["origin/refs/heads/ (5 branch)"]
        M["main<br/>da3e72d"]:::branch
        TZ["torch-mlir-zoo<br/>3057623 ✓"]:::branch
        VA["voice-app<br/>6c8024c ✓"]:::branch
        VS["vision-app<br/>5c2831e ✓"]:::branch
        RV["release/v1-wrap-up<br/>7144329 ✓"]:::branch
        CO["chore/oss-standards<br/>f4599ec ✓ (PR #1)"]:::branch
    end

    subgraph TAGS ["origin/refs/tags/ (3 tag)"]
        T1["v0.1-mvp-baseline<br/>045af08"]:::tag
        T2["exp-hyperclovax-edge-tts<br/>01625fc"]:::tag
        T3["v0.1.0<br/>9592c6f"]:::tag
    end

    subgraph PRS ["Open PRs (all draft)"]
        P1["#1 chore/oss-standards → main<br/>(유일한 merge 후보)"]:::pr
        P2["#2 torch-mlir-zoo → main<br/>(visibility only)"]:::pr
        P3["#3 voice-app → main<br/>(visibility only)"]:::pr
        P4["#4 vision-app → main<br/>(visibility only)"]:::pr
    end

    CO --> P1
    TZ --> P2
    VA --> P3
    VS --> P4
```

---

## 6. 요약 표

| 영역 | 누적 결과 |
|---|---|
| 본 세션 commit 수 | **9** (fed8dcb / 15946ac / 20199e6 / 7144329 / f4599ec / 3057623 / 6c8024c / 5c2831e / + main reset) |
| 본 세션 push 수 | 5 branch + 3 tags |
| 작성한 파일 (신규) | LICENSE · CONTRIBUTING.md · SECURITY.md · `.github/workflows/ci.yml`* · `.github/workflows/md-link-check-config.json`* · `docs/RECIPES-zoo.md` · `development.md` · 5개 시각화 markdown |
| 정비한 파일 | README × 4 branch (main / torch-mlir-zoo / voice-app / vision-app / release/v1-wrap-up) · `.gitignore` × 4 branch · 14 파일 sanitize (voice-app) + 10 파일 sanitize (torch-mlir-zoo) |
| GitHub repo 메타 | description / topics 10종 / homepage 업데이트 |
| 디스크 회수 | artifacts 17 GB rm + .codex-reviews archive |
| 메모리 규칙 | 새 3종 (self-owned narrative / no-push-sensitive / destructive-actions-guard) |
| 가드 위반 | **0** (모든 push 후 token leak 0, force-push 0, main 직접 push 0, ssl-mirror 0) |

\* `.github/workflows/*.yml` 은 working tree 에만 — PAT `workflow` scope 부족으로 사용자 후속 처리 필요.

---

## 7. "어떤 작업이 우선" — 한 줄 가이드

| 사용자가 원할 때 | 첫 작업 |
|---|---|
| 본 세션 완전 마무리 | `.github/workflows/ci.yml` push (사용자 영역) + PR #1 머지 |
| 다음 기술 진보 | **Step 6 (a) Q8_0 GGUF lowering** (High priority, 위 timeline 의 1-2 주차) |
| 다음 시각화/문서만 | (없음 — 본 세션으로 인프라 완료, 다음은 코드) |
| Deliverable 3 진입 | VP / driver / SoC 모델 수령 대기 (외부) |
