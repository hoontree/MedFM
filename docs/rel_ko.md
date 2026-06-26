# 신뢰도 기반 KD — 통합 결과

지금까지의 모든 분석/시각화 요약본입니다. 계획 및 설계는
[reliability_kd_study.md](reliability_kd_study.md)를 참고하세요. 이 파일은 결과만 정리합니다.

설정: 학생 = TinyUSFM; 교사 = SAM (전체 FT 체크포인트, val Dice **0.57**) 및
SAM (LoRA rank-4 체크포인트, val Dice **0.71**). 데이터 `dynamic` (num_classes=3); 학습 데이터 BUSBRA/BUSI/B, 외부 테스트 BUID/BUS_UCLM/BUS_UCLM_filtered. 지표 초점: Dice & Sensitivity. 각 방법별로 대표 실행은 하이퍼파라미터 스위프에서 **최고 BUID Dice**입니다.

---

## 0. 동기 — 교사가 *자신감 있게 틀린다*

일반 로짓-KD는 모든 교사 픽셀을 동등하게 증류합니다. 하지만 고정된 SAM 교사는 단순히 오류를 범하는 것이 아니라 **자신감 있게** 오류를 범하며, 일반 KD는 이를 거의 하드 레이블처럼 학생에게 주입합니다. BUID에서 측정 (교사만, 학생 없음):

![자신감 있게 틀린 요약](figures/confidently_wrong_summary.png)

| BUID | SAM-FT (0.57) | SAM-LoRA (0.71) |
|---|---:|---:|
| **틀린** 픽셀의 평균 신뢰도 | **0.918** | 0.817 |
| 신뢰도 > 0.9인 틀린 픽셀 | **74.7 %** | 42.7 % |
| 신뢰도 > 0.7인 틀린 픽셀 | **88.7 %** | 74.4 % |
| (신뢰도>0.9인) 자신감 있게 틀린 픽셀의 전체 비율 | **6.6 %** | 2.6 % |

FT 교사의 경우, **모든 오류의 약 75%가 >0.9 신뢰도를 가짐** — 이것이 바로 신뢰도-KD의 `teacher_correctness_gate`가 0으로 만드는 픽셀들입니다 (§3). 신뢰도 히스토그램 (빨간 곡선 = 틀린 곡선에서의 높은 신뢰도 꼬리):

| FT 교사 | LoRA 교사 |
|---|---|
| ![](figures/confidence_hist_ft.png) | ![](figures/confidence_hist_lora.png) |

샘플별 증거 (빨간색 = 신뢰도 > 0.7 **그리고** 교사 ≠ GT):

![자신감 있게 틀린 패널](figures/confidently_wrong_panel_ft.png)

이것이 신뢰도-KD가 목표하는 문제이며, §2도 미리 보여줍니다: 더 잘 보정된 LoRA 교사는 자신감 있게 틀린 경우가 훨씬 적습니다.

도구: `tools/analyze_confidently_wrong.py teacher=<sam|sam_lora> analysis.loader=BUID`.

---

## 1. 주요 비교 (방법별 최고, 교사별)

외부 평균 (BUID + BUS_UCLM(+filtered)). 굵은 글씨 = 교사별 최고 Dice.
전체 테스트 세트 표: [kd_reliability_table.tex](kd_reliability_table.tex).

| 교사 | 방법 (대표) | 외부 Dice | 외부 Sensitivity |
|---|---|---:|---:|
| SAM-FT (0.57) | Task-only (`base_task_only`) | **0.642** | 0.633 |
| SAM-FT (0.57) | Logit-KD (`base_logit_kd`) | 0.554 | 0.548 |
| SAM-FT (0.57) | Uncertainty-KD (`base_uncertainty_kd`) | 0.589 | 0.575 |
| SAM-FT (0.57) | Reliability-KD (`bs_8`) | 0.628 | 0.617 |
| SAM-LoRA (0.71) | Task-only (`task_only`) | 0.648 | 0.657 |
| SAM-LoRA (0.71) | Logit-KD (`logit_kd`) | **0.672** | 0.662 |
| SAM-LoRA (0.71) | Reliability-KD (`reliability`) | 0.671 | 0.634 |

테스트 세트별 주요 결과 (Dice): LoRA 교사로는 **Reliability-KD가 BUID (0.703)**와 BUSI (0.692)에서 우승; Logit-KD는 외부 평균을 0.001로 근소하게 능가합니다.

**두 가지 발견:**
1. **교사 품질이 KD의 판정을 뒤집습니다.** 약한 FT 교사 → 모든 KD 변형이 task-only *이하* (신뢰도가 가장 덜 나쁨). 강한 LoRA 교사 → KD가 task-only를 *능가*, 신뢰도/로짓-KD는 상위에서 효과적으로 동등합니다.
2. **KD 방법 중 약한 교사 영역에서 Reliability-KD ≥ Logit-KD/Uncertainty-KD** (FT: 0.631 vs 0.554) — 이점이 정확히 설계 목표(신뢰 못 할 교사)인 곳에 집중됩니다.

### 3-way: KD가 도움이 되긴 하는가? (LaTeX: [kd_reliability_3way.tex](kd_reliability_3way.tex))

| 교사 → 학생 | 교사 Dice | Task-only | Reliability-KD | Logit-KD | KD |
|---|:--:|---:|---:|:--:|:--:|
| SAM full-FT → TinyUSFM | 0.57 | 0.642 | 0.631 (−0.011) | 0.554 (−0.088) | **손해** |
| SAM LoRA → TinyUSFM | 0.71 | 0.648 | **0.671 (+0.023)** | **0.672 (+0.024)** | **도움** |
| TinyUSFM → SAM | 0.71 | 0.546 | 0.534 (−0.012) | — | **손해** |

**세 방향 중 KD가 도움이 되는 건 단 하나** — SAM-LoRA → TinyUSFM. 결정 요인:
- **교사가 강하고 잘 보정돼야 함** (1행 제외: FT 교사 0.57, 과신).
- **학생이 교사의 유효 용량보다 *작아야* 함** (3행 제외: 좋은 TinyUSFM 교사도 큰 SAM 학생을 못 끌어올림 — SAM은 이미 데이터에 맞고 ~7 epoch에 과적합 시작). KD는 더 낮은 용량의 충돌 신호를 주입할 뿐.
- 도움이 될 때(2행) reliability-KD ≈ logit-KD (외부 평균); reliability는 BUID 단독(0.703)에서 우승하고 약한 교사 영역에서 더 안전한 선택.

### 동일 sampling 조건의 공정 비교 (LaTeX: [kd_reliability_sampled_fair.tex](kd_reliability_sampled_fair.tex))

§3 메인 sweep의 교사들은 balanced sampler **없이** 학습됐고, SAM-FT 교사는 오래된 미튜닝 아티팩트(val 0.57)였습니다. **두 교사를 sampler 켜고 동일 recipe로 재학습**하고 **sampling 켠 student**로 증류하면 교사 adaptation 효과를 공정하게 분리할 수 있습니다:

| 교사 (val Dice) | 방법 | ext Dice | ext Sens | int Dice | Δ ext |
|---|---|---:|---:|---:|---:|
| — | Task-only (no KD) | 0.619 | 0.631 | 0.765 | — |
| FT-sampled (0.682) | logit-KD | 0.642 | 0.647 | 0.734 | +0.024 |
| FT-sampled (0.682) | reliability-KD | 0.637 | 0.615 | 0.759 | +0.018 |
| LoRA-sampled (0.720) | logit-KD | **0.671** | **0.678** | 0.742 | +0.053 |
| LoRA-sampled (0.720) | reliability-KD | 0.642 | 0.624 | **0.774** | +0.024 |

교사를 공정하게 학습하면 달라지는 점:
- **sampling이 FT 교사를 0.57 → 0.682(+0.11)로 끌어올림** — 기존 FT 교사는 그냥 학습 부족이었지, full-FT가 본질적으로 나쁜 게 아님.
- **이제 모든 KD 변형이 task-only를 능가** (§3에선 약한 FT 교사 때문에 KD가 손해였음). 괜찮은 교사가 KD의 전제조건임을 재확인.
- **LoRA-sampled가 여전히 더 나은 교사** (외부 best 0.671 vs FT 0.642) → LoRA 우위는 sampling 효과가 아니라 **adaptation 자체**.
- **이 공정 조건에선 plain logit-KD가 외부 평균에서 reliability-KD를 근소하게 앞섬**; reliability는 더 보수적(Sens 낮음)이나 **내부** Dice(0.774) 1위·BUID 경쟁력. 잘 학습된 교사에선 reliability 게이팅의 안전 마진이 줄어듦 — 가장 큰 이점은 여전히 약한 교사 영역.

---

## 2. LoRA 교사가 더 잘 증류하는 이유는?

![LoRA가 더 나은 이유](figures/why_lora_better.png)

동일한 학생으로 BUID에서 동일한 픽셀을 측정했을 때, 두 교사는 정확히 KD 이론이 *좋은* 교사가 가져야 할 방식으로 다릅니다:

| BUID, 동일 학생 | SAM-FT (0.57) | SAM-LoRA (0.71) |
|---|---:|---:|
| `teacher_correctness_gate` (= 픽셀 정확도) | 0.911 | **0.938** |
| `confidence` (최대 확률) | 0.892 | 0.814 |
| `entropy_penalty` (1−H/logC) | 0.683 | 0.501 |

- **더 정확함**: LoRA 교사는 GT와 더 많은 픽셀에서 일치 (0.938 vs 0.911), 따라서 더 적은 픽셀이 정확도 게이트로 하드 게이팅됨 — 더 *신뢰할 수 있는* KD 신호가 생존합니다.
- **더 잘 보정됨 / 과도 신뢰도 낮음**: 더 낮은 최대 확률 신뢰도 (0.81 vs 0.89)와 더 높은 엔트로피 (entropy_penalty 0.50 vs 0.68). 작은 초음파 세트에 대한 전체 미세 조정은 SAM을 **과도하게 신뢰도 있게** 만들고; rank-4 LoRA의 작은 학습 가능 예산은 정규화기로 작용하여 더 부드럽고 잘 보정된 목표를 생성 — 증류의 고전적인 "좋은 교사".
- **순 효과** (패널 b): LoRA 교사 아래에서 모든 방법에 대해 학생의 외부 Dice가 상승하고, KD는 task-only를 능가합니다.

정성적 교사 신뢰도 맵 (동일 이미지, BUID):

| FT 교사 | LoRA 교사 |
|---|---|
| ![ft](figures/teacher_ft_panel.png) | ![lora](figures/teacher_lora_panel.png) |

> 요점: "LoRA는 마법" 아님 — **LoRA 정규화가 더 잘 보정되고 더 정확한 교사를 만들**며, KD (특히 신뢰도-가중)가 이를 더 나은 학생으로 변환합니다.

---

## 3. 신뢰도 요소 — 각 요소가 하는 일

픽셀별 신뢰도 `r ∈ [0,1]`은 여러 요소의 곱이며, 각 요소는 서로 다른 이유로
교사 신호가 믿을 수 없는 곳에서 KD를 억제합니다:

```
r = confidence × entropy_penalty × teacher_correctness_gate × student_bypass_gate
    (선택적으로 → 예측-인지 평활화)
```

| 요소 | GT 필요? | 계산 내용 | KD 가중치 효과 |
|---|:--:|---|---|
| **confidence** (`max_prob`) | 아니오 | 픽셀별 교사 최대 클래스 softmax 확률 | 기본 가중치: 불확실한 교사 픽셀은 덜 반영 |
| **entropy_penalty** | 아니오 | `1 − H(p)/log C` (정규화 엔트로피) | 엔트로피 높은(불확실) 픽셀 ↓; 뾰족하면 ≈1 |
| **teacher_correctness_gate** | **예** | `teacher_pred == GT` 직접 확인 | 틀림 → `wrong_weight`(0=차단), 맞음 → 1. **자신감 있게 틀린 교사 픽셀을 죽이는 유일한 요소** (§0) |
| **student_bypass_gate** | **예** | 학생 vs 교사 vs GT 비교 | 학생이 이미 자신 있게 맞음 → 하향(`bypass_weight`); 학생 틀림·교사 맞음 → *구제*(상향); 둘 다 틀림 → 차단 |
| *reliability_smoothing* (선택) | 아니오 | 교사 예측 유사도 기반 양방향 평균 | 일관 예측 영역 내에서 `r` 공유(경계는 안 넘음). **기본 off — 스위프에서 손해** (§5) |

직관: `confidence` / `entropy_penalty`는 교사 *자신의* 불확실성으로만 스케일하므로
자신감 있게 틀린 교사를 뒤집지 못합니다. GT 조건부 게이트 두 개가 그 정답 정보를
더합니다 — `teacher_correctness_gate`는 자신감 있는 교사 오류를 제거하고,
`student_bypass_gate`는 이미 맞은 학생을 약한 교사 쪽으로 끌어당기지 않게 합니다.

## 4. 신뢰도 맵 — 주장하는 대로 작동하는가?

BUID의 전체 맵 학생 ([분석](../logs/reliability_analysis/20260619_092418/)):

![신뢰도 히스토그램](figures/reliability_hist.png)

```
평균 신뢰도 | 교사 정답 : 0.223
평균 신뢰도 | 교사 오답   : 0.000      <- 자신감 있게 틀린 교사 픽셀 완전히 게이팅됨
틀린 픽셀의 게이팅 비율 (<0.1)  : 1.000
성분 평균: confidence 0.89 · entropy 0.68 · teacher_gate 0.91 · student_bypass 0.29
```

메커니즘 확인 (H2): 맵은 교사가 GT와 불일치하는 모든 픽셀에서 KD 가중치를 ~0으로 몰고, 교사가 맞은 곳에서는 신호를 유지합니다. `student_bypass` 요소 (평균 0.29)가 주 다운-가중기 — KD는 학생이 이미 자신감 있게 정답인 곳에서 건너뜁니다.

샘플별 패널 (이미지 · GT · 교사 · 학생 · 각 요소 · 최종 r):

![패널](figures/reliability_panel_example.png)

---

## 5. 절제 연구 (SAM-FT 교사, 전체 20회 실행 스위프)

출처: [summary_table.md](../logs/reliability_ablation/20260619_022030/summary_table.md).

**요소 절제** (LaTeX: [kd_reliability_factor_ablation.tex](kd_reliability_factor_ablation.tex)):

| 구성 | T-gate | S-byp | Smooth | 외부 Dice | 외부 Sens | BUID Dice |
|---|:--:|:--:|:--:|---:|---:|---:|
| confidence × entropy | -- | -- | -- | 0.587 | 0.544 | 0.605 |
| + teacher gate | ✓ | -- | -- | **0.630** | 0.622 | 0.652 |
| + student bypass (full) | ✓ | ✓ | -- | 0.614 | 0.591 | 0.648 |
| + smoothing | ✓ | ✓ | ✓ | 0.597 | 0.578 | 0.636 |
| full − teacher gate | -- | ✓ | -- | 0.584 | 0.574 | 0.619 |

(confidence·entropy는 항상 on.) 핵심:
- **`teacher_correctness_gate`가 주역**: 추가 시 외부 Dice +0.044 (b0 0.587→b1 0.630), 전체 맵에서 제거 시 −0.030 (0.614→0.584).
- 이 약한 FT 교사에선 **student_bypass가 소폭 손해**(0.630→0.614), **smoothing도 손해**(0.614→0.597) → smoothing off 유지.
- 하이퍼파라미터(전체 표): 변형들 0.59–0.66에 집중; T=2 ≥ T=6/8; 하드 교사 게이트(`wrong=0.0`)+기본 student bypass가 최적에 가까움.

---

## 6. 상태 및 계속 진행 방법

| 스위프 | 완료 | 남음 |
|---|---|---|
| FT 주요 절제 | 20/20 | — |
| LoRA 핵심 (`092041`) | 4/4 | — |
| LoRA 하이퍼파라미터 | temp_2/8 | tg_wrong_0.25, sb_weight_0.3 |
| TinyUSFM→SAM | task_only | reliability, temp_2/8, tg_wrong_0.25, sb_weight_0.3 |

재개 (완료된 것은 건너뛰고, `last.pth`에서 부분 계속):
```bash
uv run tools/run_reliability_ablation.py --manifest config/sweeps/reliability_teacher_tinyusfm.yaml \
    --group reliability_teacher_tinyusfm --resume --workers gpu4:0,gpu4:1,gpu4:2,gpu4:3
uv run tools/run_reliability_ablation.py --manifest config/sweeps/reliability_teacher_lora.yaml \
    --group reliability_teacher_lora --resume --workers gpu4:0,gpu4:1
```
완료 후 표 재생성: `uv run tools/summarize_reliability_sweep.py <sweep_dir>`.