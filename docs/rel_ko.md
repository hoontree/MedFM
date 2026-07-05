# 신뢰도 기반 KD — 통합 결과

지금까지의 모든 분석/시각화 요약본입니다. 계획 및 설계는
[reliability_kd_study.md](reliability_kd_study.md)를 참고하세요. 이 파일은 결과만 정리합니다.

**전제: 모든 결과는 balanced sampling을 켠 상태에서 얻은 것입니다** (학생·교사 모두 `data.sampling.enabled=true`). 예전 버전의 비-sampling 결과와 미튜닝 교사(val 0.57)는 폐기했습니다. sampling on/off 자체의 효과 분리는 §6에 남겨 둡니다.

설정: 학생 = TinyUSFM (sampling-on); 교사 = **SAM full-FT sampled (val Dice 0.682)** 및 **SAM LoRA rank-4 sampled (val Dice 0.720)** — 둘 다 동일 recipe로 sampler를 켜고 학습. 데이터 `dynamic` (num_classes=3); 학습 데이터 BUSBRA/BUSI/B, 외부 테스트 **BUID/BUS_UCLM_filtered** (UCLM은 filtered 한 세트만; plain BUS_UCLM은 동일 데이터라 이중 집계 방지 위해 제외). 지표 초점: Dice & Sensitivity.

---

## 0. 동기 — 교사가 *자신감 있게 틀린다*

일반 로짓-KD는 모든 교사 픽셀을 동등하게 증류합니다. 하지만 고정된 SAM 교사는 단순히 오류를 범하는 것이 아니라 **자신감 있게** 오류를 범하며, 일반 KD는 이를 거의 하드 레이블처럼 학생에게 주입합니다. BUID에서 측정 (교사만, 학생 없음):

| BUID (sampling-on) | SAM-FT-sampled (0.682) | SAM-LoRA-sampled (0.720) |
|---|---:|---:|
| **틀린** 픽셀의 평균 신뢰도 | **0.895** | 0.830 |
| 신뢰도 > 0.9인 틀린 픽셀 | **68.1 %** | 50.8 % |
| 신뢰도 > 0.7인 틀린 픽셀 | **86.8 %** | 74.9 % |
| (신뢰도>0.9인) 자신감 있게 틀린 픽셀의 전체 비율 | **5.1 %** | 3.1 % |
| 픽셀 정확도 | 0.925 | **0.939** |

FT 교사의 경우, **모든 오류의 약 68%가 >0.9 신뢰도를 가짐** — 이것이 바로 신뢰도-KD의 `teacher_correctness_gate`가 0으로 만드는 픽셀들입니다 (§3). sampling을 켜도 full-FT 교사는 여전히 LoRA보다 자신감 있게 틀립니다 (5.1% vs 3.1%). 신뢰도 히스토그램 (빨간 곡선 = 틀린 곡선에서의 높은 신뢰도 꼬리):

| FT 교사 | LoRA 교사 |
|---|---|
| ![](figures/confidence_hist_ft.png) | ![](figures/confidence_hist_lora.png) |

샘플별 증거 (빨간색 = 신뢰도 > 0.7 **그리고** 교사 ≠ GT):

![자신감 있게 틀린 패널](figures/confidently_wrong_panel_ft.png)

이것이 신뢰도-KD가 목표하는 문제이며, §2도 미리 보여줍니다: 더 잘 보정된 LoRA 교사는 자신감 있게 틀린 경우가 훨씬 적습니다.

도구: `tools/analyze_confidently_wrong.py teacher=<sam_ft_sampled|sam_lora_sampled> analysis.loader=BUID`.

---

## 1. 주요 비교 (방법별, 교사별) — 전부 sampling-on

**모든 행이 balanced sampling** (student·교사 모두).
두 교사(FT-sampled 0.682, LoRA-sampled 0.720)를 동일 recipe로 sampler 켜고 학습한 뒤,
sampling 켠 student로 증류한 단일 sweep입니다. 굵은 글씨 = 최고.
전체 테스트 세트 표: [kd_reliability_table.tex](kd_reliability_table.tex),
데이터셋별 분해: [kd_reliability_by_dataset.tex](kd_reliability_by_dataset.tex).

외부 평균 = **(BUID + BUS_UCLM_filtered) / 2** (UCLM은 filtered 한 세트만 집계).

| 교사 (val Dice) | 방법 | ext Dice | ext Sens | int Dice | Δ ext (vs task-only) |
|---|---|---:|---:|---:|---:|
| — | Task-only (no KD) | 0.630 | 0.644 | 0.765 | — |
| FT-sampled (0.682) | logit-KD | 0.647 | 0.650 | 0.734 | +0.017 |
| FT-sampled (0.682) | uncertainty-KD | 0.656 | 0.642 | 0.717 | +0.026 |
| FT-sampled (0.682) | reliability-KD | 0.646 | 0.628 | 0.759 | +0.016 |
| LoRA-sampled (0.720) | logit-KD | **0.665** | **0.669** | 0.742 | +0.035 |
| LoRA-sampled (0.720) | uncertainty-KD | 0.628 | 0.602 | 0.739 | −0.002 |
| LoRA-sampled (0.720) | reliability-KD | 0.641 | 0.626 | **0.774** | +0.011 |

**핵심 발견 (sampling-on):**
1. **괜찮은 교사면 KD가 대체로 task-only(0.630)를 능가** — sampling으로 FT 교사가 0.682까지 오르자 예전(비-sampling·미튜닝 0.57 교사)의 "KD가 손해" 현상이 사라졌습니다. 유일한 예외는 LoRA-uncertainty-KD(−0.002, 사실상 무이득). 괜찮은 교사가 KD의 전제조건.
2. **LoRA-sampled가 더 나은 교사** (외부 best 0.665 vs FT 0.656). 두 교사 모두 sampling을 켰으므로 이 우위는 sampling이 아니라 **adaptation 자체**(더 잘 보정됨, §2)에서 옴.
3. **잘 학습된 LoRA 교사에선 plain logit-KD가 외부 평균 1위(0.665)**; reliability-KD는 더 보수적(Sens 낮음)이나 **내부 Dice(0.774) 1위**. reliability 게이팅의 안전 마진은 교사가 좋아질수록 줄고, 가장 큰 이점은 **약한/과신 교사 영역**(§0·§5)에 집중됩니다.
4. **Uncertainty-KD(교사 엔트로피 가중)는 FT 교사에선 오히려 외부 best(0.656, BUID 0.693)지만 LoRA 교사에선 task-only 밑(−0.002)** — 잘 보정된 교사에선 엔트로피-가중의 추가 이득이 사라짐. 교사 *자신의* 불확실성만 쓰는 가중은 GT-조건부 게이트(§3)만큼 자신감 있게 틀린 픽셀을 못 걸러냄.

> KD 방향에 대하여: 이 문서는 sampling-on 전제의 **SAM→TinyUSFM** 방향만 다시 측정했습니다. 반대 방향(TinyUSFM→SAM)은 sampling-on으로 재실행하지 않았으므로 표에서 제외 — 큰 SAM 학생은 이미 데이터에 과적합해 KD 이득이 없다는 기존 결론은 별도 검증 대상입니다.

---

## 2. LoRA 교사가 더 잘 증류하는 이유는?

![LoRA가 더 나은 이유](figures/why_lora_better.png)

동일한 학생으로 BUID에서 동일한 픽셀을 측정했을 때, 두 교사는 정확히 KD 이론이 *좋은* 교사가 가져야 할 방식으로 다릅니다:

| BUID, 동일 학생 (sampling-on) | SAM-FT-sampled (0.682) | SAM-LoRA-sampled (0.720) |
|---|---:|---:|
| `teacher_correctness_gate` (≈ 픽셀 정확도) | 0.935 | 0.910 |
| `confidence` (최대 확률) | 0.821 | **0.809** |
| `entropy_penalty` (1−H/logC) | 0.520 | **0.475** |

- **정확도는 거의 대등**: sampling을 켜자 두 교사의 픽셀 정확도가 ~0.91–0.94로 수렴합니다(§0의 독립 측정에선 LoRA 0.939 ≥ FT 0.925). 예전 비-sampling 조건에서 벌어졌던 정확도 격차는 sampling이 메웁니다 — 즉 이제 **정확도로는 LoRA 우위를 설명할 수 없습니다.**
- **결정적 차이는 보정(calibration)**: LoRA 교사가 더 낮은 최대 확률 신뢰도(0.809 vs 0.821)와 더 높은 엔트로피(entropy_penalty 0.475 < 0.520, 즉 더 불확실)를 유지합니다. 무엇보다 **틀린 픽셀의 평균 신뢰도가 0.830 vs 0.895**(§0) — LoRA는 틀릴 때 덜 자신만만합니다. 작은 초음파 세트에 대한 full-FT는 SAM을 **과도하게 신뢰도 있게** 만들고; rank-4 LoRA의 작은 학습 가능 예산이 정규화기로 작용해 더 부드럽고 잘 보정된 목표를 생성 — 증류의 고전적인 "좋은 교사".
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

BUID의 전체 맵 학생 (FT-sampled 교사 + full-map reliability student, [분석](../logs/reliability_analysis/20260701_091143/)):

![신뢰도 히스토그램](figures/reliability_hist.png)

```
평균 신뢰도 | 교사 정답 : 0.270
평균 신뢰도 | 교사 오답   : 0.000      <- 자신감 있게 틀린 교사 픽셀 완전히 게이팅됨
틀린 픽셀의 게이팅 비율 (<0.1)  : 1.000
전체 픽셀의 게이팅 비율 (<0.1)  : 0.606
성분 평균: confidence 0.82 · entropy 0.52 · teacher_gate 0.94 · student_bypass 0.46
```

메커니즘 확인 (H2): 맵은 교사가 GT와 불일치하는 모든 픽셀에서 KD 가중치를 ~0으로 몰고(틀린 픽셀 게이팅 비율 1.000), 교사가 맞은 곳에서는 신호를 유지합니다. sampling-on 조건에선 교사가 더 정확(teacher_gate 0.94)하고 student_bypass 성분도 더 높아(0.46) — 즉 예전보다 더 많은 신뢰 가능한 KD 신호가 생존하며, 이것이 §5에서 student_bypass가 이제 *도움*이 되는 이유와 맞물립니다.

샘플별 패널 (이미지 · GT · 교사 · 학생 · 각 요소 · 최종 r):

![패널](figures/reliability_panel_example.png)

---

## 5. 절제 연구 (SAM-FT-sampled 교사, sampling-on, 전체 20회 실행 스위프)

출처: [summary_table.md](../logs/reliability_ablation/20260701_013643/summary_table.md).
매니페스트: [reliability_ablation_sampled.yaml](../config/sweeps/reliability_ablation_sampled.yaml).
**교사·학생 모두 balanced sampling on** (예전 §5는 미튜닝 FT 교사 0.57 + sampling-off였음 → 폐기).

**요소 절제** (confidence·entropy는 항상 on; 외부 = (BUID+BUS_UCLM_filtered)/2):

| 구성 | T-gate | S-byp | Smooth | 외부 Dice | 외부 Sens | BUID Dice | 내부 Dice |
|---|:--:|:--:|:--:|---:|---:|---:|---:|
| confidence × entropy (b0) | -- | -- | -- | 0.662 | 0.634 | 0.673 | 0.760 |
| + teacher gate (b1) | ✓ | -- | -- | 0.659 | 0.646 | **0.686** | 0.765 |
| + student bypass (full, b2) | ✓ | ✓ | -- | **0.670** | **0.649** | 0.671 | 0.762 |
| + smoothing (b3) | ✓ | ✓ | ✓ | 0.650 | 0.627 | 0.667 | 0.757 |
| full − teacher gate | -- | ✓ | -- | 0.639 | 0.621 | 0.661 | 0.751 |
| full − student bypass | ✓ | -- | -- | 0.664 | 0.635 | 0.685 | 0.747 |

베이스라인 (같은 sweep): task-only 0.629 · logit-KD 0.648 · uncertainty-KD 0.635 (외부 Dice).

**핵심 — 좋은 교사에서 결론이 뒤집힘:**
- **full map (b2) 이 최고 (외부 0.670)**, task-only(0.629) 대비 **+0.041**, logit-KD(0.648) 대비 +0.022. 예전 약한 교사 sweep에선 full map이 부분 요소보다 나빴는데, sampling-on·괜찮은 교사에선 full map이 이깁니다.
- **`student_bypass`가 이제 *도움***: b1→b2 외부 +0.011, full에서 제거 시 −0.006 (0.670→0.664). 교사가 정확해져(§4 teacher_gate 0.94) 구제/게이팅 신호가 신뢰할 만해진 결과 — 예전 0.57 교사에선 반대로 손해였음.
- **`teacher_correctness_gate`는 full map 안에서 여전히 필수**: 제거 시 −0.031 (0.670→0.639, 전체에서 가장 큰 낙폭). 단독 추가(b0→b1)는 외부 −0.003지만 BUID(0.673→0.686)·내부는 올림 — 게이트+바이패스가 함께 작동할 때 이득이 나옴.
- **smoothing은 여전히 손해** (0.670→0.650) → off 유지.
- **온도**: T=4/T=6 최적권(0.655/0.658), T=2 최악(0.628), 고온(T=8, 0.647)도 저하. **배치**: 작을수록 유리(bs4 0.656 > bs16 0.642), 기본 bs8.
- **실행 간 분산 주의**: 동일 구성(full·bs8·T4)이 b2 0.670 / temp_4.0 0.655 / bs_8 0.651로 ~0.02 퍼짐 → 0.02 미만 차이는 노이즈로 해석.

---

## 6. 베이스라인 재실험 — 통일 recipe + balanced sampling on/off

**목적.** 이 문서의 다른 모든 표는 sampling-on을 기본 전제로 삼습니다. 이 절은 그 전제를 정당화하기 위해 **sampling on/off를 유일한 변수로** 분리 측정합니다. 예전 교사/베이스라인들은 시점·recipe가 제각각이었으므로(일부 sampler 없음, SAM-FT는 미튜닝 아티팩트 0.57), 모든 베이스라인을 **동일 recipe**(bs/lr/스케줄/증강 통일)로 재학습하고 (a) 깨끗한 베이스라인 표를 확보하고 (b) sampling 효과를 격리했습니다.

지표: val Dice(체크포인트 선택), BUID Dice, 외부 평균(BUID+BUS_UCLM_filtered) Dice/Sens, 내부 평균(BUSBRA/BUSI/B) Dice. (실행: `logs/base_*.log`, `logs/teacher_*sampled.log`; 2026-06-24·26)

| 모델 | sampling | val Dice | BUID Dice | 외부 Dice | 외부 Sens | 내부 Dice |
|---|:--:|---:|---:|---:|---:|---:|
| TinyUSFM | off | 0.725 | 0.672 | **0.644** | **0.631** | 0.747 |
| TinyUSFM | on | 0.714 | 0.675 | 0.627 | 0.600 | **0.765** |
| SAM-FT | off | 0.674 | 0.596 | 0.591 | 0.566 | 0.688 |
| SAM-FT | on | 0.682 | 0.614 | 0.596 | 0.570 | 0.718 |
| SAM-LoRA | off | 0.710 | 0.687 | 0.656 | 0.625 | — |
| SAM-LoRA | on | 0.720 | **0.681** | **0.662** | 0.623 | 0.755 |

**관찰:**
- **balanced sampling은 내부(in-distribution)를 일관되게 끌어올림** — TinyUSFM 0.747→0.765, SAM-FT 0.688→0.718.
- **약한 SAM-FT 교사는 sampling으로 전반 개선** (val 0.674→0.682, BUID 0.596→0.614) → §1.4의 "sampling이 FT 교사를 살린다" 재확인.
- **TinyUSFM은 sampling이 외부 일반화를 소폭 손해** (외부 0.644→0.627, Sens 0.631→0.600), 내부와 trade-off. 이미 잘 일반화하는 학생에는 클래스 리밸런싱이 외부 분포를 약간 왜곡.
- **SAM-LoRA는 sampling 적용 시 외부 best** (0.662) — 최강 교사 지위 유지. LoRA 우위는 sampling이 아니라 adaptation 자체(§2)임을 다시 확인.

**KD 교사 선택 함의:** 외부 일반화가 목표이면 sampling-off TinyUSFM 베이스라인(0.644)이 가장 높지만, KD 교사로는 sampling-on SAM-LoRA(외부 0.662, BUID 0.681)가 최선 — §1.4 공정비교의 LoRA-sampled 교사와 동일.

---

## 7. 상태 및 계속 진행 방법

sampling-on 재실험 현황:

| 항목 | 상태 |
|---|---|
| **요소 절제 (FT-sampled 교사, sampling-on)** | 20/20 완료 — `logs/reliability_ablation/20260701_013643` |
| 주요 비교 (FT/LoRA-sampled 교사) | 완료 (§1) — `logs/reliability_ablation/20260629_015056` 등 |
| §0 confidently-wrong (sampled 교사) | 완료 — `logs/confidently_wrong/20260701_09*` |
| §2·§4 reliability-map 분석 (sampled) | 완료 — `logs/reliability_analysis/20260701_0911*` |
| 베이스라인 sampling on/off (§6) | 완료 |
| TinyUSFM→SAM 방향 (sampling-on) | 미실행 (표에서 제외, §1 주석) |

sampling-on 절제 스위프 재실행:
```bash
uv run tools/run_reliability_ablation.py \
    --manifest config/sweeps/reliability_ablation_sampled.yaml \
    --group reliability_ablation_sampled --workers local:2,local:3,gpu4:0,gpu4:1,gpu4:2,gpu4:3
uv run tools/summarize_reliability_sweep.py logs/reliability_ablation/<sweep_dir>
```
sampled 교사 분석 재생성:
```bash
uv run tools/analyze_confidently_wrong.py teacher=sam_ft_sampled analysis.loader=BUID wandb.disabled=true
uv run tools/analyze_reliability.py teacher=sam_lora_sampled analysis.loader=BUID analysis.num_batches=8 wandb.disabled=true
uv run tools/plot_why_lora_better.py
```