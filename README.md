# Sleep Apnea Self-Learner

> iPhone 야간 오디오로 코골이·무호흡 의심 구간을 검출하고 AHI(Apnea-Hypopnea Index)를 추정하는 on-device ML 자가학습 앱.
>
> **Status**: Phase 0 — Project setup. 4-month roadmap (MVP 12w + v1 4w).
>
> **Disclaimer**: 의료 진단 목적 아님. 분석 보조 도구.

---

## 왜 만드는가

수면다원검사(PSG)는 정확하지만 1박 입원·고비용. 가정용 wearable은 SpO2/심박 정도까지만 다루고, 코골이·무호흡 음향 분석은 빈약하다.

이 프로젝트는 **iPhone 마이크 + on-device CoreML**로 PSG의 음향 분석 부분을 부분 대체하고, 사용할수록 본인 호흡·환경에 적응하는 **자가학습 루프**를 갖는다. LLM은 사용하지 않는다 — 순수 ML 시스템 설계가 목표.

---

## 확정 결정

| 항목 | 결정 |
|---|---|
| 플랫폼 | iPhone 네이티브 앱 (Swift + SwiftUI) |
| ML 추론 | CoreML on-device |
| 모델 학습 | PyTorch → coremltools → CoreML |
| MVP 분석 범위 | 코골이 감지·강도 + 무호흡 의심 구간 + AHI 추정 |
| 일정 | MVP 12주 + v1 4주 = 약 4개월 |
| 자가학습 | Hybrid (on-device head 업데이트 + 월 1회 Mac 백본 재학습) |

---

## 기술 스택

**모델 아키텍처**
- 백본: **PANN CNN14** (AudioSet 사전학습). wav2vec2/AST는 80M+ 파라미터로 야간 추론 발열·배터리 부담. PANN은 양자화 후 ~20MB, coremltools 8.x 변환 안정적.
- 보조: 1D CNN (10초 윈도우 호흡 정지 검출, 5MB).

**데이터셋**
- 사전학습: AudioSet snoring/breathing 카테고리(~4시간) + ESC-50(환경음 negative).
- Fine-tune 1차: **PSG-Audio** (PhysioNet, 287명 PSG 동기 오디오, CC-BY) — 코골이·무호흡 라벨 신뢰도 최고.
- Fine-tune 2차: SHHS의 호흡 이벤트 어노테이션 (오디오 없음, 시간 분포 prior로만).
- DREAMS Apnea는 신호 위주, MVP 제외.

**학습 인프라**: PyTorch 2.x + torchaudio, Mac MPS 또는 Colab A100. 변환은 coremltools. 실험 트래킹은 W&B 무료 티어.

**앱 인프라**: SwiftUI + AVAudioEngine(녹음) + CoreML + Accelerate(추론) + Core Data(이벤트 로그) + HealthKit(수면 단계·SpO2 read-only).

---

## 자가학습 루프

CoreML `MLUpdateTask`는 마지막 fully-connected 레이어 또는 k-NN classifier head만 업데이트 가능 → **2계층 구조**.

1. **Frozen PANN backbone** — 임베딩 추출만, 고정.
2. **Personal head** — 작은 MLP 또는 prototype 분류기, on-device로 매일 점진 업데이트.

**데이터 흐름**
1. 야간: AVAudioEngine 16kHz 모노 녹음 → 30초 청크 → on-device PANN 추론 → 이벤트 후보 + 임베딩 저장. **원음은 암호화 후 7일 자동 삭제**, 임베딩만 영구 보존.
2. 기상: 사용자 1탭 라벨(잘 잠/보통/푹 못 잠) + HealthKit 수면 시간 + SpO2 dip 자동 매칭.
3. 매일 새벽: 신규 임베딩 + 라벨로 personal head 5 epoch 업데이트.
4. 월 1회: 사용자 명시 동의 시 임베딩(원음 X)을 Mac으로 sync → PyTorch에서 백본 fine-tune → 신규 .mlmodel 생성 → 앱 업데이트로 배포.

**라벨링 UX**: 기상 알람 해제 시 컨디션 이모티콘 5개 + "이 코골이가 본인 맞나요?" 샘플 클립 1개 yes/no 1탭. 30초 이내 완료 목표.

**모델 배포**: MVP는 앱 번들 동봉. v1부터 on-demand resources로 .mlmodel 원격 갱신.

---

## MVP 마일스톤 (W1–W12)

| 주차 | 작업 | 검증 |
|---|---|---|
| W1-2 | PSG-Audio 다운로드, 라벨 EDA, 30초 윈도우 분할 파이프라인 | 데이터 통계 노트북 |
| W3-4 | PANN fine-tune 1차 (코골이 binary) | holdout F1 ≥ 0.85 |
| W5 | 무호흡 1D CNN 학습, AHI 회귀 | AHI MAE ≤ 5 |
| W6 | coremltools 변환 + int8 양자화, iPhone 추론 벤치 | 30초 청크 < 200ms, RAM < 150MB |
| W7-8 | SwiftUI 앱: 녹음, 추론, 결과 화면, Core Data 저장 | 본인 침실 7일 연속 무중단 |
| W9 | 라벨링 UX + personal head 업데이트 로직 | 본인 데이터 7일 후 F1 향상 측정 |
| W10 | HealthKit 통합, SpO2 교차 검증 화면 | Apple Watch SpO2 dip ↔ 무호흡 detection 시각화 |
| W11-12 | TestFlight 베타, 본인 30일 + 지인 2-3명 검증 | 배터리 ≤ 15%/야간, 크래시 free |

## v1 (W13–W16)
- 백본 재학습 파이프라인 (Mac sync → PyTorch fine-tune → 모델 배포)
- 화자 분리 (pyannote → CoreML)
- 주간/월간 리포트
- on-demand resources로 .mlmodel 원격 업데이트

---

## 위험 매트릭스

| 위험 | 완화책 | 우선순위 |
|---|---|---|
| 의료기기 규제 (KFDA/FDA) | "진단 아님, 참고용" 약관·UI 명시. AHI 표기에 "추정" 강제. 의료 권고 문구 금지. | 최상 |
| 침실 녹음 개인정보 | 원음 7일 자동 삭제, 임베딩만 보존, iCloud 미사용, Mac sync 시 사용자 명시 동의 + AES-256 | 최상 |
| iOS 백그라운드 6-9시간 녹음 중단 | UIBackgroundModes audio + AVAudioSession .playAndRecord, 무음 keep-alive 트랙 | 상 |
| 배터리·발열 | 30초 청크 단위 추론, int8 양자화, Neural Engine 강제 | 상 |
| PSG ground truth 부재 | 공개 데이터셋 holdout + Apple Watch SpO2 dip 교차 검증 + Withings Sleep Analyzer 1회 비교 | 상 |
| CoreML on-device 학습 한계 | head-only 업데이트로 범위 한정, 백본은 월 1회 Mac 재학습 | 중 |
| 커플 침대 화자 혼동 | MVP는 단일 사용자 가정 명시. v1에서 pyannote 임베딩 클러스터링 도입 | 중 |

---

## 평가 지표

**모델 성능**
- 코골이 F1 ≥ 0.85 (PSG-Audio holdout)
- 무호흡 event-level recall ≥ 0.75 (IoU ≥ 0.5)
- AHI MAE ≤ 5 events/h, 중증도 4단계 분류 정확도 ≥ 70%
- 추론: iPhone 13 기준 30초 청크 < 200ms, 야간 배터리 ≤ 15%

**자가학습 효과**
- 사용 7일/30일 후 본인 데이터 F1 향상 곡선 (vs day 1)
- 라벨링 완료율 ≥ 60%

**사용자 가치**
- D7/D30 retention
- 주간 리포트 열람률
- HealthKit SpO2 dip ↔ 무호흡 detection 일치율

---

## 디렉토리 구조 (계획)

```
SleepApnea/
├── training/
│   ├── data/
│   │   ├── psg_audio_loader.py        # PSG-Audio 라벨 파싱 + 30초 윈도우 분할
│   │   └── audioset_filter.py         # AudioSet snoring/breathing 필터링
│   ├── pann_finetune.py               # PyTorch fine-tune + W&B 트래킹
│   ├── apnea_1dcnn.py                 # 무호흡 1D CNN 학습
│   ├── export_coreml.py               # coremltools 변환 + int8 양자화 + head 분리
│   └── eval/
│       ├── ahi_metrics.py             # AHI MAE, 중증도 분류 평가
│       └── crossval_spo2.py           # Apple Watch SpO2 교차 검증
├── ios/SleepApneaSelfLearner/
│   ├── Audio/
│   │   ├── NightRecorder.swift        # AVAudioEngine 백그라운드 녹음 + 청크
│   │   └── BackgroundSessionManager.swift
│   ├── ML/
│   │   ├── InferenceEngine.swift      # CoreML 추론 + 임베딩 추출
│   │   ├── PersonalHeadUpdater.swift  # MLUpdateTask 기반 head 업데이트
│   │   └── AHIEstimator.swift
│   ├── Storage/
│   │   ├── EmbeddingStore.swift       # Core Data 임베딩 저장
│   │   └── AudioRetentionPolicy.swift # 원음 7일 자동 삭제
│   ├── Views/
│   │   ├── MorningLabelView.swift     # 1탭 라벨링 UX
│   │   ├── NightlyReportView.swift
│   │   └── TrendView.swift
│   └── Health/
│       └── HealthKitBridge.swift
└── README.md
```

---

## Verification

**모델 단계 (W1–W6)**
- `pytest training/tests/` — 데이터 로더·전처리 단위 테스트
- W&B 대시보드에서 코골이 F1, 무호흡 recall, AHI MAE 추이
- coremltools 변환 후 `xcrun coremlcompiler` 검증 + iPhone 13에서 30초 청크 추론 시간 측정

**앱 단계 (W7–W10)**
- 본인 침실에서 7일 연속 야간 녹음 무중단 (백그라운드 녹음 안정성)
- 매일 아침 결과 화면에 코골이 시간/강도, 무호흡 의심 횟수, 추정 AHI 표시
- HealthKit 수면 시간과 녹음 시간 일치
- Apple Watch SpO2 dip ↔ 무호흡 detection 시간 정합성

**자가학습 단계 (W9 이후)**
- 7일 / 30일 사용 후 personal head 적용 전후 본인 데이터 F1 비교
- 라벨링 완료율 60% 이상
- 월 1회 Mac sync → 백본 재학습 → 신규 .mlmodel 배포 사이클 1회 완주

**베타 단계 (W11–W12)**
- TestFlight 배포 + 지인 2-3명 30일 사용
- 야간 배터리 소모 ≤ 15%
- 크래시 0건 (Xcode Organizer 모니터링)

---

## 포트폴리오 어필 포인트

- **End-to-end ML 오너십**: 데이터 큐레이션 → 사전학습 활용 → fine-tune → 양자화 → 모바일 배포 → 자가학습 루프까지 단독 수행
- **on-device ML 깊이**: CoreML `MLUpdateTask` 한계 우회 설계 (frozen backbone + personal head), 양자화·Neural Engine 활용
- **자가학습 시스템 설계**: hybrid online/offline 재학습, label-efficient UX, 모델 버저닝·배포 전략
- **헬스케어 규제·개인정보 의식**: 의료기기 표현 회피, on-device 우선 아키텍처
- **검증 방법론**: PSG ground truth 부재 환경에서 공개 데이터셋 + 웨어러블 교차 검증 설계

면접 토픽 후보: "왜 wav2vec2 대신 PANN인가", "MLUpdateTask로 어디까지 학습 가능한가", "AHI MAE 5의 임상적 의미와 한계".
