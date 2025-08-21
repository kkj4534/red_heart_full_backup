# Red Heart AI 워크플로우 검증 문서
작성일: 2025-08-21

## 1. 전체 워크플로우

### 1.1 실행 명령어
```bash
# 기본 명령어 (수정 전)
bash run_learning.sh unified-test --samples 3 --no-param-update --debug --verbose

# LR 스윕 포함 명령어 (수정 후) ✅
bash run_learning.sh unified-test --lr-sweep --samples 3 --no-param-update --debug --verbose

# nohup으로 백그라운드 실행
nohup timeout 3600 bash run_learning.sh unified-test --lr-sweep --samples 3 --no-param-update --debug --verbose > test_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

### 1.2 실행 흐름

#### Stage 1: 초기화
1. **run_learning.sh 실행**
   - unified-test 모드 진입
   - --lr-sweep 옵션 확인 ✅
   - training/run_hierarchical_lr_sweep.py 실행

2. **환경 설정**
   - venv/conda 환경 활성화
   - GPU 메모리 확인 및 초기화
   - 로깅 설정

#### Stage 2: 데이터 준비
3. **데이터 로드**
   - claude_preprocessed_complete.json 또는 .embedded.json 확인
   - 임베딩 상태 체크
     - 있으면: 재사용 ✅
     - 없으면: SentenceTransformer로 생성 ✅
   - 생성된 임베딩 .embedded.json에 저장 ✅

#### Stage 3: LR 스윕
4. **계층적 LR 스윕 (5-5-5-5)**
   - lr_sweep_cumulative.json 로드 (기존 결과) ✅
   - 중복 LR 체크 및 10% 간격 조정 ✅
   - Stage 0-4 순차 실행 (총 25포인트)
   - 각 LR은 독립적으로 초기 가중치에서 시작
   - 결과 누적 저장 ✅
   - Stage별 PNG 생성 ✅

5. **최적 LR 선택**
   - optimal_lr.json 저장
   - 최적 LR로 본 학습 준비

#### Stage 4: 본 학습
6. **모델 초기화**
   - UnifiedModel 생성
   - 730M 파라미터 검증 ✅
     - 목표: 730M
     - 실제 카운트 및 차이 표시
     - 모듈별 상세 분석
   - GPU 메모리 배치

7. **학습 실행**
   - 60 에폭 학습 (--samples로 제한 가능)
   - 파라미터 업데이트 모니터링 ✅
     - 100배치마다 업데이트 확인
     - 미업데이트 모듈 경고
   - Gradient norm 추적
   - 체크포인트 저장

#### Stage 5: 분석
8. **Sweet Spot 분석**
   - 모듈별 최적 에폭 탐색
   - 5가지 고급 분석 기법 적용

9. **Parameter Crossover**
   - 최적 에폭의 파라미터 결합
   - crossover_final.pth 저장

10. **종료**
    - 임베딩 저장 확인 ✅
    - 학습 곡선 export
    - OOM 통계 저장
    - 최종 리포트 생성

## 2. 핵심 파일 및 기능

### 2.1 수정된 파일
- ✅ **run_learning.sh**: --lr-sweep 옵션 추가
- ✅ **unified_training_final.py**: 
  - SentenceTransformer 임베딩 통합
  - 파라미터 업데이트 모니터링
  - 730M 검증 강화
- ✅ **run_hierarchical_lr_sweep.py**: 실제 데이터 사용
- ✅ **hierarchical_lr_sweep.py**: 
  - 누적 결과 저장/로드
  - 중복 LR 10% 조정

### 2.2 생성되는 파일
- `lr_sweep_cumulative.json`: 모든 LR 테스트 누적 결과
- `optimal_lr.json`: 최적 LR 정보
- `*.embedded.json`: 임베딩이 포함된 데이터셋
- `hierarchical_lr_sweep_stage*.png`: Stage별 시각화
- `checkpoints/`: 에폭별 체크포인트
- `crossover_final.pth`: Parameter Crossover 결과

## 3. 예상 문제점 및 해결책

### 3.1 메모리 문제
- **문제**: 730M 모델 + 임베딩 생성 시 OOM
- **해결**: 
  - 배치 사이즈 자동 조정
  - 임베딩 배치 처리
  - GPU/CPU 하이브리드 배치

### 3.2 임베딩 시간
- **문제**: 150,000개 샘플 임베딩 시간
- **해결**: 
  - 첫 실행만 생성, 이후 재사용
  - LR 스윕은 일부 샘플만 사용 (1000개)

### 3.3 파라미터 누락
- **문제**: neural_analyzers, advanced_wrappers 미사용
- **해결**: 
  - nn.ModuleDict 사용 ✅
  - 파라미터 업데이트 모니터링 ✅
  - Optimizer 등록 확인 ✅

## 4. 검증 체크리스트

- [x] 명령어에 --lr-sweep 인자 포함
- [x] 임베딩 생성/재사용 로직
- [x] LR 중복 체크 및 조정
- [x] 730M 파라미터 검증
- [x] 파라미터 업데이트 모니터링
- [x] 누적 결과 저장
- [ ] 실제 테스트 실행
- [ ] 로그 확인
- [ ] 결과 분석

## 5. 테스트 명령어

### 최소 테스트 (1 에폭, LR 스윕)
```bash
bash run_learning.sh unified-test --lr-sweep --samples 1 --no-param-update --debug --verbose
```

### 표준 테스트 (3 에폭, LR 스윕)
```bash
bash run_learning.sh unified-test --lr-sweep --samples 3 --no-param-update --debug --verbose
```

### 전체 학습 (60 에폭, 최적 LR 사용)
```bash
bash run_learning.sh unified-train --epochs 60 --lr $(cat training/lr_sweep_results/optimal_lr.json | grep optimal_lr | cut -d'"' -f4)
```

## 6. 모니터링 포인트

1. **임베딩 생성**
   - "📊 임베딩 상태:" 로그 확인
   - "✅ 임베딩이 저장되었습니다" 확인

2. **LR 스윕**
   - "📂 기존 LR 테스트 결과 로드" 확인
   - "⚠️ LR * 이미 테스트됨" 경고 확인
   - "💾 누적 결과 저장 완료" 확인

3. **파라미터 검증**
   - "✅ 목표 파라미터 수 달성: *M ≈ 730M" 또는
   - "⚠️ 파라미터 개수 불일치!" 경고 확인
   - "✅ 파라미터 업데이트됨" 로그 확인

4. **학습 진행**
   - Epoch별 loss 감소
   - Gradient norm 변화
   - 체크포인트 저장

---
작성: 2025-08-21
상태: 테스트 준비 완료