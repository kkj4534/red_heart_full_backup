"""
계층적 Learning Rate 스윕 구현
5-5-5-5 Coarse-to-Fine 최적화 전략
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class LRTestResult:
    """LR 테스트 결과"""
    lr: float
    stage: int
    epoch_losses: List[float]
    val_loss: float
    train_loss: float
    gradient_norm: float
    convergence_rate: float
    time_taken: float
    accuracy: float = 0.0
    

class HierarchicalLRSweep:
    """
    계층적 학습률 탐색 (5-5-5-5 전략)
    총 25개 포인트로 최적 LR 탐색
    """
    
    def __init__(self, 
                 test_epochs: int = 3,
                 test_steps: int = 50,
                 warmup_steps: int = 10,
                 output_dir: str = "training/lr_sweep_results"):
        """
        Args:
            test_epochs: 각 LR 테스트 에폭 수
            test_steps: 각 에폭당 스텝 수 
            warmup_steps: 워밍업 스텝
            output_dir: 결과 저장 디렉토리
        """
        self.test_epochs = test_epochs
        self.test_steps = test_steps
        self.warmup_steps = warmup_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 결과 저장
        self.all_results: Dict[int, List[LRTestResult]] = {
            0: [],  # Stage 0
            1: [],  # Stage 1
            2: [],  # Stage 2
            3: [],  # Stage 3
            4: []   # Stage 4
        }
        
        # 최적 LR 추적
        self.best_lr = None
        self.best_loss = float('inf')
        
        # Stage별 탐색 구간
        self.search_intervals: Dict[int, List[Tuple[float, float]]] = {}
        
        # 누적 결과 파일 경로
        self.cumulative_results_path = self.output_dir / "lr_sweep_cumulative.json"
        self.tested_lrs_history = self._load_cumulative_results()
        
        logger.info("=" * 70)
        logger.info("🎯 Hierarchical LR Sweep 초기화")
        logger.info(f"  - 전략: 5-5-5-5 (총 25개 포인트)")
        logger.info(f"  - 테스트 에폭: {test_epochs}")
        logger.info(f"  - 테스트 스텝: {test_steps}/epoch")
        logger.info(f"  - 기존 테스트 LR: {len(self.tested_lrs_history)}개")
        logger.info("=" * 70)
    
    def _load_cumulative_results(self) -> Dict[float, Dict[str, Any]]:
        """기존 누적 결과 로드"""
        if self.cumulative_results_path.exists():
            try:
                with open(self.cumulative_results_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"📂 기존 LR 테스트 결과 로드: {len(data)} 개 LR 기록")
                    # float 키로 변환
                    return {float(k): v for k, v in data.items()}
            except Exception as e:
                logger.warning(f"기존 결과 로드 실패: {e}")
                return {}
        return {}
    
    def _save_cumulative_results(self):
        """누적 결과 저장"""
        try:
            # 문자열 키로 변환 (JSON 호환성)
            save_data = {str(k): v for k, v in self.tested_lrs_history.items()}
            with open(self.cumulative_results_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"💾 누적 결과 저장 완료: {len(self.tested_lrs_history)} 개 LR")
        except Exception as e:
            logger.error(f"누적 결과 저장 실패: {e}")
    
    def _adjust_lr_if_duplicate(self, lr: float, tolerance: float = 0.1) -> float:
        """중복 LR 체크 및 조정 (10% 간격 추가)"""
        if lr in self.tested_lrs_history:
            # 이미 테스트한 LR이면 10% 간격 추가
            adjusted_lr = lr * (1 + tolerance)
            logger.warning(f"⚠️ LR {lr:.1e} 이미 테스트됨. {adjusted_lr:.1e}로 조정")
            # 재귀적으로 체크 (조정된 값도 중복일 수 있음)
            return self._adjust_lr_if_duplicate(adjusted_lr, tolerance)
        return lr
    
    def run_hierarchical_sweep(self,
                               model: nn.Module,
                               train_loader: Any,
                               val_loader: Any,
                               criterion: nn.Module,
                               device: torch.device) -> Dict[str, Any]:
        """
        계층적 LR 스윕 실행
        
        Returns:
            최종 결과 딕셔너리
        """
        logger.info("\n🚀 Hierarchical LR Sweep 시작...")
        start_time = time.time()
        
        # Stage 0: 초기 탐색 (5개 포인트)
        stage0_lrs = [1e-5, 5.6e-5, 3.2e-4, 1.8e-3, 1e-2]
        # 중복 체크 및 조정
        stage0_lrs = [self._adjust_lr_if_duplicate(lr) for lr in stage0_lrs]
        logger.info(f"\n📊 [Stage 0] 초기 탐색: {stage0_lrs}")
        
        for lr in stage0_lrs:
            result = self._test_single_lr(
                model, train_loader, val_loader, criterion, device, 
                lr, stage=0
            )
            self.all_results[0].append(result)
            
            # 최적 LR 업데이트
            if result.val_loss < self.best_loss:
                self.best_loss = result.val_loss
                self.best_lr = lr
                logger.info(f"  🏆 새로운 최적 LR: {lr:.1e} (loss: {result.val_loss:.4f})")
        
        # Stage 0 분석 및 상위 2개 구간 선택
        top_intervals = self._analyze_stage_results(0)
        logger.info(f"\n✅ Stage 0 완료. 선택된 구간: {top_intervals}")
        
        # Stage 1-4: 점진적 세분화
        for stage in range(1, 5):
            logger.info(f"\n📊 [Stage {stage}] 세분화 탐색")
            
            # 이전 stage의 상위 구간에서 새로운 포인트 생성
            stage_lrs = self._generate_stage_points(top_intervals, stage)
            logger.info(f"  - 탐색 포인트: {[f'{lr:.1e}' for lr in stage_lrs]}")
            
            # 각 LR 테스트
            for lr in stage_lrs:
                # 이미 테스트한 LR은 건너뛰기
                if self._already_tested(lr):
                    logger.info(f"  ⏭️ {lr:.1e} 이미 테스트됨, 건너뛰기")
                    continue
                    
                result = self._test_single_lr(
                    model, train_loader, val_loader, criterion, device,
                    lr, stage=stage
                )
                self.all_results[stage].append(result)
                
                # 최적 LR 업데이트
                if result.val_loss < self.best_loss:
                    self.best_loss = result.val_loss
                    self.best_lr = lr
                    logger.info(f"  🏆 새로운 최적 LR: {lr:.1e} (loss: {result.val_loss:.4f})")
            
            # 다음 stage를 위한 상위 구간 선택
            if stage < 4:
                top_intervals = self._analyze_stage_results(stage)
                logger.info(f"  ✅ Stage {stage} 완료. 다음 구간: {top_intervals}")
        
        # 누적 결과에 추가
        for stage_results in self.all_results.values():
            for result in stage_results:
                self.tested_lrs_history[result.lr] = {
                    'val_loss': result.val_loss,
                    'train_loss': result.train_loss,
                    'accuracy': result.accuracy,
                    'timestamp': datetime.now().isoformat(),
                    'stage': result.stage
                }
        
        # 누적 결과 저장
        self._save_cumulative_results()
        
        # 최종 분석 및 보고서 생성
        total_time = time.time() - start_time
        final_report = self._generate_final_report(total_time)
        
        # 결과 저장
        self._save_results(final_report)
        
        # 시각화
        self._visualize_results()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 Hierarchical LR Sweep 완료!")
        logger.info(f"  - 최적 LR: {self.best_lr:.1e}")
        logger.info(f"  - 최적 Loss: {self.best_loss:.4f}")
        logger.info(f"  - 총 소요 시간: {total_time/60:.1f}분")
        logger.info(f"  - 테스트된 포인트: {self._count_total_tests()}개")
        logger.info("=" * 70)
        
        return final_report
    
    def _test_single_lr(self,
                       model: nn.Module,
                       train_loader: Any,
                       val_loader: Any,
                       criterion: nn.Module,
                       device: torch.device,
                       lr: float,
                       stage: int) -> LRTestResult:
        """단일 LR 테스트"""
        logger.info(f"\n  📌 LR {lr:.1e} 테스트 시작 (Stage {stage})")
        
        # 모델 복사 (원본 보존)
        test_model = self._copy_model(model, device)
        optimizer = torch.optim.AdamW(test_model.parameters(), lr=lr)
        
        epoch_losses = []
        start_time = time.time()
        
        # 테스트 에폭 실행
        for epoch in range(self.test_epochs):
            test_model.train()
            batch_losses = []
            grad_norms = []
            
            # 제한된 스텝만 실행
            for step, batch in enumerate(train_loader):
                if step >= self.test_steps:
                    break
                
                # Forward - 실제 모델 구조에 맞는 전체 손실 계산
                inputs = batch['input'].to(device)
                
                # 백본 통과
                backbone_outputs = test_model.backbone(inputs, return_all_tasks=True)
                features = backbone_outputs.get('emotion', inputs)
                
                # 모든 헤드의 손실 계산
                head_losses = []
                
                # 1. Emotion Head
                if hasattr(test_model, 'emotion_head') and 'emotion_label' in batch:
                    emotion_output = test_model.emotion_head(features)
                    emotion_pred = emotion_output['emotions'] if isinstance(emotion_output, dict) else emotion_output
                    emotion_target = batch['emotion_label'].to(device)
                    emotion_loss = test_model.emotion_head.compute_loss(emotion_pred, emotion_target)
                    head_losses.append(emotion_loss)
                
                # 2. Bentham Head
                if hasattr(test_model, 'bentham_head') and 'bentham_label' in batch:
                    bentham_output = test_model.bentham_head(features)
                    bentham_pred = bentham_output['bentham_scores'] if isinstance(bentham_output, dict) else bentham_output
                    bentham_target = batch['bentham_label'].to(device)
                    bentham_loss = test_model.bentham_head.compute_loss(bentham_pred, bentham_target)
                    head_losses.append(bentham_loss)
                
                # 3. Regret Head
                if hasattr(test_model, 'regret_head') and 'regret_label' in batch:
                    regret_output = test_model.regret_head(features)
                    regret_pred = regret_output['regret_score'] if isinstance(regret_output, dict) else regret_output
                    regret_target = batch['regret_label'].to(device)
                    regret_loss = test_model.regret_head.compute_loss(regret_pred, regret_target)
                    head_losses.append(regret_loss)
                
                # 4. SURD Head
                if hasattr(test_model, 'surd_head') and 'surd_label' in batch:
                    surd_output = test_model.surd_head(features)
                    surd_pred = surd_output['surd_values'] if isinstance(surd_output, dict) else surd_output
                    
                    # SURD 타겟 계산 (unified_training_final.py의 실제 구현과 동일)
                    batch_size = surd_pred.shape[0]
                    surd_target = torch.zeros((batch_size, 4), device=device)
                    
                    # Synergy: 감정 다양성 (엔트로피 기반)
                    if 'emotion_label' in batch:
                        emotion_probs = torch.nn.functional.one_hot(batch['emotion_label'].to(device), num_classes=7).float()
                        emotion_entropy = -(emotion_probs * (emotion_probs + 1e-10).log()).sum(dim=1)
                        surd_target[:, 0] = emotion_entropy / np.log(7)  # 정규화
                    
                    # Unique: 레이블 고유성
                    if 'surd_label' in batch:
                        label_unique = torch.nn.functional.one_hot(batch['surd_label'].to(device), num_classes=5).float()
                        surd_target[:, 1] = label_unique.max(dim=1)[0]
                    
                    # Redundant: 벤담 상관도
                    if 'bentham_label' in batch:
                        bentham = batch['bentham_label'].to(device)
                        bentham_mean = bentham.mean(dim=1)
                        bentham_std = bentham.std(dim=1) + 1e-10
                        surd_target[:, 2] = 1.0 - (bentham_std / (bentham_mean + 1e-10)).clamp(0, 1)
                    
                    # Deterministic: 후회 결정성
                    if 'regret_label' in batch:
                        regret = batch['regret_label'].to(device)
                        if regret.dim() == 1:
                            regret = regret.unsqueeze(1)
                        surd_target[:, 3] = regret.abs().squeeze()
                    
                    surd_loss = test_model.surd_head.compute_loss(surd_pred, surd_target)
                    head_losses.append(surd_loss)
                
                # 전체 손실 = 모든 헤드 손실의 평균
                if head_losses:
                    loss = sum(head_losses) / len(head_losses)
                else:
                    # fallback 없음 - 에러 발생
                    raise RuntimeError("No head losses computed - model structure error")
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient norm 계산
                total_norm = 0.0
                for p in test_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(test_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                batch_losses.append(loss.item())
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            avg_grad_norm = np.mean(grad_norms)
            
            logger.info(f"    Epoch {epoch+1}: loss={epoch_loss:.4f}, grad_norm={avg_grad_norm:.4f}")
        
        # Validation
        test_model.eval()
        val_losses = []
        val_accs = []
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= 20:  # 빠른 검증
                    break
                    
                # Forward - 실제 모델 구조에 맞는 전체 손실 계산
                inputs = batch['input'].to(device)
                
                # 백본 통과
                backbone_outputs = test_model.backbone(inputs, return_all_tasks=True)
                features = backbone_outputs.get('emotion', inputs)
                
                # 모든 헤드의 손실 계산
                head_losses = []
                head_accuracies = []
                
                # 1. Emotion Head
                if hasattr(test_model, 'emotion_head') and 'emotion_label' in batch:
                    emotion_output = test_model.emotion_head(features)
                    emotion_pred = emotion_output['emotions'] if isinstance(emotion_output, dict) else emotion_output
                    emotion_target = batch['emotion_label'].to(device)
                    emotion_loss = test_model.emotion_head.compute_loss(emotion_pred, emotion_target)
                    head_losses.append(emotion_loss)
                    
                    # Accuracy 계산
                    preds = emotion_pred.argmax(dim=-1)
                    acc = (preds == emotion_target).float().mean().item()
                    head_accuracies.append(acc)
                
                # 2. Bentham Head
                if hasattr(test_model, 'bentham_head') and 'bentham_label' in batch:
                    bentham_output = test_model.bentham_head(features)
                    bentham_pred = bentham_output['bentham_scores'] if isinstance(bentham_output, dict) else bentham_output
                    bentham_target = batch['bentham_label'].to(device)
                    bentham_loss = test_model.bentham_head.compute_loss(bentham_pred, bentham_target)
                    head_losses.append(bentham_loss)
                    
                    # Regression accuracy (threshold-based)
                    acc = ((bentham_pred - bentham_target).abs() < 0.5).float().mean().item()
                    head_accuracies.append(acc)
                
                # 3. Regret Head
                if hasattr(test_model, 'regret_head') and 'regret_label' in batch:
                    regret_output = test_model.regret_head(features)
                    regret_pred = regret_output['regret_score'] if isinstance(regret_output, dict) else regret_output
                    regret_target = batch['regret_label'].to(device)
                    regret_loss = test_model.regret_head.compute_loss(regret_pred, regret_target)
                    head_losses.append(regret_loss)
                    
                    # Regression accuracy
                    acc = ((regret_pred - regret_target).abs() < 0.5).float().mean().item()
                    head_accuracies.append(acc)
                
                # 4. SURD Head (정보이론 기반 4차원 타겟)
                if hasattr(test_model, 'surd_head') and 'surd_label' in batch:
                    surd_output = test_model.surd_head(features)
                    surd_pred = surd_output['surd_values'] if isinstance(surd_output, dict) else surd_output
                    
                    # SURD 타겟을 실제 데이터에서 계산 (4차원)
                    batch_size = surd_pred.shape[0]
                    surd_target = torch.zeros((batch_size, 4), device=device)
                    
                    # Synergy: 감정 다양성 (엔트로피 기반)
                    if 'emotion_label' in batch:
                        emotion_probs = F.one_hot(batch['emotion_label'].to(device), num_classes=7).float()
                        emotion_entropy = -(emotion_probs * (emotion_probs + 1e-10).log()).sum(dim=1)
                        surd_target[:, 0] = emotion_entropy / np.log(7)  # 정규화
                    
                    # Unique: 레이블 고유성 (one-hot 인코딩)
                    if 'surd_label' in batch:
                        label_unique = F.one_hot(batch['surd_label'].to(device), num_classes=5).float()
                        surd_target[:, 1] = label_unique.max(dim=1)[0]  # 최대값 = 1.0
                    
                    # Redundant: 벤담 상관도 (평균과 분산)
                    if 'bentham_label' in batch:
                        bentham = batch['bentham_label'].to(device)
                        bentham_mean = bentham.mean(dim=1)
                        bentham_std = bentham.std(dim=1) + 1e-10
                        surd_target[:, 2] = 1.0 - (bentham_std / (bentham_mean + 1e-10)).clamp(0, 1)
                    
                    # Deterministic: 후회 결정성 (절대값)
                    if 'regret_label' in batch:
                        regret = batch['regret_label'].to(device)
                        if regret.dim() == 1:
                            regret = regret.unsqueeze(1)
                        surd_target[:, 3] = regret.abs().squeeze()
                    
                    surd_loss = test_model.surd_head.compute_loss(surd_pred, surd_target)
                    head_losses.append(surd_loss)
                    
                    # Multi-dimensional regression accuracy (threshold 기반)
                    acc = ((surd_pred - surd_target).abs() < 0.3).float().mean().item()
                    head_accuracies.append(acc)
                
                # 전체 손실 = 모든 헤드 손실의 평균
                if head_losses:
                    loss = sum(head_losses) / len(head_losses)
                    val_losses.append(loss.item())
                    
                    # 전체 accuracy = 헤드별 accuracy의 평균
                    if head_accuracies:
                        overall_acc = np.mean(head_accuracies)
                        val_accs.append(overall_acc)
                else:
                    raise RuntimeError("No head losses computed in validation")
        
        val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_acc = np.mean(val_accs) if val_accs else 0.0
        
        # Convergence rate 계산
        if len(epoch_losses) >= 2:
            convergence_rate = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0]
        else:
            convergence_rate = 0.0
        
        time_taken = time.time() - start_time
        
        # 메모리 정리
        del test_model
        torch.cuda.empty_cache()
        
        result = LRTestResult(
            lr=lr,
            stage=stage,
            epoch_losses=epoch_losses,
            val_loss=val_loss,
            train_loss=epoch_losses[-1] if epoch_losses else float('inf'),
            gradient_norm=avg_grad_norm,
            convergence_rate=convergence_rate,
            time_taken=time_taken,
            accuracy=val_acc
        )
        
        logger.info(f"    ✅ 완료: val_loss={val_loss:.4f}, acc={val_acc:.4f}, time={time_taken:.1f}s")
        
        return result
    
    def _copy_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """모델 복사 - 새로운 인스턴스 생성"""
        # deepcopy는 thread lock 문제가 있으므로 
        # 모델의 state_dict를 저장하고 새 모델에 로드하는 방식 사용
        
        # 원본 모델의 가중치 저장
        original_state = model.state_dict()
        
        # 새 모델 인스턴스 생성 (UnifiedModel 가정)
        from training.unified_training_final import UnifiedModel
        
        # config 추출 (UnifiedModel은 config를 가지고 있음)
        if hasattr(model, 'config'):
            config = model.config
        else:
            # 기본 config 생성
            from training.unified_training_final import UnifiedTrainingConfig
            config = UnifiedTrainingConfig()
        
        # 새 모델 생성
        model_copy = UnifiedModel(config, device=device)
        
        # 원본 가중치 로드
        model_copy.load_state_dict(original_state)
        model_copy.to(device)
        
        return model_copy
    
    def _analyze_stage_results(self, stage: int) -> List[Tuple[float, float]]:
        """
        Stage 결과 분석 및 상위 2개 구간 선택
        """
        results = self.all_results[stage]
        if not results:
            return []
        
        # 결과를 loss 기준으로 정렬
        sorted_results = sorted(results, key=lambda x: x.val_loss)
        
        # LR 값들을 정렬
        all_lrs = sorted([r.lr for r in results])
        
        # 상위 2개 구간 찾기
        intervals = []
        
        # 최고 성능 LR 주변 구간
        best_lr = sorted_results[0].lr
        best_idx = all_lrs.index(best_lr)
        
        # 구간 1: 최고 성능 LR 주변
        if best_idx > 0:
            intervals.append((all_lrs[best_idx-1], all_lrs[best_idx]))
        if best_idx < len(all_lrs) - 1:
            intervals.append((all_lrs[best_idx], all_lrs[best_idx+1]))
        
        # 구간이 부족하면 두 번째 좋은 LR 주변도 고려
        if len(intervals) < 2 and len(sorted_results) > 1:
            second_lr = sorted_results[1].lr
            second_idx = all_lrs.index(second_lr)
            
            if second_idx > 0 and (all_lrs[second_idx-1], all_lrs[second_idx]) not in intervals:
                intervals.append((all_lrs[second_idx-1], all_lrs[second_idx]))
            elif second_idx < len(all_lrs) - 1 and (all_lrs[second_idx], all_lrs[second_idx+1]) not in intervals:
                intervals.append((all_lrs[second_idx], all_lrs[second_idx+1]))
        
        # 최대 2개 구간만 반환
        return intervals[:2]
    
    def _generate_stage_points(self, intervals: List[Tuple[float, float]], stage: int) -> List[float]:
        """
        주어진 구간들에서 새로운 탐색 포인트 생성
        """
        points = []
        points_per_interval = 5 // len(intervals) if intervals else 5
        extra_points = 5 % len(intervals) if intervals else 0
        
        for i, (low, high) in enumerate(intervals):
            # 이 구간에 할당할 포인트 수
            n_points = points_per_interval + (1 if i < extra_points else 0)
            
            # 로그 스케일로 균등 분포
            interval_points = np.logspace(
                np.log10(low),
                np.log10(high),
                n_points + 2  # 경계 포함
            )[1:-1]  # 경계 제외
            
            points.extend(interval_points)
        
        # 정확히 5개로 조정
        if len(points) > 5:
            points = points[:5]
        elif len(points) < 5:
            # 부족하면 기존 구간을 더 세밀하게
            while len(points) < 5:
                if intervals:
                    mid_lr = np.sqrt(intervals[0][0] * intervals[0][1])
                    if mid_lr not in points:
                        points.append(mid_lr)
                else:
                    break
        
        return sorted(points)
    
    def _already_tested(self, lr: float, tolerance: float = 1e-10) -> bool:
        """이미 테스트된 LR인지 확인"""
        for stage_results in self.all_results.values():
            for result in stage_results:
                if abs(result.lr - lr) < tolerance:
                    return True
        return False
    
    def _count_total_tests(self) -> int:
        """총 테스트 수 계산"""
        return sum(len(results) for results in self.all_results.values())
    
    def _generate_final_report(self, total_time: float) -> Dict[str, Any]:
        """최종 보고서 생성"""
        
        # 모든 결과 수집
        all_results_flat = []
        for stage, results in self.all_results.items():
            for r in results:
                all_results_flat.append({
                    'lr': r.lr,
                    'stage': r.stage,
                    'val_loss': r.val_loss,
                    'train_loss': r.train_loss,
                    'convergence_rate': r.convergence_rate,
                    'gradient_norm': r.gradient_norm,
                    'accuracy': r.accuracy,
                    'time': r.time_taken
                })
        
        # Stage별 최고 성능
        stage_best = {}
        for stage, results in self.all_results.items():
            if results:
                best = min(results, key=lambda x: x.val_loss)
                stage_best[f'stage_{stage}'] = {
                    'lr': best.lr,
                    'val_loss': best.val_loss,
                    'accuracy': best.accuracy
                }
        
        report = {
            'strategy': '5-5-5-5 Hierarchical',
            'total_points_tested': self._count_total_tests(),
            'total_time_minutes': total_time / 60,
            'best_lr': self.best_lr,
            'best_loss': self.best_loss,
            'stage_results': stage_best,
            'all_results': all_results_flat,
            'efficiency_gain': {
                'vs_grid_search': f"{(500 / self._count_total_tests()):.1f}x faster",
                'points_saved': 500 - self._count_total_tests()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _save_results(self, report: Dict[str, Any]):
        """결과 저장 (전체 및 각 Stage별)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 전체 JSON 저장
        json_path = self.output_dir / f"hierarchical_lr_sweep_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 각 Stage별 JSON 저장
        for stage, results in self.all_results.items():
            if results:
                stage_data = {
                    'stage': stage,
                    'results': [{
                        'lr': r.lr,
                        'val_loss': r.val_loss,
                        'train_loss': r.train_loss,
                        'accuracy': r.accuracy,
                        'convergence_rate': r.convergence_rate
                    } for r in results]
                }
                stage_json_path = self.output_dir / f"hierarchical_lr_sweep_stage{stage}_{timestamp}.json"
                with open(stage_json_path, 'w') as f:
                    json.dump(stage_data, f, indent=2)
                logger.info(f"  📁 Stage {stage} 저장: {stage_json_path}")
        
        logger.info(f"\n📁 전체 결과 저장: {json_path}")
    
    def _visualize_results(self):
        """결과 시각화 (전체 및 각 Stage별)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 각 Stage별 개별 PNG 생성
        for stage in range(5):
            if self.all_results[stage]:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                lrs = [r.lr for r in self.all_results[stage]]
                losses = [r.val_loss for r in self.all_results[stage]]
                
                ax.semilogx(lrs, losses, 'bo-', markersize=10, linewidth=2)
                ax.set_xlabel('Learning Rate', fontsize=12)
                ax.set_ylabel('Validation Loss', fontsize=12)
                ax.set_title(f'Stage {stage} LR Sweep Results', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # 최고 성능 표시
                best_idx = np.argmin(losses)
                ax.plot(lrs[best_idx], losses[best_idx], 'r*', markersize=20)
                ax.annotate(f'Best: {lrs[best_idx]:.1e}\nLoss: {losses[best_idx]:.4f}',
                           xy=(lrs[best_idx], losses[best_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
                
                # Stage별 PNG 저장
                stage_plot_path = self.output_dir / f"hierarchical_lr_sweep_stage{stage}_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(stage_plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                logger.info(f"  📊 Stage {stage} 시각화: {stage_plot_path}")
        
        # 전체 통합 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hierarchical LR Sweep Results (5-5-5-5 Strategy)', fontsize=16)
        
        # Stage별 결과 플롯
        for stage in range(5):
            ax = axes[stage // 3, stage % 3]
            
            if self.all_results[stage]:
                lrs = [r.lr for r in self.all_results[stage]]
                losses = [r.val_loss for r in self.all_results[stage]]
                
                ax.semilogx(lrs, losses, 'bo-', markersize=8)
                ax.set_xlabel('Learning Rate')
                ax.set_ylabel('Validation Loss')
                ax.set_title(f'Stage {stage}')
                ax.grid(True, alpha=0.3)
                
                # 최고 성능 표시
                best_idx = np.argmin(losses)
                ax.plot(lrs[best_idx], losses[best_idx], 'r*', markersize=15)
                ax.annotate(f'Best: {lrs[best_idx]:.1e}',
                           xy=(lrs[best_idx], losses[best_idx]),
                           xytext=(5, 5), textcoords='offset points')
        
        # 전체 결과 종합
        ax = axes[1, 2]
        all_lrs = []
        all_losses = []
        all_stages = []
        
        for stage, results in self.all_results.items():
            for r in results:
                all_lrs.append(r.lr)
                all_losses.append(r.val_loss)
                all_stages.append(stage)
        
        if all_lrs:
            scatter = ax.scatter(all_lrs, all_losses, c=all_stages, 
                               cmap='viridis', s=100, alpha=0.6)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Validation Loss')
            ax.set_title('All Stages Combined')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Stage')
            
            # 최적 LR 표시
            ax.plot(self.best_lr, self.best_loss, 'r*', markersize=20)
            ax.annotate(f'Optimal: {self.best_lr:.1e}',
                       xy=(self.best_lr, self.best_loss),
                       xytext=(5, 5), textcoords='offset points',
                       fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        plot_path = self.output_dir / f"hierarchical_lr_sweep_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 시각화 저장: {plot_path}")


def main():
    """테스트용 메인 함수"""
    import sys
    sys.path.append('/mnt/c/large_project/linux_red_heart')
    
    from models.red_heart_model_final import RedHeartModel
    from training.data_loader import create_data_loaders
    
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 생성
    model = RedHeartModel(config={
        'vocab_size': 50000,
        'max_length': 512,
        'd_model': 896,
        'num_heads': 16,
        'num_layers': 6,
        'd_ff': 3584,
        'dropout': 0.1
    }).to(device)
    
    # 데이터 로더 생성  
    train_loader, val_loader = create_data_loaders(
        batch_size=2,
        num_workers=0
    )
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # Hierarchical LR Sweep 실행
    sweep = HierarchicalLRSweep(
        test_epochs=3,
        test_steps=50,
        warmup_steps=10
    )
    
    results = sweep.run_hierarchical_sweep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )
    
    print(f"\n최적 LR: {results['best_lr']:.1e}")
    print(f"최적 Loss: {results['best_loss']:.4f}")
    print(f"총 테스트 포인트: {results['total_points_tested']}")


if __name__ == "__main__":
    main()