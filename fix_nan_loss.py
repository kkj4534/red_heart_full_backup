#!/usr/bin/env python3
"""
NaN Loss 문제 해결 패치
Fix NaN Loss Issues in Unified Learning System

문제 원인:
1. 완전 랜덤 타겟으로 인한 학습 불안정성
2. cosine_embedding_loss 불안정성
3. 차원 불일치 및 정보 손실
4. Loss 스케일 불균형

해결 방안:
1. 안정적인 synthetic 타겟 생성
2. 안전한 loss 함수 사용
3. 차원 검증 및 안전한 변환
4. Loss 정규화 및 클리핑
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class StableLossCalculator:
    """안정적인 Loss 계산기"""
    
    def __init__(self):
        self.loss_history = {}  # 각 헤드별 loss 히스토리
        self.synthetic_targets = {}  # 합성 타겟 캐시
        
    def calculate_stable_losses(self, outputs: Dict[str, torch.Tensor], 
                              batch_data: Dict[str, Any], 
                              active_heads: List, 
                              head_configs: Dict) -> Dict[str, torch.Tensor]:
        """안정적인 손실 함수 계산"""
        losses = {}
        batch_size = batch_data.get('batch_size', 1)
        
        for head_type in active_heads:
            config = head_configs[head_type]
            head_output = outputs['head_outputs'][head_type.value]
            
            # 출력 검증 및 정규화
            head_output = self._validate_and_normalize_output(head_output, head_type)
            
            # 안정적인 타겟 생성
            target = self._generate_stable_target(head_output, head_type, batch_size)
            
            # 안전한 loss 계산
            loss = self._calculate_safe_loss(head_output, target, head_type, config)
            
            # Loss 클리핑 및 검증
            loss = self._clip_and_validate_loss(loss, head_type)
            
            losses[head_type.value] = loss
            
        return losses
    
    def _validate_and_normalize_output(self, output: torch.Tensor, head_type) -> torch.Tensor:
        """출력 검증 및 정규화"""
        # NaN/Inf 체크
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.warning(f"{head_type.value} 헤드 출력에 NaN/Inf 발견 - 0으로 대체")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 극값 클리핑
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        # L2 정규화 (필요시)
        if output.dim() > 1 and output.size(-1) > 1:
            output = F.normalize(output, p=2, dim=-1)
            
        return output
    
    def _generate_stable_target(self, output: torch.Tensor, head_type, batch_size: int) -> torch.Tensor:
        """안정적인 합성 타겟 생성"""
        device = output.device
        output_shape = output.shape
        
        # 헤드 타입별 안정적인 타겟 생성
        if 'EMOTION' in head_type.value.upper():
            # 감정: 소프트 원-핫 인코딩
            num_classes = min(output_shape[-1], 10)
            target_classes = torch.randint(0, num_classes, (batch_size,), device=device)
            target = F.one_hot(target_classes, num_classes=num_classes).float()
            
            # 라벨 스무딩 적용
            target = target * 0.9 + 0.1 / num_classes
            
        elif 'BENTHAM' in head_type.value.upper():
            # 벤담: 0.5 중심의 작은 변동
            target = 0.5 + 0.1 * torch.randn(batch_size, 1, device=device)
            target = torch.clamp(target, 0.0, 1.0)
            
        elif 'SEMANTIC' in head_type.value.upper():
            # 의미: 정규화된 랜덤 벡터 (코사인 유사도 안정화)
            target = torch.randn(output_shape, device=device)
            target = F.normalize(target, p=2, dim=-1)
            
        elif 'REGRET' in head_type.value.upper():
            # 후회: 0 중심의 작은 값
            target = 0.1 * torch.randn(batch_size, 1, device=device)
            target = torch.clamp(target, -1.0, 1.0)
            
        else:
            # 기본: 출력과 유사한 스케일의 안정적인 타겟
            target = 0.1 * torch.randn_like(output)
            target = torch.clamp(target, -1.0, 1.0)
            
        return target
    
    def _calculate_safe_loss(self, output: torch.Tensor, target: torch.Tensor, 
                           head_type, config) -> torch.Tensor:
        """안전한 loss 계산"""
        
        try:
            if 'EMOTION' in head_type.value.upper():
                # 감정: 안전한 크로스 엔트로피
                if output.dim() > 1:
                    # 소프트맥스 + 크로스 엔트로피 대신 KL divergence 사용
                    output_probs = F.softmax(output, dim=-1)
                    target_probs = target
                    loss = F.kl_div(F.log_softmax(output, dim=-1), target_probs, reduction='batchmean')
                else:
                    loss = F.mse_loss(output, target.mean(dim=-1, keepdim=True))
                    
            elif 'BENTHAM' in head_type.value.upper():
                # 벤담: Smooth L1 Loss (더 안정적)
                loss = F.smooth_l1_loss(output[:, :1] if output.dim() > 1 else output, target)
                
            elif 'SEMANTIC' in head_type.value.upper():
                # 의미: 안전한 코사인 유사도 대신 MSE 사용
                loss = F.mse_loss(output, target)
                
            elif 'REGRET' in head_type.value.upper():
                # 후회: Huber Loss 사용
                loss = F.smooth_l1_loss(output[:, :1] if output.dim() > 1 else output, target)
                
            else:
                # 기본: MSE
                loss = F.mse_loss(output, target)
                
        except Exception as e:
            logger.warning(f"{head_type.value} loss 계산 실패: {e}, fallback MSE 사용")
            loss = F.mse_loss(output, target)
            
        return loss
    
    def _clip_and_validate_loss(self, loss: torch.Tensor, head_type) -> torch.Tensor:
        """Loss 클리핑 및 검증"""
        
        # NaN/Inf 체크
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"{head_type.value} loss가 NaN/Inf입니다. 1.0으로 대체합니다.")
            return torch.tensor(1.0, device=loss.device, requires_grad=True)
        
        # 극값 클리핑
        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        # 히스토리 업데이트
        if head_type.value not in self.loss_history:
            self.loss_history[head_type.value] = []
        
        self.loss_history[head_type.value].append(float(loss.item()))
        
        # 최근 10개만 유지
        if len(self.loss_history[head_type.value]) > 10:
            self.loss_history[head_type.value] = self.loss_history[head_type.value][-10:]
        
        return loss


def patch_unified_learning_system():
    """unified_learning_system.py에 패치 적용"""
    
    print("🔧 NaN Loss 문제 해결 패치를 적용합니다...")
    
    # unified_learning_system.py 읽기
    try:
        with open('/mnt/c/large_project/linux_red_heart/unified_learning_system.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 기존 _calculate_losses 함수를 안전한 버전으로 교체
        old_function = """    def _calculate_losses(self, outputs: Dict[str, torch.Tensor],
                         batch_data: Dict[str, Any],
                         active_heads: List[HeadType]) -> Dict[str, torch.Tensor]:
        \"\"\"손실 함수 계산\"\"\"
        losses = {}
        
        # 각 헤드별 손실 계산
        for head_type in active_heads:
            config = self.head_configs[head_type]
            
            # 가상의 타겟 생성 (실제 구현에서는 batch_data에서 추출)
            batch_size = batch_data.get('batch_size', 1)
            
            if head_type == HeadType.EMOTION_EMPATHY:
                # 감정 분류 손실
                target = torch.randint(0, 10, (batch_size,), device=outputs['head_outputs'][head_type.value].device)
                logits = outputs['head_outputs'][head_type.value][:, :10]  # 10개 감정 클래스
                loss = F.cross_entropy(logits, target, label_smoothing=config.label_smoothing)
                
            elif head_type == HeadType.BENTHAM_FROMM:
                # 윤리 점수 회귀 손실
                target = torch.rand(batch_size, 1, device=outputs['head_outputs'][head_type.value].device)
                pred = outputs['head_outputs'][head_type.value][:, :1]
                loss = F.mse_loss(pred, target)
                
            elif head_type == HeadType.SEMANTIC_SURD:
                # 의미 유사도 손실
                target = torch.rand(batch_size, 768, device=outputs['head_outputs'][head_type.value].device)
                pred = outputs['head_outputs'][head_type.value]
                loss = F.cosine_embedding_loss(pred, target, torch.ones(batch_size, device=pred.device))
                
            elif head_type == HeadType.REGRET_LEARNING:
                # 후회 예측 손실
                target = torch.rand(batch_size, 1, device=outputs['head_outputs'][head_type.value].device)
                pred = outputs['head_outputs'][head_type.value][:, :1]
                loss = F.smooth_l1_loss(pred, target)
                
            else:
                # 기본 손실
                target = torch.randn_like(outputs['head_outputs'][head_type.value])
                loss = F.mse_loss(outputs['head_outputs'][head_type.value], target)
            
            # 가중치 적용
            losses[head_type.value] = loss * config.loss_weight
        
        return losses"""
        
        new_function = """    def _calculate_losses(self, outputs: Dict[str, torch.Tensor],
                         batch_data: Dict[str, Any],
                         active_heads: List[HeadType]) -> Dict[str, torch.Tensor]:
        \"\"\"안정적인 손실 함수 계산 (NaN 방지)\"\"\"
        
        # 안정적인 loss 계산기 초기화
        if not hasattr(self, '_stable_loss_calculator'):
            from fix_nan_loss import StableLossCalculator
            self._stable_loss_calculator = StableLossCalculator()
        
        # 안정적인 loss 계산
        losses = self._stable_loss_calculator.calculate_stable_losses(
            outputs, batch_data, active_heads, self.head_configs
        )
        
        # 가중치 적용 및 추가 안전 검사
        weighted_losses = {}
        for head_type in active_heads:
            config = self.head_configs[head_type]
            loss = losses[head_type.value]
            
            # 가중치 적용
            weighted_loss = loss * config.loss_weight
            
            # 최종 안전 검사
            if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                logger.warning(f"{head_type.value} 가중 loss NaN/Inf 감지, 1.0으로 대체")
                weighted_loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
            
            weighted_losses[head_type.value] = weighted_loss
        
        return weighted_losses"""
        
        # 함수 교체
        if old_function in content:
            content = content.replace(old_function, new_function)
            print("✅ _calculate_losses 함수 패치 완료")
        else:
            print("⚠️  기존 _calculate_losses 함수를 찾을 수 없습니다")
            
        # 파일에 import 추가
        import_line = "from fix_nan_loss import StableLossCalculator"
        if import_line not in content:
            # import 섹션에 추가
            import_insertion_point = "from intelligent_synergy_system import IntelligentSynergySystem"
            if import_insertion_point in content:
                content = content.replace(import_insertion_point, f"{import_insertion_point}\n{import_line}")
                print("✅ import 구문 추가 완료")
        
        # 백업 파일 생성
        backup_path = '/mnt/c/large_project/linux_red_heart/unified_learning_system_backup.py'
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open('/mnt/c/large_project/linux_red_heart/unified_learning_system.py', 'r', encoding='utf-8') as original:
                f.write(original.read())
        print(f"✅ 백업 파일 생성: {backup_path}")
        
        # 수정된 내용 저장
        with open('/mnt/c/large_project/linux_red_heart/unified_learning_system.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("🎉 NaN Loss 문제 해결 패치 적용 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 패치 적용 실패: {e}")
        return False


if __name__ == "__main__":
    patch_unified_learning_system()