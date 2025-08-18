"""
최적화된 차원 변환 어댑터 - Red Heart AI System
Optimized Dimension Adapter for Red Heart AI System

통일된 차원 변환 전략으로 정보 손실 최소화 및 계산 효율성 향상
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OptimizedDimensionAdapter(nn.Module):
    """
    최적화된 차원 변환 어댑터
    
    특징:
    - 표준 internal dimension (1024) 사용
    - Residual connection으로 정보 손실 최소화
    - 점진적 차원 변환으로 안정성 향상
    - 각 헤드별 맞춤형 차원 지원
    """
    
    def __init__(self, 
                 backbone_dim: int = 1280,
                 target_dim: Optional[int] = None,
                 standard_dim: int = 1024,
                 use_residual: bool = True,
                 dropout_rate: float = 0.1):
        """
        Args:
            backbone_dim: 백본 차원 (기본 1280)
            target_dim: 타겟 헤드의 요구 차원 (None이면 standard_dim 사용)
            standard_dim: 표준 internal 차원 (기본 1024)
            use_residual: residual connection 사용 여부
            dropout_rate: 드롭아웃 비율
        """
        super().__init__()
        
        self.backbone_dim = backbone_dim
        self.target_dim = target_dim or standard_dim
        self.standard_dim = standard_dim
        self.use_residual = use_residual
        
        # 1. Backbone → Standard 변환 (1280 → 1024)
        self.backbone_to_standard = nn.Sequential(
            nn.Linear(backbone_dim, standard_dim),
            nn.LayerNorm(standard_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 2. Standard → Target 변환 (필요시)
        if self.target_dim != standard_dim:
            self.standard_to_target = nn.Sequential(
                nn.Linear(standard_dim, self.target_dim),
                nn.LayerNorm(self.target_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            
            # 3. Target → Standard 역변환 (필요시)
            self.target_to_standard = nn.Sequential(
                nn.Linear(self.target_dim, standard_dim),
                nn.LayerNorm(standard_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.standard_to_target = None
            self.target_to_standard = None
        
        # 4. Standard → Backbone 역변환 (1024 → 1280)
        self.standard_to_backbone = nn.Sequential(
            nn.Linear(standard_dim, backbone_dim),
            nn.LayerNorm(backbone_dim)
        )
        
        # 5. Residual connection용 projection (차원이 다를 때)
        if use_residual and backbone_dim != backbone_dim:  # 실제로는 같지만 일반화용
            self.residual_proj = nn.Linear(backbone_dim, backbone_dim)
        else:
            self.residual_proj = None
            
        logger.info(f"OptimizedDimensionAdapter 초기화: {backbone_dim}→{standard_dim}→{self.target_dim}→{standard_dim}→{backbone_dim}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 인코딩: backbone_dim → target_dim
        
        Args:
            x: 백본 출력 텐서 [batch_size, seq_len, backbone_dim]
            
        Returns:
            타겟 차원으로 변환된 텐서 [batch_size, seq_len, target_dim]
        """
        # 1. Backbone → Standard
        standard = self.backbone_to_standard(x)
        
        # 2. Standard → Target (필요시)  
        if self.standard_to_target is not None:
            target = self.standard_to_target(standard)
            return target
        else:
            return standard
    
    def decode(self, x: torch.Tensor, original_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        출력 디코딩: target_dim → backbone_dim
        
        Args:
            x: 헤드 출력 텐서 [batch_size, seq_len, target_dim]
            original_input: residual connection용 원본 입력 (선택사항)
            
        Returns:
            백본 차원으로 복원된 텐서 [batch_size, seq_len, backbone_dim]
        """
        # 1. Target → Standard (필요시)
        if self.target_to_standard is not None:
            standard = self.target_to_standard(x)
        else:
            standard = x
        
        # 2. Standard → Backbone
        output = self.standard_to_backbone(standard)
        
        # 3. Residual connection (선택사항)
        if self.use_residual and original_input is not None:
            if self.residual_proj is not None:
                residual = self.residual_proj(original_input)
            else:
                residual = original_input
            output = output + residual
            
        return output
    
    def forward(self, x: torch.Tensor, head_function: Optional[callable] = None) -> Dict[str, torch.Tensor]:
        """
        전체 forward pass: encode → head processing → decode
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, backbone_dim]
            head_function: 헤드 처리 함수 (선택사항)
            
        Returns:
            결과 딕셔너리 {'encoded', 'processed', 'decoded'}
        """
        # 인코딩
        encoded = self.encode(x)
        
        # 헤드 처리 (제공된 경우)
        if head_function is not None:
            processed = head_function(encoded)
        else:
            processed = encoded
            
        # 디코딩
        decoded = self.decode(processed, original_input=x)
        
        return {
            'encoded': encoded,
            'processed': processed, 
            'decoded': decoded
        }
    
    def get_dimension_info(self) -> Dict[str, int]:
        """차원 정보 반환"""
        return {
            'backbone_dim': self.backbone_dim,
            'standard_dim': self.standard_dim,
            'target_dim': self.target_dim,
            'parameter_count': sum(p.numel() for p in self.parameters())
        }


class HeadSpecificAdapters:
    """각 헤드별 최적화된 어댑터 팩토리"""
    
    @staticmethod
    def create_emotion_adapter() -> OptimizedDimensionAdapter:
        """감정 헤드용 어댑터 (1280→1024→1280)"""
        return OptimizedDimensionAdapter(
            backbone_dim=1280,
            target_dim=1024,
            use_residual=True,
            dropout_rate=0.1
        )
    
    @staticmethod  
    def create_bentham_adapter() -> OptimizedDimensionAdapter:
        """벤담 헤드용 어댑터 (1280→768→1280)"""
        return OptimizedDimensionAdapter(
            backbone_dim=1280,
            target_dim=768,
            use_residual=True,
            dropout_rate=0.1
        )
    
    @staticmethod
    def create_semantic_adapter() -> OptimizedDimensionAdapter:
        """의미 헤드용 어댑터 (1280→512→1280)"""
        return OptimizedDimensionAdapter(
            backbone_dim=1280,
            target_dim=512,
            use_residual=True,
            dropout_rate=0.1
        )
    
    @staticmethod
    def create_regret_adapter() -> OptimizedDimensionAdapter:
        """후회 헤드용 어댑터 (1280→768→1280)"""
        return OptimizedDimensionAdapter(
            backbone_dim=1280,
            target_dim=768,
            use_residual=True,
            dropout_rate=0.15  # 후회 헤드는 더 강한 정규화
        )
    
    @staticmethod
    def create_meta_adapter() -> OptimizedDimensionAdapter:
        """메타 통합 헤드용 어댑터 (1280→256→1280)"""
        return OptimizedDimensionAdapter(
            backbone_dim=1280,
            target_dim=256,
            use_residual=True,
            dropout_rate=0.05  # 메타 헤드는 정보 보존 우선
        )


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트 설정
    batch_size = 4
    seq_len = 128
    backbone_dim = 1280
    
    # 더미 입력 생성
    dummy_input = torch.randn(batch_size, seq_len, backbone_dim)
    
    # 각 헤드별 어댑터 테스트
    adapters = {
        'emotion': HeadSpecificAdapters.create_emotion_adapter(),
        'bentham': HeadSpecificAdapters.create_bentham_adapter(),
        'semantic': HeadSpecificAdapters.create_semantic_adapter(),
        'regret': HeadSpecificAdapters.create_regret_adapter(),
        'meta': HeadSpecificAdapters.create_meta_adapter()
    }
    
    print("=== 최적화된 차원 변환 어댑터 테스트 ===")
    
    for name, adapter in adapters.items():
        print(f"\n[{name.upper()} ADAPTER]")
        
        # 차원 정보 출력
        dim_info = adapter.get_dimension_info()
        print(f"차원 변환: {dim_info['backbone_dim']} → {dim_info['target_dim']} → {dim_info['backbone_dim']}")
        print(f"파라미터 수: {dim_info['parameter_count']:,}")
        
        # Forward pass 테스트
        with torch.no_grad():
            result = adapter.forward(dummy_input)
            
            print(f"입력 shape: {dummy_input.shape}")
            print(f"인코딩 shape: {result['encoded'].shape}")
            print(f"디코딩 shape: {result['decoded'].shape}")
            
            # 차원 일관성 검증
            assert result['decoded'].shape == dummy_input.shape, f"{name} 어댑터 차원 불일치!"
            print("✅ 차원 일관성 검증 통과")
    
    print(f"\n🎉 모든 어댑터 테스트 완료!")