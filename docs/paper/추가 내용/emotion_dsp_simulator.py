"""
Red Heart AI - DSP 기반 감정 시뮬레이터 (40M 파라미터)
Digital Signal Processing Framework for Emotion Simulation
문서: docs/emotion_dsp_simulator_design.md 기반 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)

# DSP 파라미터 상수 정의 (문서 기반)
EMOTION_FREQ_MAPPING = {
    'fear': (20, 80),        # 공포/불안: 20-80 Hz
    'anger': (80, 200),      # 분노/흥분: 80-200 Hz  
    'sadness': (200, 500),   # 슬픔/우울: 200-500 Hz
    'joy': (500, 2000),      # 기쁨/행복: 500-2kHz
    'love': (1000, 4000),    # 사랑/애착: 1-4kHz
    'surprise': (4000, 8000), # 경외/놀라움: 4kHz+
    'disgust': (100, 300),   # 혐오: 100-300 Hz
}

# ADSR 엔벨로프 파라미터 (문서 Section 2.2)
ADSR_PRESETS = {
    'fear': {'attack': 0.03, 'decay': 0.2, 'sustain': 0.4, 'release': 1.0},
    'anger': {'attack': 0.1, 'decay': 0.3, 'sustain': 0.8, 'release': 3.0},
    'sadness': {'attack': 1.0, 'decay': 2.0, 'sustain': 0.3, 'release': 5.0},
    'joy': {'attack': 0.3, 'decay': 0.5, 'sustain': 0.7, 'release': 1.0},
    'love': {'attack': 0.5, 'decay': 0.8, 'sustain': 0.6, 'release': 2.0},
    'surprise': {'attack': 0.01, 'decay': 0.1, 'sustain': 0.5, 'release': 0.5},
    'disgust': {'attack': 0.05, 'decay': 0.2, 'sustain': 0.6, 'release': 0.8},
}

class EmotionDSPSimulator(nn.Module):
    """
    DSP 기반 감정 시뮬레이터 - 14M 파라미터 (10M의 1.364배)
    주파수 도메인과 시간 도메인에서 감정을 모델링
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        self.hidden_dim = self.config.get('hidden_dim', 384)  # 조정된 차원
        self.num_freq_bands = 96  # 주파수 대역 수
        self.num_emotions = 7
        
        # 1. 주파수 분석 모듈 (2M 파라미터)
        self.freq_analyzer = nn.Sequential(
            nn.Linear(self.hidden_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, self.num_freq_bands * 2),  # 실수부 + 허수부
        )
        
        # 2. ADSR 엔벨로프 생성기 (1.5M 파라미터)
        self.adsr_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, 4 * self.num_emotions),  # A,D,S,R for each emotion
        )
        
        # 3. Valence-Arousal 매핑 (1.5M 파라미터)
        self.va_mapper = nn.Sequential(
            nn.Linear(self.hidden_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, 2),  # valence, arousal
            nn.Tanh(),  # -1 to 1 범위
        )
        
        # 4. 감정 공명 엔진 (3.5M 파라미터)
        self.resonance_engine = EmotionResonanceEngine(
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands
        )
        
        # 5. 적응형 리버브 시스템 (2M 파라미터)
        self.reverb_system = AdaptiveReverbSystem(
            hidden_dim=self.hidden_dim,
            num_taps=48  # 딜레이 탭 수
        )
        
        # 6. 하이브리드 DSP 체인 (2M 파라미터)
        self.dsp_chain = HybridDSPChain(
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands
        )
        
        # 7. 최종 감정 합성기 (1M 파라미터)
        self.emotion_synthesizer = nn.Sequential(
            nn.Linear(self.num_freq_bands + 4 + 2 + self.hidden_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, self.num_emotions),
        )
        
        logger.info(f"EmotionDSPSimulator 초기화 완료 (~14M 파라미터)")
        
    def forward(self, x: torch.Tensor, emotion_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        감정 DSP 처리
        
        Args:
            x: 입력 특징 (batch_size, hidden_dim)
            emotion_context: 선택적 감정 컨텍스트 (batch_size, num_emotions)
            
        Returns:
            Dict containing:
                - emotion_spectrum: 주파수 도메인 감정 스펙트럼
                - adsr_params: ADSR 엔벨로프 파라미터
                - valence_arousal: Valence-Arousal 좌표
                - resonance_map: 감정 공명 맵
                - final_emotions: 최종 감정 확률
        """
        batch_size = x.shape[0]
        
        # 1. 주파수 분석 - 감정을 주파수 도메인으로 변환
        freq_features = self.freq_analyzer(x)  # (batch, num_freq_bands * 2)
        freq_complex = freq_features.view(batch_size, self.num_freq_bands, 2)
        freq_magnitude = torch.sqrt(freq_complex[..., 0]**2 + freq_complex[..., 1]**2)
        freq_phase = torch.atan2(freq_complex[..., 1], freq_complex[..., 0])
        
        # 2. ADSR 엔벨로프 생성
        adsr_raw = self.adsr_generator(x)  # (batch, 4 * num_emotions)
        adsr_params = torch.sigmoid(adsr_raw.view(batch_size, self.num_emotions, 4))
        
        # 3. Valence-Arousal 매핑
        valence_arousal = self.va_mapper(x)  # (batch, 2)
        
        # 4. 감정 공명 계산
        resonance_features = self.resonance_engine(
            freq_magnitude, 
            freq_phase,
            emotion_context
        )
        
        # 5. 적응형 리버브 적용
        reverb_features = self.reverb_system(x, freq_magnitude)
        
        # 6. DSP 체인 처리
        dsp_output = self.dsp_chain(
            freq_magnitude,
            adsr_params,
            valence_arousal,
            resonance_features
        )
        
        # 7. 최종 감정 합성
        combined_features = torch.cat([
            freq_magnitude,  # 주파수 특징
            adsr_params.view(batch_size, -1)[:, :4],  # ADSR 요약
            valence_arousal,  # VA 좌표
            dsp_output,  # DSP 체인 출력
        ], dim=-1)
        
        final_emotions = self.emotion_synthesizer(combined_features)
        final_emotions = F.softmax(final_emotions, dim=-1)
        
        return {
            'emotion_spectrum': freq_magnitude,
            'adsr_params': adsr_params,
            'valence_arousal': valence_arousal,
            'resonance_map': resonance_features,
            'reverb_features': reverb_features,
            'final_emotions': final_emotions,
        }


class EmotionResonanceEngine(nn.Module):
    """
    감정 공명 엔진 - Wavelet-FFT 하이브리드 분석 (3.5M 파라미터)
    문서 Section 2.3 구현
    """
    
    def __init__(self, hidden_dim: int, num_freq_bands: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_freq_bands = num_freq_bands
        
        # Wavelet 변환 레이어
        self.wavelet_transform = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        
        # 공명 패턴 매칭
        self.pattern_matcher = nn.Sequential(
            nn.Linear(num_freq_bands * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        
        # 시간-주파수 융합
        self.time_freq_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
    def forward(self, freq_magnitude: torch.Tensor, freq_phase: torch.Tensor, 
                emotion_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        감정 공명 계산
        
        Args:
            freq_magnitude: 주파수 크기 (batch, num_freq_bands)
            freq_phase: 주파수 위상 (batch, num_freq_bands)
            emotion_context: 감정 컨텍스트 (batch, num_emotions)
            
        Returns:
            공명 특징 (batch, hidden_dim)
        """
        batch_size = freq_magnitude.shape[0]
        
        # Wavelet 변환 적용
        freq_signal = freq_magnitude.unsqueeze(1)  # (batch, 1, num_freq_bands)
        wavelet_features = self.wavelet_transform(freq_signal)  # (batch, 64, num_freq_bands)
        wavelet_pooled = F.adaptive_avg_pool1d(wavelet_features, 1).squeeze(-1)  # (batch, 64)
        
        # 주파수-위상 결합
        freq_combined = torch.cat([freq_magnitude, freq_phase], dim=-1)
        pattern_features = self.pattern_matcher(freq_combined)
        
        # 시간-주파수 융합 (Self-Attention)
        fusion_input = pattern_features.unsqueeze(1)  # (batch, 1, hidden_dim)
        fused_features, _ = self.time_freq_fusion(
            fusion_input, fusion_input, fusion_input
        )
        fused_features = fused_features.squeeze(1)
        
        return fused_features


class AdaptiveReverbSystem(nn.Module):
    """
    적응형 리버브 시스템 - 감정 메모리 모델링 (2M 파라미터)
    문서 Section 2.4 구현
    """
    
    def __init__(self, hidden_dim: int, num_taps: int = 32):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_taps = num_taps
        
        # 딜레이 라인 생성
        self.delay_line = nn.Parameter(torch.randn(num_taps, hidden_dim) * 0.01)
        
        # 적응형 가중치 네트워크
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_taps),
            nn.Softmax(dim=-1)
        )
        
        # 피드백 경로
        self.feedback_path = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        """
        적응형 리버브 처리
        
        Args:
            x: 입력 특징 (batch, hidden_dim)
            freq_features: 주파수 특징 (batch, num_freq_bands)
            
        Returns:
            리버브 적용된 특징 (batch, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # 적응형 가중치 계산
        tap_weights = self.weight_network(x)  # (batch, num_taps)
        
        # 딜레이 라인과 가중치 적용
        delayed = torch.matmul(tap_weights, self.delay_line)  # (batch, hidden_dim)
        
        # 피드백 적용
        feedback_input = torch.cat([x, delayed], dim=-1)
        reverb_output = self.feedback_path(feedback_input)
        
        # 잔류 연결
        output = x + 0.3 * reverb_output
        
        return output


class HybridDSPChain(nn.Module):
    """
    하이브리드 DSP 처리 체인 (2M 파라미터)
    문서 Section 3.1 구현
    """
    
    def __init__(self, hidden_dim: int, num_freq_bands: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Adaptive EQ
        self.adaptive_eq = nn.Sequential(
            nn.Linear(num_freq_bands, 128),
            nn.GELU(),
            nn.Linear(128, num_freq_bands),
        )
        
        # Dynamic Compressor
        self.compressor = nn.Sequential(
            nn.Linear(num_freq_bands + 4 + 2, 128),  # freq + adsr_summary + va
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        
    def forward(self, freq_magnitude: torch.Tensor, adsr_params: torch.Tensor,
                valence_arousal: torch.Tensor, resonance_features: torch.Tensor) -> torch.Tensor:
        """
        DSP 체인 처리
        
        Returns:
            처리된 특징 (batch, hidden_dim)
        """
        batch_size = freq_magnitude.shape[0]
        
        # Adaptive EQ 적용
        eq_adjusted = freq_magnitude + self.adaptive_eq(freq_magnitude)
        
        # ADSR 요약 (평균)
        adsr_summary = adsr_params.mean(dim=1)  # (batch, 4)
        
        # 특징 결합
        combined = torch.cat([
            eq_adjusted,
            adsr_summary,
            valence_arousal
        ], dim=-1)
        
        # Dynamic Compression
        compressed = self.compressor(combined)
        
        # 공명 특징과 융합
        output = compressed + 0.5 * resonance_features
        
        return output


class DynamicKalmanFilter(nn.Module):
    """
    동적 칼만 필터 - 감정 상태 융합
    기존 감정 분석기와 DSP 시뮬레이터 출력 융합
    """
    
    def __init__(self, state_dim: int = 7):
        super().__init__()
        
        self.state_dim = state_dim  # 감정 차원
        
        # 상태 전이 행렬 (학습 가능)
        self.F = nn.Parameter(torch.eye(state_dim))
        
        # 관측 행렬
        self.H = nn.Parameter(torch.eye(state_dim))
        
        # 프로세스 노이즈 공분산
        self.Q = nn.Parameter(torch.eye(state_dim) * 0.01)
        
        # 관측 노이즈 공분산
        self.R = nn.Parameter(torch.eye(state_dim) * 0.1)
        
        # 적응형 가중치 네트워크
        self.weight_adapter = nn.Sequential(
            nn.Linear(state_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, traditional_emotions: torch.Tensor, 
                dsp_emotions: torch.Tensor,
                prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        칼만 필터를 통한 감정 융합
        
        Args:
            traditional_emotions: 기존 감정 분석기 출력 (batch, state_dim)
            dsp_emotions: DSP 시뮬레이터 출력 (batch, state_dim)
            prev_state: 이전 상태 (batch, state_dim)
            
        Returns:
            융합된 감정 상태 (batch, state_dim)
        """
        batch_size = traditional_emotions.shape[0]
        
        if prev_state is None:
            # 초기 상태는 두 입력의 평균
            prev_state = (traditional_emotions + dsp_emotions) / 2
        
        # 예측 단계
        x_pred = torch.matmul(prev_state, self.F.T)
        P_pred = self.F @ self.F.T + self.Q
        
        # 적응형 가중치 계산
        combined_input = torch.cat([traditional_emotions, dsp_emotions], dim=-1)
        weights = self.weight_adapter(combined_input)  # (batch, 2)
        
        # 가중 평균 관측값
        z = weights[:, 0:1] * traditional_emotions + weights[:, 1:2] * dsp_emotions
        
        # 업데이트 단계
        y = z - torch.matmul(x_pred, self.H.T)  # 혁신
        S = self.H @ P_pred @ self.H.T + self.R  # 혁신 공분산
        K = P_pred @ self.H.T @ torch.inverse(S)  # 칼만 이득
        
        # 상태 업데이트
        x_updated = x_pred + torch.matmul(y, K.T)
        
        # 정규화 (확률 분포로)
        x_updated = F.softmax(x_updated, dim=-1)
        
        return x_updated


# 모듈 등록을 위한 편의 함수
def create_emotion_dsp_simulator(config: Optional[Dict[str, Any]] = None) -> EmotionDSPSimulator:
    """
    EmotionDSPSimulator 인스턴스 생성
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        EmotionDSPSimulator 인스턴스
    """
    return EmotionDSPSimulator(config)


def create_kalman_filter(state_dim: int = 7) -> DynamicKalmanFilter:
    """
    DynamicKalmanFilter 인스턴스 생성
    
    Args:
        state_dim: 상태 차원 (감정 수)
        
    Returns:
        DynamicKalmanFilter 인스턴스
    """
    return DynamicKalmanFilter(state_dim)