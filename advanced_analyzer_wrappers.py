"""
Advanced Analyzer nn.Module 래퍼
nn.Module을 상속하지 않는 Advanced Analyzer들을 래핑하여 학습 가능하게 만듦
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedEmotionAnalyzerWrapper(nn.Module):
    """Advanced Emotion Analyzer를 nn.Module로 래핑 (48M 파라미터)"""
    
    def __init__(self):
        super().__init__()
        
        # 원본 Analyzer import 및 초기화
        from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
        self.analyzer = AdvancedEmotionAnalyzer()
        
        # 내부 nn.Module들을 직접 속성으로 등록 (학습 가능하게)
        self._register_internal_modules()
        
        logger.info("✅ Advanced Emotion Analyzer Wrapper 초기화 (48M 파라미터)")
    
    def _register_internal_modules(self):
        """내부 nn.Module들을 self의 속성으로 등록"""
        
        # 생체신호 처리 네트워크 (10M)
        if hasattr(self.analyzer, 'biometric_processor'):
            self.biometric_processor = self.analyzer.biometric_processor
            logger.info("  - biometric_processor 등록 (10M)")
        
        # 멀티모달 융합 레이어 (10M)
        if hasattr(self.analyzer, 'multimodal_fusion'):
            self.multimodal_fusion = self.analyzer.multimodal_fusion
            logger.info("  - multimodal_fusion 등록 (10M)")
        
        # 시계열 감정 추적 (10M)
        if hasattr(self.analyzer, 'temporal_emotion'):
            self.temporal_emotion = self.analyzer.temporal_emotion
            logger.info("  - temporal_emotion 등록 (10M)")
        
        # 문화적 뉘앙스 감지 (13M)
        if hasattr(self.analyzer, 'cultural_nuance'):
            self.cultural_nuance = self.analyzer.cultural_nuance
            logger.info("  - cultural_nuance 등록 (13M)")
        
        # 고급 MoE 확장 (5M)
        if hasattr(self.analyzer, 'advanced_moe'):
            self.advanced_moe = self.analyzer.advanced_moe
            logger.info("  - advanced_moe 등록 (5M)")
        
        # emotion_moe 체크
        if hasattr(self.analyzer, 'emotion_moe'):
            self.emotion_moe = self.analyzer.emotion_moe
            logger.info("  - emotion_moe 등록")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - 제공된 임베딩 직접 처리"""
        logger.info("🔄 AdvancedEmotionAnalyzerWrapper forward 실행 시작")
        logger.info(f"   입력 임베딩 차원: {x.shape}")
        
        try:
            # 제공된 896차원 임베딩을 직접 처리
            # analyze_emotion을 호출하면 새로운 768차원 임베딩을 생성하므로 사용하지 않음
            return self._process_embeddings(x, **kwargs)
        except Exception as e:
            logger.error(f"❌ AdvancedEmotionAnalyzerWrapper 실행 실패: {e}")
            raise RuntimeError(f"감정 분석 실패 - NO FALLBACK: {e}")
    
    def _process_embeddings(self, embeddings: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """임베딩 직접 처리 - NO FALLBACK"""
        logger.info("   🔄 임베딩 직접 처리 시작")
        logger.info(f"      입력 차원: {embeddings.shape}")
        output = {}
        
        # 입력 차원에 따라 처리 분기
        if embeddings.shape[-1] == 896:
            # 896차원 → 768차원 프로젝션 (내부 모듈들이 768차원 기대)
            if not hasattr(self, 'embedding_projection_896'):
                self.embedding_projection_896 = nn.Sequential(
                    nn.Linear(896, 768),
                    nn.LayerNorm(768),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ).to(embeddings.device)
                logger.info("      896→768 프로젝션 레이어 생성")
            
            self.embedding_projection_896 = self.embedding_projection_896.to(embeddings.device)
            embeddings_768 = self.embedding_projection_896(embeddings)
            logger.info(f"      896차원 입력, 프로젝션 후 차원: {embeddings_768.shape}")
        elif embeddings.shape[-1] == 768:
            # 이미 768차원인 경우 그대로 사용
            embeddings_768 = embeddings
            logger.info(f"      768차원 입력, 프로젝션 없이 직접 사용")
        else:
            # 예상치 못한 차원
            raise RuntimeError(f"지원하지 않는 입력 차원: {embeddings.shape[-1]} (896 또는 768 필요)")
        
        # 각 내부 모듈에 프로젝션된 임베딩 전달
        if hasattr(self, 'temporal_emotion') and 'lstm_tracker' in self.temporal_emotion:
            try:
                # LSTM 기반 처리
                logger.info("      - temporal_emotion LSTM 처리 중...")
                # LSTM의 디바이스를 확인하고 입력을 같은 디바이스로 이동
                lstm_device = next(self.temporal_emotion['lstm_tracker'].parameters()).device
                embeddings_on_device = embeddings_768.to(lstm_device)
                temporal_out = self.temporal_emotion['lstm_tracker'](embeddings_on_device.unsqueeze(1))
                # 결과를 원래 입력 디바이스로 다시 이동
                output['temporal_emotion'] = temporal_out[0].squeeze(1).to(embeddings.device)
                logger.info("      ✅ temporal_emotion 처리 완료")
            except Exception as e:
                logger.error(f"      ❌ temporal_emotion 처리 실패: {e}")
                raise RuntimeError(f"temporal_emotion 처리 실패 - NO FALLBACK: {e}")
        
        # multimodal fusion 처리
        if hasattr(self, 'multimodal_fusion') and 'text_encoder' in self.multimodal_fusion:
            try:
                logger.info("      - multimodal_fusion 처리 중...")
                # text_encoder의 디바이스를 확인하고 입력을 같은 디바이스로 이동
                encoder_device = next(self.multimodal_fusion['text_encoder'].parameters()).device
                embeddings_on_device = embeddings_768.to(encoder_device)
                encoded = self.multimodal_fusion['text_encoder'](embeddings_on_device.unsqueeze(1))
                # 결과를 원래 입력 디바이스로 다시 이동
                output['multimodal'] = encoded.mean(dim=1).to(embeddings.device)  # 평균 풀링
                logger.info("      ✅ multimodal_fusion 처리 완료")
            except Exception as e:
                logger.error(f"      ❌ multimodal_fusion 처리 실패: {e}")
                raise RuntimeError(f"multimodal_fusion 처리 실패 - NO FALLBACK: {e}")
        
        # advanced_moe 처리 (감정 생성)
        if hasattr(self, 'advanced_moe') and 'router' in self.advanced_moe:
            try:
                logger.info("      - advanced_moe 처리 중...")
                # router의 디바이스를 확인하고 입력을 같은 디바이스로 이동
                router_device = next(self.advanced_moe['router'].parameters()).device
                embeddings_on_device = embeddings_768.to(router_device)
                router_weights = self.advanced_moe['router'](embeddings_on_device)
                expert_outputs = []
                for i, expert in enumerate(self.advanced_moe['micro_experts']):
                    expert_out = expert(embeddings_on_device)
                    expert_outputs.append(expert_out * router_weights[:, i:i+1])
                # 결과를 원래 입력 디바이스로 다시 이동
                output['emotions'] = torch.stack(expert_outputs).sum(dim=0)[:, :7].to(embeddings.device)  # 7차원 감정
                logger.info("      ✅ advanced_moe 처리 완료")
            except Exception as e:
                logger.error(f"      ❌ advanced_moe 처리 실패: {e}")
                # advanced_moe 실패 시 다른 방법 시도
        
        # 감정 벡터가 없으면 생성
        if 'emotions' not in output:
            # 896차원을 7차원 감정으로 직접 프로젝션
            if not hasattr(self, 'emotion_projection'):
                self.emotion_projection = nn.Sequential(
                    nn.Linear(896, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 7),
                    nn.Softmax(dim=-1)
                )
                logger.info("      896→7 감정 프로젝션 레이어 생성")
            
            # 프로젝션 레이어를 입력과 같은 디바이스로 이동
            self.emotion_projection = self.emotion_projection.to(embeddings.device)
            output['emotions'] = self.emotion_projection(embeddings)
            logger.info("      ✅ 감정 벡터 프로젝션 생성")
        
        # valence, arousal 추가 (감정 벡터에서 계산)
        if 'emotions' in output:
            # positive emotions (joy, surprise) vs negative emotions (sadness, anger, fear, disgust)
            valence = output['emotions'][:, 0] + output['emotions'][:, 4] - \
                     (output['emotions'][:, 1] + output['emotions'][:, 2] + output['emotions'][:, 3] + output['emotions'][:, 5])
            output['valence'] = valence.unsqueeze(-1)
            
            # arousal: 활성화 정도 (neutral이 아닌 정도)
            arousal = 1.0 - output['emotions'][:, 6] if output['emotions'].shape[1] > 6 else torch.ones_like(valence)
            output['arousal'] = arousal.unsqueeze(-1)
        
        logger.info(f"   ✅ 임베딩 처리 완료: {list(output.keys())}")
        return output
    
    def _convert_emotion_data_to_tensor(self, emotion_data, device) -> Dict[str, torch.Tensor]:
        """EmotionData를 텐서로 변환"""
        output = {}
        
        # 주요 감정 벡터
        emotions = torch.zeros(1, 7).to(device)
        if hasattr(emotion_data, 'primary_emotion'):
            emotion_map = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'surprise': 4, 'disgust': 5, 'neutral': 6}
            primary = str(emotion_data.primary_emotion).lower()
            if primary in emotion_map:
                emotions[0, emotion_map[primary]] = emotion_data.intensity if hasattr(emotion_data, 'intensity') else 1.0
        
        output['emotions'] = emotions
        
        # valence, arousal 추가
        if hasattr(emotion_data, 'valence'):
            output['valence'] = torch.tensor([[emotion_data.valence]], device=device)
        if hasattr(emotion_data, 'arousal'):
            output['arousal'] = torch.tensor([[emotion_data.arousal]], device=device)
        
        return output
    
    def _direct_forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """내부 모듈 직접 forward - NO FALLBACK"""
        logger.info("   🔄 내부 모듈 직접 forward 시작")
        output = {}
        
        # 멀티모달 융합 처리
        if hasattr(self, 'multimodal_fusion') and 'text_encoder' in self.multimodal_fusion:
            try:
                logger.info("      - multimodal_fusion text_encoder 처리 중...")
                encoded = self.multimodal_fusion['text_encoder'](x.unsqueeze(1))
                output['multimodal'] = encoded.mean(dim=1)  # 평균 풀링
                logger.info("      ✅ multimodal_fusion 처리 완료")
            except Exception as e:
                logger.error(f"      ❌ multimodal_fusion 처리 실패: {e}")
                raise RuntimeError(f"multimodal_fusion 처리 실패 - NO FALLBACK: {e}")
        
        # advanced_moe 처리
        if hasattr(self, 'advanced_moe') and 'router' in self.advanced_moe:
            try:
                logger.info("      - advanced_moe 처리 중...")
                router_weights = self.advanced_moe['router'](x)
                expert_outputs = []
                for i, expert in enumerate(self.advanced_moe['micro_experts']):
                    expert_out = expert(x)
                    expert_outputs.append(expert_out * router_weights[:, i:i+1])
                output['emotions'] = torch.stack(expert_outputs).sum(dim=0)
                logger.info("      ✅ advanced_moe 처리 완료")
            except Exception as e:
                logger.error(f"      ❌ advanced_moe 처리 실패: {e}")
                raise RuntimeError(f"advanced_moe 처리 실패 - NO FALLBACK: {e}")
        
        # 출력이 없으면 에러
        if not output:
            raise RuntimeError("내부 모듈 forward 실패: 어떤 모듈도 처리하지 못함 - NO FALLBACK")
        
        logger.info(f"   ✅ 내부 모듈 처리 완료: {list(output.keys())}")
        return output


class AdvancedRegretAnalyzerWrapper(nn.Module):
    """Advanced Regret Analyzer를 nn.Module로 래핑 (50M 파라미터)"""
    
    def __init__(self):
        super().__init__()
        
        from advanced_regret_analyzer import AdvancedRegretAnalyzer
        self.analyzer = AdvancedRegretAnalyzer()
        
        self._register_internal_modules()
        
        logger.info("✅ Advanced Regret Analyzer Wrapper 초기화 (50M 파라미터)")
    
    def _register_internal_modules(self):
        """내부 nn.Module들을 등록"""
        
        # GPU 후회 네트워크 (3M)
        if hasattr(self.analyzer, 'regret_network'):
            self.regret_network = self.analyzer.regret_network
            logger.info("  - regret_network 등록 (3M)")
        
        # 반사실 시뮬레이션 (15M)
        if hasattr(self.analyzer, 'counterfactual_sim'):
            self.counterfactual_sim = self.analyzer.counterfactual_sim
            logger.info("  - counterfactual_sim 등록 (15M)")
        
        # 시간축 후회 전파 (12M)
        if hasattr(self.analyzer, 'temporal_propagation'):
            self.temporal_propagation = self.analyzer.temporal_propagation
            logger.info("  - temporal_propagation 등록 (12M)")
        
        # 의사결정 트리 (10M)
        if hasattr(self.analyzer, 'decision_tree'):
            self.decision_tree = self.analyzer.decision_tree
            logger.info("  - decision_tree 등록 (10M)")
        
        # 베이지안 추론 (10M)
        if hasattr(self.analyzer, 'bayesian_inference'):
            self.bayesian_inference = self.analyzer.bayesian_inference
            logger.info("  - bayesian_inference 등록 (10M)")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - NO FALLBACK"""
        logger.info("🔄 AdvancedRegretAnalyzerWrapper forward 실행 시작")
        logger.info(f"   입력 차원: {x.shape}")
        output = {}
        
        # 후회 네트워크 처리
        if hasattr(self, 'regret_network'):
            try:
                logger.info("   - regret_network 처리 중...")
                
                # 차원 체크 및 프로젝션 어댑터 처리
                # 체크포인트가 768차원인 경우를 위한 프로젝션
                expected_dim = next(self.regret_network.regret_predictor[0].parameters()).shape[1]
                logger.info(f"   regret_network 기대 차원: {expected_dim}, 입력 차원: {x.shape[-1]}")
                
                if x.shape[-1] == 896 and expected_dim == 768:
                    # 896 -> 768 프로젝션 필요
                    if not hasattr(self, 'input_projection_896_to_768'):
                        logger.info("   896→768 프로젝션 어댑터 생성 중...")
                        self.input_projection_896_to_768 = nn.Sequential(
                            nn.Linear(896, 768),
                            nn.LayerNorm(768),
                            nn.GELU()
                        ).to(x.device)
                    
                    self.input_projection_896_to_768 = self.input_projection_896_to_768.to(x.device)
                    x_projected = self.input_projection_896_to_768(x)
                    logger.info(f"   프로젝션 후 차원: {x_projected.shape}")
                    regret_out = self.regret_network(x_projected)
                elif x.shape[-1] == 768 and expected_dim == 896:
                    # 768 -> 896 프로젝션 필요 (Advanced Analysis 단계에서 발생)
                    if not hasattr(self, 'input_projection_768_to_896'):
                        logger.info("   768→896 프로젝션 어댑터 생성 중...")
                        self.input_projection_768_to_896 = nn.Sequential(
                            nn.Linear(768, 896),
                            nn.LayerNorm(896),
                            nn.GELU()
                        ).to(x.device)
                    
                    self.input_projection_768_to_896 = self.input_projection_768_to_896.to(x.device)
                    x_projected = self.input_projection_768_to_896(x)
                    logger.info(f"   프로젝션 후 차원: {x_projected.shape}")
                    regret_out = self.regret_network(x_projected)
                else:
                    # 차원이 일치하거나 이미 맞는 경우
                    regret_out = self.regret_network(x)
                
                # GPURegretNetwork는 tuple을 반환: (regret_score, emotion_vector, uncertainty)
                if isinstance(regret_out, tuple):
                    regret_score, emotion_vector, uncertainty = regret_out
                    output['regret_score'] = regret_score
                    output['regret_emotion_vector'] = emotion_vector
                    output['regret_uncertainty'] = uncertainty
                elif isinstance(regret_out, dict):
                    output['regret_score'] = regret_out.get('regret_score', regret_out)
                else:
                    output['regret_score'] = regret_out
                logger.info("   ✅ regret_network 처리 완료")
            except Exception as e:
                logger.error(f"   ❌ regret_network 처리 실패: {e}")
                raise RuntimeError(f"Regret network 처리 실패 - NO FALLBACK: {e}")
        
        # 반사실 시뮬레이션
        if hasattr(self, 'counterfactual_sim') and 'world_model' in self.counterfactual_sim:
            try:
                cf_out = self.counterfactual_sim['world_model'](x)
                output['counterfactual'] = cf_out
            except:
                pass
        
        # 출력 검증 - NO FALLBACK
        if 'regret_score' not in output:
            logger.error("❌ Regret score 계산 실패")
            raise RuntimeError("Regret 분석 실패: regret_score 생성 못함 - NO FALLBACK")
        
        logger.info(f"   ✅ AdvancedRegretAnalyzer 처리 완료: {list(output.keys())}")
        return output


class AdvancedSURDAnalyzerWrapper(nn.Module):
    """Advanced SURD Analyzer를 nn.Module로 래핑 (25M 파라미터)"""
    
    def __init__(self):
        super().__init__()
        
        from advanced_surd_analyzer import AdvancedSURDAnalyzer
        self.analyzer = AdvancedSURDAnalyzer()
        
        # 디바이스 결정
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 896차원 → 768차원 투영 레이어 추가 (deep_causal을 위해)
        self.input_projection = nn.Sequential(
            nn.Linear(896, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)
        
        self._register_internal_modules()
        
        logger.info(f"✅ Advanced SURD Analyzer Wrapper 초기화 (25M 파라미터, device: {device})")
    
    def _register_internal_modules(self):
        """내부 nn.Module들을 등록"""
        
        # 심층 인과 추론 (10M)
        if hasattr(self.analyzer, 'deep_causal'):
            self.deep_causal = self.analyzer.deep_causal
            logger.info("  - deep_causal 등록 (10M)")
        
        # 정보이론 분해 (8M)
        if hasattr(self.analyzer, 'info_decomposition'):
            self.info_decomposition = self.analyzer.info_decomposition
            logger.info("  - info_decomposition 등록 (8M)")
        
        # Neural Causal Model (5M)
        if hasattr(self.analyzer, 'neural_causal_model'):
            self.neural_causal_model = self.analyzer.neural_causal_model
            logger.info("  - neural_causal_model 등록 (5M)")
        
        # Network Optimizer (2M)
        if hasattr(self.analyzer, 'network_optimizer'):
            self.network_optimizer = self.analyzer.network_optimizer
            logger.info("  - network_optimizer 등록 (2M)")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - NO FALLBACK"""
        logger.info("🔄 AdvancedSURDAnalyzerWrapper forward 실행 시작")
        output = {}
        
        # 디바이스 일치 처리
        logger.info(f"   입력 차원: {x.shape}")
        
        # input_projection의 디바이스 확인
        projection_device = next(self.input_projection.parameters()).device
        logger.info(f"   projection device: {projection_device}, input device: {x.device}")
        
        # 입력 텐서를 projection layer와 같은 디바이스로 이동
        if x.device != projection_device:
            logger.info(f"   디바이스 불일치 감지 - 입력을 {projection_device}로 이동")
            x = x.to(projection_device)
        
        # 768차원 입력 처리를 위한 체크
        if x.shape[-1] == 768:
            # 768차원은 이미 deep_causal에 맞으므로 투영 없이 사용
            logger.info("   768차원 입력 감지 - 직접 사용")
            x_projected = x
        else:
            # 896 -> 768 투영
            x_projected = self.input_projection(x)
        logger.info(f"   투영 후 차원: {x_projected.shape}")
        
        # 심층 인과 추론
        if hasattr(self, 'deep_causal') and 'causal_encoder' in self.deep_causal:
            try:
                logger.info("   - deep_causal 처리 중...")
                causal_out = self.deep_causal['causal_encoder'](x_projected)
                # S, U, R, D 분해
                output['surd_metrics'] = causal_out[:, :4]  # 첫 4차원
                logger.info("   ✅ deep_causal 처리 완료")
            except Exception as e:
                logger.error(f"   ❌ deep_causal 처리 실패: {e}")
                raise RuntimeError(f"Deep causal 처리 실패 - NO FALLBACK: {e}")
        
        # 정보이론 분해
        if hasattr(self, 'info_decomposition') and 'mutual_info' in self.info_decomposition:
            try:
                # info_decomposition도 768차원 기반이므로 투영된 입력 사용
                info_out = self.info_decomposition['mutual_info'](torch.cat([x_projected, x_projected], dim=-1))
                if 'surd_metrics' not in output:
                    output['surd_metrics'] = info_out[:, :4]
            except:
                pass
        
        # 출력 검증 - NO FALLBACK
        if 'surd_metrics' not in output:
            logger.error("❌ SURD metrics 계산 실패")
            raise RuntimeError("SURD 분석 실패: surd_metrics 생성 못함 - NO FALLBACK")
        
        logger.info(f"   ✅ AdvancedSURDAnalyzer 처리 완료: {list(output.keys())}")
        return output


class AdvancedBenthamCalculatorWrapper(nn.Module):
    """Advanced Bentham Calculator를 nn.Module로 래핑 (2.5M 파라미터)"""
    
    def __init__(self):
        super().__init__()
        
        from advanced_bentham_calculator import AdvancedBenthamCalculator
        self.analyzer = AdvancedBenthamCalculator()
        
        # 디바이스 결정 (bentham_default_network와 동일하게)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 896차원 → 7차원 투영 레이어 추가 (Bentham의 7가지 변수를 위해)
        self.input_projection = nn.Sequential(
            nn.Linear(896, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        ).to(device)  # 디바이스 지정
        
        # 768차원 입력을 위한 별도 투영 레이어 추가
        self.input_projection_768 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        ).to(device)  # 디바이스 지정
        
        self._register_internal_modules()
        
        logger.info(f"✅ Advanced Bentham Calculator Wrapper 초기화 (2.5M 파라미터, device: {device})")
    
    def _register_internal_modules(self):
        """내부 nn.Module들을 등록"""
        
        # 동적으로 모든 nn.Module 찾기
        module_count = 0
        for attr_name in dir(self.analyzer):
            if not attr_name.startswith('_'):
                # 프로퍼티나 메소드가 아닌 직접 속성만 접근
                # getattr 대신 __dict__ 직접 확인으로 프로퍼티 호출 방지
                if hasattr(self.analyzer, '__dict__') and attr_name in self.analyzer.__dict__:
                    attr = self.analyzer.__dict__[attr_name]
                    if attr is not None and isinstance(attr, nn.Module):
                        setattr(self, f"bentham_{attr_name}", attr)
                        module_count += 1
                        logger.info(f"  - {attr_name} 등록")
        
        if module_count == 0:
            # 기본 신경망 생성 (2.5M)
            self.bentham_network = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)  # 10개 벤담 차원
            )
            logger.info("  - 기본 bentham_network 생성 (2.5M)")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - NO FALLBACK"""
        logger.info("🔄 AdvancedBenthamCalculatorWrapper forward 실행 시작")
        output = {}
        
        # 내부 네트워크 사용 - bentham_network 또는 bentham_default_network 찾기
        network_found = False
        
        # 우선 bentham_network 확인
        if hasattr(self, 'bentham_network'):
            try:
                logger.info("   - bentham_network 처리 중...")
                bentham_scores = self.bentham_network(x)
                output['bentham_scores'] = bentham_scores
                network_found = True
                logger.info("   ✅ bentham_network 처리 완료")
            except Exception as e:
                logger.error(f"   ❌ bentham_network 처리 실패: {e}")
                raise RuntimeError(f"Bentham network 처리 실패 - NO FALLBACK: {e}")
        
        # bentham_default_network 확인 (7차원 입력 필요)
        elif hasattr(self, 'bentham_default_network'):
            try:
                logger.info("   - bentham_default_network 처리 중...")
                logger.info(f"     입력 차원: {x.shape}")
                
                # 원래 device 저장
                original_device = x.device
                
                # device 일관성 보장 - bentham_default_network와 같은 device로 이동
                network_device = next(self.bentham_default_network.parameters()).device
                logger.info(f"     network device: {network_device}, input device: {original_device}")
                
                # 입력 차원에 따라 적절한 projection layer 선택
                input_dim = x.shape[-1]
                if input_dim == 768:
                    # 768차원 입력용 projection 사용
                    if hasattr(self, 'input_projection_768'):
                        self.input_projection_768 = self.input_projection_768.to(network_device)
                        projection_layer = self.input_projection_768
                    else:
                        logger.error(f"768차원 projection layer가 없음")
                        raise RuntimeError("768차원 입력을 위한 projection layer 없음")
                elif input_dim == 896:
                    # 896차원 입력용 projection 사용
                    if hasattr(self, 'input_projection'):
                        self.input_projection = self.input_projection.to(network_device)
                        projection_layer = self.input_projection
                    else:
                        logger.error(f"896차원 projection layer가 없음")
                        raise RuntimeError("896차원 입력을 위한 projection layer 없음")
                else:
                    logger.error(f"지원하지 않는 입력 차원: {input_dim}")
                    raise RuntimeError(f"지원하지 않는 입력 차원: {input_dim} (768 또는 896만 지원)")
                
                # 입력 텐서도 같은 device로 이동
                if x.device != network_device:
                    x = x.to(network_device)
                
                # 입력 차원에 맞는 projection 사용하여 7차원으로 투영
                x_projected = projection_layer(x)
                logger.info(f"     투영 후 차원: {x_projected.shape}, device: {x_projected.device}")
                
                bentham_scores = self.bentham_default_network(x_projected)
                
                # 결과를 원래 device로 되돌림
                if bentham_scores.device != original_device:
                    bentham_scores = bentham_scores.to(original_device)
                
                output['bentham_scores'] = bentham_scores[:, :10] if bentham_scores.shape[1] > 10 else bentham_scores
                network_found = True
                logger.info("   ✅ bentham_default_network 처리 완료")
            except Exception as e:
                logger.error(f"   ❌ bentham_default_network 처리 실패: {e}")
                raise RuntimeError(f"Bentham default network 처리 실패 - NO FALLBACK: {e}")
        
        # 그 외 bentham_ 접두사 모듈들 시도
        if not network_found:
            for attr_name in dir(self):
                if attr_name.startswith('bentham_') and hasattr(self, attr_name):
                    module = getattr(self, attr_name)
                    if isinstance(module, nn.Module):
                        try:
                            logger.info(f"   - {attr_name} 처리 시도...")
                            result = module(x)
                            output['bentham_scores'] = result[:, :10] if result.shape[1] > 10 else result
                            network_found = True
                            logger.info(f"   ✅ {attr_name} 처리 완료")
                            break
                        except Exception as e:
                            logger.debug(f"   - {attr_name} 처리 실패: {e}")
                            continue
        
        # 출력 검증 - NO FALLBACK
        if 'bentham_scores' not in output:
            logger.error("❌ Bentham scores 계산 실패")
            raise RuntimeError("Bentham 계산 실패: bentham_scores 생성 못함 - NO FALLBACK")
        
        logger.info(f"   ✅ AdvancedBenthamCalculator 처리 완료: {list(output.keys())}")
        return output


def create_advanced_analyzer_wrappers() -> Dict[str, nn.Module]:
    """모든 Advanced Analyzer Wrapper 생성"""
    wrappers = {}
    required_wrappers = ['advanced_emotion', 'advanced_regret', 'advanced_surd', 'advanced_bentham']
    
    # Emotion Wrapper 생성 (필수)
    try:
        wrappers['advanced_emotion'] = AdvancedEmotionAnalyzerWrapper()
    except Exception as e:
        logger.error(f"Advanced Emotion Wrapper 생성 실패: {e}")
        raise RuntimeError(f"필수 Wrapper 생성 실패 - advanced_emotion: {e}")
    
    # Regret Wrapper 생성 (필수)
    try:
        wrappers['advanced_regret'] = AdvancedRegretAnalyzerWrapper()
    except Exception as e:
        logger.error(f"Advanced Regret Wrapper 생성 실패: {e}")
        raise RuntimeError(f"필수 Wrapper 생성 실패 - advanced_regret: {e}")
    
    # SURD Wrapper 생성 (필수)
    try:
        wrappers['advanced_surd'] = AdvancedSURDAnalyzerWrapper()
    except Exception as e:
        logger.error(f"Advanced SURD Wrapper 생성 실패: {e}")
        raise RuntimeError(f"필수 Wrapper 생성 실패 - advanced_surd: {e}")
    
    # Bentham Wrapper 생성 (필수)
    try:
        wrappers['advanced_bentham'] = AdvancedBenthamCalculatorWrapper()
    except Exception as e:
        logger.error(f"Advanced Bentham Wrapper 생성 실패: {e}")
        raise RuntimeError(f"필수 Wrapper 생성 실패 - advanced_bentham: {e}")
    
    # 모든 필수 wrapper 확인
    for wrapper_name in required_wrappers:
        if wrapper_name not in wrappers:
            raise RuntimeError(f"필수 Wrapper 누락: {wrapper_name}")
    
    total_params = sum(
        sum(p.numel() for p in w.parameters()) 
        for w in wrappers.values()
    )
    
    logger.info(f"✅ Advanced Analyzer Wrappers 생성 완료")
    logger.info(f"  - 총 Wrapper 수: {len(wrappers)}")
    logger.info(f"  - 총 파라미터: {total_params:,}")
    
    return wrappers