#!/usr/bin/env python3
"""
GPT-OSS-20B 하이브리드 전처리 파이프라인
8GB VRAM 제한으로 CPU+GPU 하이브리드 모드 사용
"""

import os
import json
import torch
import numpy as np
import logging
import gc
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RedHeart.OSS20B.Hybrid')

class OSS20BHybridProcessor:
    """GPT-OSS-20B 하이브리드 전처리기"""
    
    def __init__(self):
        """초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"디바이스: {self.device}")
        
        # GPU 메모리 상태 확인
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU 메모리: {gpu_memory:.2f} GB")
            
            # 현재 사용 중인 메모리
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"현재 사용: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        self.model = None
        self.model_type = None
        self.is_initialized = False
        
    def check_available_models(self):
        """사용 가능한 모델 옵션 확인"""
        options = []
        
        # 1. GGUF 모델 (llama-cpp-python 필요)
        try:
            from llama_cpp import Llama
            logger.info("✅ llama-cpp-python 사용 가능")
            
            # 가장 작은 GGUF 모델 사용 (IQ2_XS)
            gguf_path = "/mnt/c/large_project/linux_red_heart/llm_oss_20b_gguf/openai_gpt-oss-20b-IQ2_XS.gguf"
            if os.path.exists(gguf_path):
                options.append(("gguf", gguf_path))
                logger.info(f"✅ GGUF 모델 발견: {gguf_path}")
            else:
                logger.info("❌ GGUF 모델 없음")
                
        except ImportError:
            logger.info("❌ llama-cpp-python 없음")
        
        # 2. Transformers 8bit/4bit 양자화
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import bitsandbytes as bnb
            logger.info("✅ bitsandbytes 사용 가능 (8bit/4bit 양자화)")
            
            safetensors_path = "/mnt/c/large_project/linux_red_heart/llm_oss_20b_model"
            if os.path.exists(safetensors_path):
                options.append(("transformers_quantized", safetensors_path))
                logger.info(f"✅ Safetensors 모델 발견: {safetensors_path}")
                
        except ImportError as e:
            logger.info(f"❌ bitsandbytes 없음: {e}")
        
        # 3. 대안: 더 작은 모델 제안
        logger.info("\n=== 대안 제안 ===")
        logger.info("1. Mistral-7B-Instruct (7B, 8GB VRAM 가능)")
        logger.info("2. Llama-3-8B (8B, 8GB VRAM 가능)")
        logger.info("3. Phi-3-mini (3.8B, 매우 효율적)")
        
        return options
    
    def load_gguf_hybrid(self, model_path: str):
        """GGUF 모델을 하이브리드 모드로 로드"""
        from llama_cpp import Llama
        
        # 8GB VRAM 최적화: 매우 적은 레이어만 GPU에
        # OSS-20B는 ~80개 레이어, 5-10개만 GPU에 올림
        n_gpu_layers = 8  # 매우 보수적으로 시작
        
        logger.info(f"GGUF 하이브리드 로드 시작 (GPU layers: {n_gpu_layers})")
        
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,  # 컨텍스트 줄임
                n_batch=128,  # 배치 크기 줄임
                n_threads=8,  # CPU 스레드 늘림
                n_gpu_layers=n_gpu_layers,
                verbose=True
            )
            self.model_type = "gguf"
            self.is_initialized = True
            logger.info("✅ GGUF 모델 하이브리드 로드 성공")
            
            # 테스트 생성
            test_output = self.model.create_completion(
                "Hello, world!",
                max_tokens=10,
                temperature=0.7
            )
            logger.info(f"테스트 성공: {test_output['choices'][0]['text'][:50]}")
            
        except Exception as e:
            logger.error(f"GGUF 로드 실패: {e}")
            
            # GPU 레이어 더 줄여서 재시도
            if n_gpu_layers > 0:
                logger.info("GPU 레이어 0으로 재시도 (CPU 전용)")
                try:
                    self.model = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_batch=64,
                        n_threads=8,
                        n_gpu_layers=0,  # CPU 전용
                        verbose=True
                    )
                    self.model_type = "gguf_cpu"
                    self.is_initialized = True
                    logger.info("✅ CPU 전용 모드로 로드 성공")
                except Exception as e2:
                    logger.error(f"CPU 모드도 실패: {e2}")
                    raise
    
    def load_transformers_8bit(self, model_path: str):
        """Transformers 모델을 8bit 양자화로 로드"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info("Transformers 8bit 양자화 로드 시작")
        
        # 8bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 모델 로드 (8bit)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",  # 자동으로 GPU/CPU 분배
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            self.model_type = "transformers_8bit"
            self.is_initialized = True
            logger.info("✅ 8bit 양자화 로드 성공")
            
            # 메모리 상태
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"GPU 메모리 사용: {allocated:.2f} GB")
                
        except Exception as e:
            logger.error(f"8bit 로드 실패: {e}")
            
            # 4bit 재시도
            logger.info("4bit 양자화로 재시도")
            self.load_transformers_4bit(model_path)
    
    def load_transformers_4bit(self, model_path: str):
        """Transformers 모델을 4bit 양자화로 로드"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info("Transformers 4bit 양자화 로드 시작")
        
        # 4bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 모델 로드 (4bit)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            self.model_type = "transformers_4bit"
            self.is_initialized = True
            logger.info("✅ 4bit 양자화 로드 성공")
            
            # 메모리 상태
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"GPU 메모리 사용: {allocated:.2f} GB")
                
        except Exception as e:
            logger.error(f"4bit 로드도 실패: {e}")
            raise RuntimeError("모든 로드 방법 실패")
    
    def generate_emotion_analysis(self, text: str, retry_count: int = 0) -> Optional[List[float]]:
        """감정 분석 생성"""
        if not self.is_initialized:
            raise RuntimeError("모델이 초기화되지 않음")
        
        # Chain of Thought 프롬프트
        prompt = f"""<|im_start|>system
You are an advanced emotion analysis system. Analyze emotions and return ONLY a JSON array with 7 float values.
<|im_end|>
<|im_start|>user
Analyze the emotions in this text. Return ONLY a JSON array with 7 values (0.0-1.0) for:
[Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger]

Text: {text}

Think step by step:
1. What is the main emotion expressed?
2. What secondary emotions are present?
3. Rate each emotion from 0.0 to 1.0

Response (JSON array only):
<|im_end|>
<|im_start|>assistant
"""
        
        try:
            if self.model_type in ["gguf", "gguf_cpu"]:
                # GGUF 생성
                output = self.model.create_completion(
                    prompt,
                    max_tokens=150,
                    temperature=0.85 + (retry_count * 0.05),  # 재시도마다 온도 증가
                    top_p=0.98,
                    stop=["<|im_end|>"]
                )
                generated = output['choices'][0]['text']
                
            else:  # transformers
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.85 + (retry_count * 0.05),
                        top_p=0.98,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
            
            # JSON 파싱
            json_match = re.search(r'\[[\d\.,\s]+\]', generated)
            if json_match:
                emotion_vector = json.loads(json_match.group())
                if len(emotion_vector) == 7:
                    logger.info(f"✅ 감정 벡터 생성 성공: {emotion_vector}")
                    return emotion_vector
            
            logger.warning(f"파싱 실패: {generated[:100]}")
            return None
            
        except Exception as e:
            logger.error(f"생성 실패: {e}")
            return None
    
    def process_sample(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        """단일 샘플 처리"""
        for retry in range(max_retries):
            emotion_vector = self.generate_emotion_analysis(text, retry)
            
            if emotion_vector:
                # 템플릿 응답 검증
                if emotion_vector == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]:  # Trust 100%
                    logger.warning(f"템플릿 응답 감지, 재시도 {retry+1}/{max_retries}")
                    continue
                    
                return {
                    'success': True,
                    'emotions': emotion_vector,
                    'retry_count': retry
                }
        
        return {
            'success': False,
            'emotions': None,
            'retry_count': max_retries
        }

def main():
    """메인 테스트"""
    processor = OSS20BHybridProcessor()
    
    # 사용 가능한 옵션 확인
    options = processor.check_available_models()
    
    if not options:
        logger.error("사용 가능한 모델 옵션 없음")
        
        # 대안: GGUF 다운로드 제안
        logger.info("\n=== GGUF 다운로드 명령 ===")
        logger.info("huggingface-cli download bartowski/openai_gpt-oss-20b-GGUF openai_gpt-oss-20b-IQ2_XS.gguf --local-dir ./llm_oss_20b_gguf/")
        return
    
    # 첫 번째 옵션으로 로드 시도
    model_type, model_path = options[0]
    
    if model_type == "gguf":
        processor.load_gguf_hybrid(model_path)
    elif model_type == "transformers_quantized":
        processor.load_transformers_8bit(model_path)
    
    # 테스트
    if processor.is_initialized:
        test_texts = [
            "I'm so disappointed in myself for trusting them.",
            "This is absolutely amazing! I can't believe it worked!",
            "I feel anxious about the upcoming presentation."
        ]
        
        for text in test_texts:
            logger.info(f"\n테스트: {text}")
            result = processor.process_sample(text)
            if result['success']:
                logger.info(f"결과: {result['emotions']}")
            else:
                logger.error("처리 실패")

if __name__ == "__main__":
    main()