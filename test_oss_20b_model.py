#!/usr/bin/env python3
"""
GPT-OSS-20B 모델 로드 및 기본 테스트
"""

import torch
import sys
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
from typing import List, Dict, Any

def get_memory_stats():
    """메모리 상태 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CPU mode"

def load_model_with_optimization(model_path: str):
    """메모리 최적화와 함께 모델 로드"""
    print(f"모델 로드 시작: {model_path}")
    print(f"초기 메모리: {get_memory_stats()}")
    
    # Tokenizer 로드
    print("Tokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 모델 설정
    print("모델 설정 준비 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 로드 옵션
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    
    # 8bit 양자화 시도 (GPU 메모리 제한 대응)
    if device == "cuda":
        try:
            import bitsandbytes as bnb
            load_kwargs["load_in_8bit"] = True
            print("8bit 양자화 활성화")
        except ImportError:
            print("bitsandbytes 없음 - 16bit 모드로 진행")
    
    # 모델 로드
    print("모델 로드 중 (시간이 걸릴 수 있습니다)...")
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        load_time = time.time() - start_time
        print(f"모델 로드 완료! ({load_time:.1f}초)")
        print(f"로드 후 메모리: {get_memory_stats()}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        
        # CPU 모드로 재시도
        if device == "cuda":
            print("\nCPU 모드로 재시도...")
            load_kwargs["device_map"] = None
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs.pop("load_in_8bit", None)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            device = "cpu"
            print("CPU 모드로 로드 성공")
            
            return model, tokenizer, device
        else:
            raise

def test_model_generation(model, tokenizer, device):
    """모델 생성 테스트"""
    print("\n=== 모델 생성 테스트 ===")
    
    test_prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "def fibonacci(n):",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n테스트 {i}: {prompt}")
        
        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            gen_time = time.time() - start_time
        
        # 디코딩
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"생성 결과: {generated}")
        print(f"생성 시간: {gen_time:.2f}초")
        print(f"메모리: {get_memory_stats()}")

def test_emotion_analysis(model, tokenizer, device):
    """감정 분석 테스트"""
    print("\n=== 감정 분석 프롬프트 테스트 ===")
    
    test_text = "I can't believe my friend betrayed me like that. I trusted them completely."
    
    prompt = f"""Analyze the emotions in this text and return ONLY a JSON array with 7 float values (0.0-1.0) for:
[Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger]

Text: {test_text}

Response (JSON array only):"""
    
    print(f"테스트 텍스트: {test_text}")
    
    # 토큰화
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.85,
            do_sample=True,
            top_p=0.98,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # 디코딩
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"모델 응답: {generated}")
    
    # JSON 파싱 시도
    try:
        import re
        json_match = re.search(r'\[[\d\.,\s]+\]', generated)
        if json_match:
            emotion_vector = json.loads(json_match.group())
            print(f"파싱된 감정 벡터: {emotion_vector}")
            if len(emotion_vector) == 7:
                emotions = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger"]
                for emotion, value in zip(emotions, emotion_vector):
                    print(f"  {emotion}: {value:.2f}")
        else:
            print("JSON 파싱 실패")
    except Exception as e:
        print(f"파싱 에러: {e}")

def main():
    """메인 테스트 함수"""
    model_path = "/mnt/c/large_project/linux_red_heart/llm_oss_20b_model"
    
    # CUDA 확인
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 디바이스: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # 모델 로드
        model, tokenizer, device = load_model_with_optimization(model_path)
        
        # 기본 생성 테스트
        test_model_generation(model, tokenizer, device)
        
        # 감정 분석 테스트
        test_emotion_analysis(model, tokenizer, device)
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 메모리 정리
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n최종 메모리: {get_memory_stats()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())