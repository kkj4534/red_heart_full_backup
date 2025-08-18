#!/usr/bin/env python3
"""
Red Heart AI 필수 모델 사전 다운로드 스크립트
Pre-download essential models for Red Heart AI

테스트 및 학습 시 온라인 다운로드 없이 로컬 캐시에서 사용할 수 있도록
필요한 transformers 및 sentence-transformers 모델들을 미리 다운로드합니다.
"""

import os
import sys
import logging
from pathlib import Path
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_transformers_models():
    """Transformers 모델들 다운로드"""
    
    print("🚀 Transformers 모델 다운로드 시작...")
    
    models_to_download = [
        "Helsinki-NLP/opus-mt-ko-en",  # 한국어-영어 번역
        "jhgan/ko-sroberta-multitask",  # 한국어 문장 임베딩
        "beomi/KcELECTRA-base-v2022",  # 한국어 ELECTRA
        "j-hartmann/emotion-english-distilroberta-base",  # 감정 분석
        "klue/bert-base-kor-ner"  # 한국어 NER
    ]
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        from transformers import MarianTokenizer, MarianMTModel
        
        for i, model_name in enumerate(models_to_download, 1):
            print(f"\n📥 [{i}/{len(models_to_download)}] {model_name} 다운로드 중...")
            start_time = time.time()
            
            try:
                # 모델별 특별 처리
                if "opus-mt" in model_name:
                    # 번역 모델
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    print(f"   ✅ 번역 모델 다운로드 완료")
                elif "emotion" in model_name:
                    # 감정 분석 모델
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    print(f"   ✅ 감정 분석 모델 다운로드 완료")
                else:
                    # 일반 모델
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    print(f"   ✅ 일반 모델 다운로드 완료")
                
                elapsed = time.time() - start_time
                print(f"   ⏱️ 소요 시간: {elapsed:.1f}초")
                
            except Exception as e:
                print(f"   ❌ {model_name} 다운로드 실패: {e}")
                logger.error(f"모델 다운로드 실패: {model_name} - {e}")
                continue
        
        print(f"\n✅ Transformers 모델 다운로드 완료!")
        
    except ImportError as e:
        print(f"❌ transformers 라이브러리가 필요합니다: {e}")
        return False
    
    return True

def download_sentence_transformers():
    """SentenceTransformer 모델들 다운로드"""
    
    print("\n🔤 SentenceTransformer 모델 다운로드 시작...")
    
    sentence_models = [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 다국어 문장 임베딩
        "jhgan/ko-sroberta-multitask"  # 한국어 문장 임베딩 (SentenceTransformer 버전)
    ]
    
    try:
        from sentence_transformers import SentenceTransformer
        
        for i, model_name in enumerate(sentence_models, 1):
            print(f"\n📥 [{i}/{len(sentence_models)}] {model_name} 다운로드 중...")
            start_time = time.time()
            
            try:
                # SentenceTransformer 모델 다운로드
                model = SentenceTransformer(model_name)
                
                # 간단한 테스트로 모델 로딩 확인
                test_encoding = model.encode(["테스트 문장"])
                print(f"   ✅ SentenceTransformer 모델 다운로드 및 테스트 완료")
                print(f"   📊 임베딩 차원: {test_encoding.shape[1]}")
                
                elapsed = time.time() - start_time
                print(f"   ⏱️ 소요 시간: {elapsed:.1f}초")
                
            except Exception as e:
                print(f"   ❌ {model_name} 다운로드 실패: {e}")
                logger.error(f"SentenceTransformer 다운로드 실패: {model_name} - {e}")
                continue
        
        print(f"\n✅ SentenceTransformer 모델 다운로드 완료!")
        
    except ImportError as e:
        print(f"❌ sentence-transformers 라이브러리가 필요합니다: {e}")
        return False
    
    return True

def check_cache_status():
    """캐시 상태 확인"""
    print("\n📂 캐시 상태 확인...")
    
    cache_paths = [
        Path.home() / '.cache' / 'huggingface' / 'transformers',
        Path.home() / '.cache' / 'torch' / 'sentence_transformers'
    ]
    
    total_size = 0
    for cache_path in cache_paths:
        if cache_path.exists():
            # 캐시 크기 계산
            size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            total_size += size_mb
            
            print(f"   📁 {cache_path}: {size_mb:.1f} MB")
        else:
            print(f"   📁 {cache_path}: 존재하지 않음")
    
    print(f"   💾 총 캐시 크기: {total_size:.1f} MB")
    return total_size

def main():
    """메인 실행"""
    print("=" * 60)
    print("🚀 Red Heart AI 필수 모델 사전 다운로드")
    print("=" * 60)
    
    print("\n📋 다운로드할 모델 목록:")
    print("   1. Helsinki-NLP/opus-mt-ko-en (번역)")
    print("   2. jhgan/ko-sroberta-multitask (한국어 임베딩)")
    print("   3. beomi/KcELECTRA-base-v2022 (한국어)")
    print("   4. j-hartmann/emotion-english-distilroberta-base (감정 분석)")
    print("   5. klue/bert-base-kor-ner (한국어 NER)")
    print("   6. sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (다국어)")
    
    # 시작 전 캐시 상태
    print("\n" + "=" * 40)
    print("시작 전 캐시 상태")
    print("=" * 40)
    initial_cache_size = check_cache_status()
    
    # 다운로드 시작
    start_time = time.time()
    
    # 1. Transformers 모델 다운로드
    transformers_success = download_transformers_models()
    
    # 2. SentenceTransformer 모델 다운로드  
    sentence_success = download_sentence_transformers()
    
    # 최종 결과
    total_time = time.time() - start_time
    
    print("\n" + "=" * 40)
    print("다운로드 완료 후 캐시 상태")
    print("=" * 40)
    final_cache_size = check_cache_status()
    
    print("\n" + "=" * 60)
    print("🎉 모델 다운로드 완료!")
    print("=" * 60)
    print(f"   ⏱️ 총 소요 시간: {total_time:.1f}초")
    print(f"   📈 캐시 증가량: {final_cache_size - initial_cache_size:.1f} MB")
    print(f"   🔧 Transformers: {'✅ 성공' if transformers_success else '❌ 실패'}")
    print(f"   🔤 SentenceTransformers: {'✅ 성공' if sentence_success else '❌ 실패'}")
    
    if transformers_success and sentence_success:
        print("\n✅ 모든 모델이 성공적으로 다운로드되었습니다!")
        print("   이제 unified-test를 온라인 다운로드 없이 실행할 수 있습니다.")
        print("\n💡 다음 명령어로 테스트하세요:")
        print("   ./run_learning.sh unified-test --samples 3 --debug --verbose")
        return 0
    else:
        print("\n⚠️ 일부 모델 다운로드에 실패했습니다.")
        print("   네트워크 연결을 확인하고 다시 시도하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)