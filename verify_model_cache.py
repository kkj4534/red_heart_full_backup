#!/usr/bin/env python3
"""
모델 캐시 상태 검증 스크립트
Verify Model Cache Status Script

다운로드된 모델들이 실제로 오프라인에서 로드 가능한지 검증합니다.
"""

import os
import sys
import logging
from pathlib import Path
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_offline_mode():
    """강력한 오프라인 모드 설정"""
    offline_env = {
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1', 
        'HF_DATASETS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    for key, value in offline_env.items():
        os.environ[key] = value
        print(f"   🔒 {key}={value}")

def check_cache_directories():
    """캐시 디렉토리 상태 확인"""
    print("\n" + "=" * 50)
    print("📂 캐시 디렉토리 상태 확인")
    print("=" * 50)
    
    cache_paths = [
        Path.home() / '.cache' / 'huggingface' / 'hub',
        Path.home() / '.cache' / 'huggingface' / 'transformers', 
        Path.home() / '.cache' / 'torch' / 'sentence_transformers',
        Path.home() / '.cache' / 'huggingface' / 'datasets'
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            # 파일 개수와 총 크기 계산
            files = list(cache_path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            print(f"   ✅ {cache_path}")
            print(f"      📄 파일 수: {file_count:,}개")
            print(f"      💾 크기: {size_mb:.1f} MB")
            
            # 모델 디렉토리 확인 (hub 캐시인 경우)
            if cache_path.name == 'hub':
                model_dirs = [d for d in cache_path.iterdir() if d.is_dir() and d.name.startswith('models--')]
                print(f"      🤖 모델 디렉토리: {len(model_dirs)}개")
                for model_dir in sorted(model_dirs):
                    model_name = model_dir.name.replace('models--', '').replace('--', '/')
                    print(f"         - {model_name}")
        else:
            print(f"   ❌ {cache_path} (존재하지 않음)")

def test_transformers_models():
    """Transformers 모델들 개별 테스트"""
    print("\n" + "=" * 50)
    print("🤖 Transformers 모델 로딩 테스트")
    print("=" * 50)
    
    models_to_test = [
        ("Helsinki-NLP/opus-mt-ko-en", "번역 모델"),
        ("j-hartmann/emotion-english-distilroberta-base", "감정 분석 모델"),
        ("jhgan/ko-sroberta-multitask", "한국어 문장 임베딩"),
        ("beomi/KcELECTRA-base-v2022", "한국어 ELECTRA")
    ]
    
    results = {}
    
    for model_name, description in models_to_test:
        print(f"\n🔄 테스트 중: {model_name} ({description})")
        start_time = time.time()
        
        try:
            if "opus-mt" in model_name:
                # 번역 모델 테스트
                from transformers import MarianTokenizer, MarianMTModel
                tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
                model = MarianMTModel.from_pretrained(model_name, local_files_only=True)
                
                # 간단한 번역 테스트
                inputs = tokenizer("안녕하세요", return_tensors="pt")
                outputs = model.generate(**inputs)
                translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print(f"   ✅ 로딩 성공 - 번역 결과: '{translated}'")
                results[model_name] = "성공"
                
            elif "emotion" in model_name:
                # 감정 분석 모델 테스트
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                
                # 간단한 감정 분석 테스트
                inputs = tokenizer("I am happy today", return_tensors="pt")
                outputs = model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
                
                print(f"   ✅ 로딩 성공 - 예측 결과: {predictions.item()}")
                results[model_name] = "성공"
                
            else:
                # 일반 모델 테스트
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModel.from_pretrained(model_name, local_files_only=True)
                
                # 간단한 인코딩 테스트
                inputs = tokenizer("테스트 문장", return_tensors="pt")
                outputs = model(**inputs)
                
                print(f"   ✅ 로딩 성공 - 임베딩 크기: {outputs.last_hidden_state.shape}")
                results[model_name] = "성공"
            
            elapsed = time.time() - start_time
            print(f"   ⏱️ 소요 시간: {elapsed:.2f}초")
            
        except Exception as e:
            print(f"   ❌ 로딩 실패: {str(e)}")
            logger.error(f"모델 테스트 실패: {model_name} - {e}")
            results[model_name] = f"실패: {str(e)}"
    
    return results

def test_sentence_transformers():
    """SentenceTransformers 모델 테스트"""
    print("\n" + "=" * 50)  
    print("🔤 SentenceTransformers 모델 테스트")
    print("=" * 50)
    
    sentence_models = [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "jhgan/ko-sroberta-multitask"
    ]
    
    results = {}
    
    try:
        from sentence_transformers import SentenceTransformer
        
        for model_name in sentence_models:
            print(f"\n🔄 테스트 중: {model_name}")
            start_time = time.time()
            
            try:
                # local_files_only 옵션으로 로드 시도
                model = SentenceTransformer(model_name, device='cpu')
                
                # 간단한 인코딩 테스트
                test_sentences = ["안녕하세요", "Hello world"]
                embeddings = model.encode(test_sentences)
                
                print(f"   ✅ 로딩 성공")
                print(f"      📊 임베딩 차원: {embeddings.shape[1]}")
                print(f"      📄 테스트 문장 수: {len(test_sentences)}")
                
                elapsed = time.time() - start_time
                print(f"   ⏱️ 소요 시간: {elapsed:.2f}초")
                
                results[model_name] = "성공"
                
            except Exception as e:
                print(f"   ❌ 로딩 실패: {str(e)}")
                results[model_name] = f"실패: {str(e)}"
        
    except ImportError as e:
        print(f"❌ sentence-transformers 라이브러리를 가져올 수 없음: {e}")
        return {"sentence-transformers": f"라이브러리 오류: {e}"}
    
    return results

def test_subprocess_mode():
    """Subprocess 모드에서 sentence-transformers 테스트"""
    print("\n" + "=" * 50)
    print("🔄 Subprocess 모드 SentenceTransformers 테스트")
    print("=" * 50)
    
    try:
        # sentence_transformer_singleton 사용 테스트
        from sentence_transformer_singleton import get_sentence_transformer
        
        print("🔄 sentence_transformer_singleton을 통한 모델 로드 테스트...")
        
        # 다국어 모델 테스트
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        print(f"   테스트 모델: {model_name}")
        
        start_time = time.time()
        model = get_sentence_transformer(model_name)
        
        # 간단한 인코딩 테스트
        result = model.encode(["테스트 문장", "Test sentence"])
        elapsed = time.time() - start_time
        
        print(f"   ✅ Subprocess 모드 성공")
        print(f"      📊 임베딩 형태: {result.shape if hasattr(result, 'shape') else type(result)}")
        print(f"   ⏱️ 소요 시간: {elapsed:.2f}초")
        
        return "성공"
        
    except Exception as e:
        print(f"   ❌ Subprocess 모드 실패: {str(e)}")
        logger.error(f"Subprocess 모드 테스트 실패: {e}")
        return f"실패: {str(e)}"

def main():
    """메인 검증 실행"""
    print("=" * 60)
    print("🔍 Red Heart AI 모델 캐시 상태 검증")
    print("=" * 60)
    
    # 1. 오프라인 모드 설정
    print("\n🔒 오프라인 모드 설정:")
    setup_offline_mode()
    
    # 2. 캐시 디렉토리 확인
    check_cache_directories()
    
    # 3. Transformers 모델 테스트
    transformers_results = test_transformers_models()
    
    # 4. SentenceTransformers 모델 테스트
    sentence_results = test_sentence_transformers()
    
    # 5. Subprocess 모드 테스트
    subprocess_result = test_subprocess_mode()
    
    # 최종 결과 요약
    print("\n" + "=" * 60)
    print("📊 검증 결과 요약")
    print("=" * 60)
    
    print("\n🤖 Transformers 모델:")
    for model, result in transformers_results.items():
        status = "✅" if result == "성공" else "❌"
        print(f"   {status} {model.split('/')[-1]}: {result}")
    
    print("\n🔤 SentenceTransformers 모델:")
    for model, result in sentence_results.items():
        status = "✅" if result == "성공" else "❌"
        model_short = model.split('/')[-1]
        print(f"   {status} {model_short}: {result}")
    
    print(f"\n🔄 Subprocess 모드: {'✅' if subprocess_result == '성공' else '❌'} {subprocess_result}")
    
    # 전체 성공 여부 판단
    all_transformers_ok = all(r == "성공" for r in transformers_results.values())
    all_sentence_ok = all(r == "성공" for r in sentence_results.values())
    subprocess_ok = subprocess_result == "성공"
    
    if all_transformers_ok and all_sentence_ok and subprocess_ok:
        print("\n🎉 모든 모델이 오프라인에서 정상 작동합니다!")
        print("   unified-test 실행이 가능합니다.")
        return 0
    else:
        print("\n⚠️ 일부 모델에 문제가 있습니다.")
        print("   문제 해결 후 다시 테스트하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)