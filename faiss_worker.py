#!/usr/bin/env python3
"""
FAISS 워커 - conda 환경에서 FAISS 작업을 처리하는 별도 프로세스
"""

import json
import numpy as np
import sys
import os
import logging

# conda 환경에서 실행되므로 FAISS가 정상 작동함
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FAISS_Worker')


def create_faiss_index(data):
    """FAISS 인덱스 생성"""
    dimension = data.get('dimension', 128)
    index_type = data.get('index_type', 'IndexFlatL2')
    
    if index_type == 'IndexFlatL2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IndexFlatIP':
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"지원되지 않는 인덱스 타입: {index_type}")
    
    logger.info(f"FAISS 인덱스 생성: {index_type}, 차원={dimension}")
    return {'status': 'success', 'index_type': index_type, 'dimension': dimension}


def add_vectors_to_index(data):
    """벡터를 FAISS 인덱스에 추가"""
    vectors = np.array(data['vectors'], dtype='float32')
    dimension = data.get('dimension', vectors.shape[1])
    index_type = data.get('index_type', 'IndexFlatL2')
    
    # 인덱스 생성
    if index_type == 'IndexFlatL2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IndexFlatIP':
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"지원되지 않는 인덱스 타입: {index_type}")
    
    # 벡터 추가
    index.add(vectors)
    
    logger.info(f"FAISS에 {len(vectors)}개 벡터 추가 완료")
    return {'status': 'success', 'total_vectors': index.ntotal}


def search_vectors(data):
    """FAISS에서 벡터 검색"""
    vectors = np.array(data['vectors'], dtype='float32')
    query_vectors = np.array(data['query_vectors'], dtype='float32')
    dimension = data.get('dimension', vectors.shape[1])
    index_type = data.get('index_type', 'IndexFlatL2')
    k = data.get('k', 5)
    
    # 인덱스 생성 및 벡터 추가
    if index_type == 'IndexFlatL2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IndexFlatIP':
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"지원되지 않는 인덱스 타입: {index_type}")
    
    index.add(vectors)
    
    # 검색 수행
    distances, indices = index.search(query_vectors, k)
    
    logger.info(f"FAISS 검색 완료: {len(query_vectors)}개 쿼리, {k}개 결과")
    return {
        'status': 'success',
        'distances': distances.tolist(),
        'indices': indices.tolist(),
        'total_vectors': index.ntotal
    }


def process_embeddings(data):
    """임베딩 처리 (SentenceTransformers + FAISS)"""
    texts = data['texts']
    model_name = data.get('model_name', 'all-MiniLM-L6-v2')
    search_text = data.get('search_text', None)
    k = data.get('k', 5)
    
    # SentenceTransformers 사용
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    
    # 임베딩 생성
    embeddings = model.encode(texts)
    embeddings_flat = embeddings.astype('float32')
    
    # FAISS 인덱스 생성 및 추가
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_flat)
    
    result = {
        'status': 'success',
        'dimension': dimension,
        'total_vectors': index.ntotal,
        'embeddings_shape': embeddings.shape
    }
    
    # 검색이 요청된 경우
    if search_text:
        search_embedding = model.encode([search_text]).astype('float32')
        distances, indices = index.search(search_embedding, k)
        
        result.update({
            'search_distances': distances.tolist(),
            'search_indices': indices.tolist(),
            'search_results': [texts[i] for i in indices[0] if i < len(texts)]
        })
    
    logger.info(f"임베딩 처리 완료: {len(texts)}개 텍스트, 차원={dimension}")
    return result


def test_faiss_environment(data):
    """FAISS 환경 테스트 - 환경 분리 검증용"""
    try:
        # FAISS 기본 기능 테스트
        dimension = data.get('dimension', 64)
        test_vectors = np.random.random((10, dimension)).astype('float32')
        
        # 인덱스 생성 및 테스트
        index = faiss.IndexFlatL2(dimension)
        index.add(test_vectors)
        
        # 검색 테스트
        query_vector = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query_vector, 3)
        
        # 환경 정보 수집
        import sys
        import platform
        
        result = {
            'status': 'success',
            'test_type': 'environment_verification',
            'faiss_version': faiss.__version__ if hasattr(faiss, '__version__') else 'unknown',
            'python_version': sys.version,
            'platform': platform.platform(),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
            'virtual_env': os.environ.get('VIRTUAL_ENV', 'none'),
            'test_results': {
                'index_creation': True,
                'vector_addition': True,
                'vector_search': True,
                'total_vectors': index.ntotal,
                'dimension': dimension,
                'search_results_count': len(indices[0])
            }
        }
        
        logger.info("FAISS 환경 테스트 성공")
        return result
        
    except Exception as e:
        logger.error(f"FAISS 환경 테스트 실패: {e}")
        return {
            'status': 'error',
            'test_type': 'environment_verification',
            'error': str(e),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
            'virtual_env': os.environ.get('VIRTUAL_ENV', 'none')
        }


def process_faiss_operation(input_file: str, output_file: str):
    """FAISS 작업 처리 메인 함수"""
    try:
        # 입력 데이터 로딩
        with open(input_file, 'r') as f:
            request = json.load(f)
        
        operation = request['operation']
        data = request['data']
        
        # 작업 처리
        if operation == 'create_index':
            result = create_faiss_index(data)
        elif operation == 'add_vectors':
            result = add_vectors_to_index(data)
        elif operation == 'search_vectors':
            result = search_vectors(data)
        elif operation == 'process_embeddings':
            result = process_embeddings(data)
        elif operation == 'test':
            result = test_faiss_environment(data)
        else:
            raise ValueError(f"지원되지 않는 작업: {operation}")
        
        # 결과 저장
        with open(output_file, 'w') as f:
            json.dump(result, f)
        
        logger.info(f"FAISS 작업 완료: {operation}")
        
    except Exception as e:
        logger.error(f"FAISS 작업 실패: {e}")
        error_result = {'status': 'error', 'error': str(e)}
        with open(output_file, 'w') as f:
            json.dump(error_result, f)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python faiss_worker.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_faiss_operation(input_file, output_file)