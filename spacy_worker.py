#!/usr/bin/env python3
"""
spacy 워커 - conda 환경에서 spacy 작업을 처리하는 별도 프로세스
"""

import json
import numpy as np
import sys
import logging
from typing import List, Dict, Any

# conda 환경에서 실행되므로 spacy가 정상 작동함
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('spacy_Worker')

# spacy 모델 전역 변수
nlp_models = {}

def load_spacy_model(model_name='en_core_web_sm'):
    """spacy 모델 로딩"""
    global nlp_models
    
    if model_name in nlp_models:
        return nlp_models[model_name]
    
    try:
        nlp = spacy.load(model_name)
        nlp_models[model_name] = nlp
        logger.info(f"spacy 모델 로드 완료: {model_name}")
        return nlp
    except OSError:
        # 모델이 없으면 빈 모델 생성
        try:
            nlp = spacy.blank('en')
            nlp_models[model_name] = nlp
            logger.warning(f"spacy 모델 {model_name} 없음, 빈 모델 사용")
            return nlp
        except Exception as e:
            logger.error(f"spacy 모델 로딩 실패: {e}")
            raise


def process_text_nlp(data):
    """텍스트 NLP 처리"""
    texts = data['texts']
    model_name = data.get('model_name', 'en_core_web_sm')
    operations = data.get('operations', ['tokenize', 'pos', 'ner'])
    
    nlp = load_spacy_model(model_name)
    
    results = []
    for text in texts:
        doc = nlp(text)
        
        text_result = {
            'text': text,
            'length': len(doc)
        }
        
        if 'tokenize' in operations:
            text_result['tokens'] = [token.text for token in doc]
            text_result['lemmas'] = [token.lemma_ for token in doc]
        
        if 'pos' in operations:
            text_result['pos_tags'] = [(token.text, token.pos_, token.tag_) for token in doc]
        
        if 'ner' in operations:
            text_result['entities'] = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        
        if 'dependency' in operations:
            text_result['dependencies'] = [(token.text, token.dep_, token.head.text) for token in doc]
        
        if 'similarity' in operations and len(texts) > 1:
            # 다른 텍스트들과의 유사도 계산
            similarities = []
            for other_text in texts:
                if other_text != text:
                    other_doc = nlp(other_text)
                    similarity = doc.similarity(other_doc)
                    similarities.append({'text': other_text, 'similarity': similarity})
            text_result['similarities'] = similarities
        
        results.append(text_result)
    
    logger.info(f"spacy 텍스트 처리 완료: {len(texts)}개 텍스트")
    return {
        'status': 'success',
        'model_used': model_name,
        'operations': operations,
        'results': results
    }


def extract_keywords(data):
    """키워드 추출"""
    texts = data['texts']
    model_name = data.get('model_name', 'en_core_web_sm')
    max_keywords = data.get('max_keywords', 10)
    
    nlp = load_spacy_model(model_name)
    
    all_keywords = []
    for text in texts:
        doc = nlp(text)
        
        # 중요한 토큰들 추출 (명사, 형용사, 동사)
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'frequency': 1  # 단순화
                })
        
        # 중복 제거 및 빈도 계산
        keyword_dict = {}
        for kw in keywords:
            lemma = kw['lemma'].lower()
            if lemma in keyword_dict:
                keyword_dict[lemma]['frequency'] += 1
            else:
                keyword_dict[lemma] = kw
        
        # 빈도순 정렬
        sorted_keywords = sorted(keyword_dict.values(), 
                               key=lambda x: x['frequency'], 
                               reverse=True)[:max_keywords]
        
        all_keywords.append({
            'text': text,
            'keywords': sorted_keywords
        })
    
    logger.info(f"키워드 추출 완료: {len(texts)}개 텍스트")
    return {
        'status': 'success',
        'model_used': model_name,
        'results': all_keywords
    }


def analyze_sentence_structure(data):
    """문장 구조 분석"""
    texts = data['texts']
    model_name = data.get('model_name', 'en_core_web_sm')
    
    nlp = load_spacy_model(model_name)
    
    results = []
    for text in texts:
        doc = nlp(text)
        
        sentences = []
        for sent in doc.sents:
            sentence_info = {
                'text': sent.text,
                'start': sent.start_char,
                'end': sent.end_char,
                'length': len(sent),
                'root': sent.root.text,
                'subject': None,
                'object': None
            }
            
            # 주어와 목적어 찾기
            for token in sent:
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    sentence_info['subject'] = token.text
                elif token.dep_ in ['dobj', 'pobj']:
                    sentence_info['object'] = token.text
            
            sentences.append(sentence_info)
        
        results.append({
            'text': text,
            'sentence_count': len(sentences),
            'sentences': sentences
        })
    
    logger.info(f"문장 구조 분석 완료: {len(texts)}개 텍스트")
    return {
        'status': 'success',
        'model_used': model_name,
        'results': results
    }


def process_spacy_operation(input_file: str, output_file: str):
    """spacy 작업 처리 메인 함수"""
    try:
        # 입력 데이터 로딩
        with open(input_file, 'r') as f:
            request = json.load(f)
        
        operation = request['operation']
        data = request['data']
        
        # 작업 처리
        if operation == 'process_text_nlp':
            result = process_text_nlp(data)
        elif operation == 'extract_keywords':
            result = extract_keywords(data)
        elif operation == 'analyze_sentence_structure':
            result = analyze_sentence_structure(data)
        else:
            raise ValueError(f"지원되지 않는 작업: {operation}")
        
        # 결과 저장
        with open(output_file, 'w') as f:
            json.dump(result, f)
        
        logger.info(f"spacy 작업 완료: {operation}")
        
    except Exception as e:
        logger.error(f"spacy 작업 실패: {e}")
        error_result = {'status': 'error', 'error': str(e)}
        with open(output_file, 'w') as f:
            json.dump(error_result, f)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python spacy_worker.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_spacy_operation(input_file, output_file)