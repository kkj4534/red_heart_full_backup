"""
고급 데이터 로더 - Linux 전용
Advanced Data Loader for Red Heart Linux

기존 Red Heart의 모든 데이터를 로드하고 고급 AI 분석을 위한 
전처리를 수행하는 시스템
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, field

# 고급 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
import torch

from config import ADVANCED_CONFIG, DEVICE, MODELS_DIR
from data_models import EthicalScenario, DatasetSummary, ProcessingMetrics

logger = logging.getLogger('RedHeart.AdvancedDataLoader')


@dataclass
class DatasetInfo:
    """데이터셋 정보"""
    name: str
    file_path: str
    total_scenarios: int
    categories: List[str]
    ethical_themes: List[str]
    last_updated: str
    processing_status: str = "ready"
    error_message: Optional[str] = None


@dataclass
class LoadingProgress:
    """데이터 로딩 진행 상황"""
    total_files: int = 0
    processed_files: int = 0
    current_file: str = ""
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    estimated_completion: float = 0.0


class AdvancedDataLoader:
    """고급 데이터 로더"""
    
    def __init__(self):
        self.logger = logger
        self.device = DEVICE
        
        # 데이터 경로
        self.data_dir = Path("./data")
        self.processed_datasets_dir = Path("./processed_datasets")
        self.korean_literature_dir = Path("./korean_literature_data")
        
        # 로딩된 데이터
        self.datasets = {}
        self.dataset_info = {}
        self.embeddings_cache = {}
        
        # 고급 처리 도구
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        
        # 스레드 안전성
        self.loading_lock = threading.Lock()
        self.progress = LoadingProgress()
        
        # 성능 메트릭
        self.processing_metrics = ProcessingMetrics()
        
        self.logger.info("고급 데이터 로더 초기화 완료")
        
    def initialize_models(self):
        """모델 초기화"""
        try:
            if ADVANCED_CONFIG['use_sentence_transformers']:
                self.logger.info("Sentence Transformer 모델 로딩 중...")
                from sentence_transformer_singleton import get_sentence_transformer
                
                self.sentence_transformer = get_sentence_transformer(
                    'paraphrase-multilingual-mpnet-base-v2',
                    device=str(self.device),
                    cache_folder=os.path.join(MODELS_DIR, 'sentence_transformers')
                )
                self.logger.info("Sentence Transformer 모델 로딩 완료")
                
            # TF-IDF 벡터라이저
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            # 라벨 인코더
            self.label_encoder = LabelEncoder()
            
            self.logger.info("모든 모델 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}")
            raise
            
    def discover_datasets(self) -> Dict[str, DatasetInfo]:
        """사용 가능한 데이터셋 발견"""
        self.logger.info("데이터셋 검색 시작...")
        
        datasets = {}
        
        # 1. 처리된 데이터셋들
        if self.processed_datasets_dir.exists():
            for json_file in self.processed_datasets_dir.glob("*.json"):
                try:
                    info = self._analyze_dataset_file(json_file)
                    if info:
                        datasets[info.name] = info
                        self.logger.debug(f"데이터셋 발견: {info.name} ({info.total_scenarios} 시나리오)")
                except Exception as e:
                    self.logger.error(f"데이터셋 분석 실패 {json_file}: {e}")
                    
        # 2. 한국 문학 데이터
        if self.korean_literature_dir.exists():
            for json_file in self.korean_literature_dir.glob("*.json"):
                try:
                    info = self._analyze_korean_literature_file(json_file)
                    if info:
                        datasets[info.name] = info
                        self.logger.debug(f"한국 문학 데이터셋 발견: {info.name}")
                except Exception as e:
                    self.logger.error(f"한국 문학 데이터 분석 실패 {json_file}: {e}")
                    
        self.dataset_info = datasets
        self.logger.info(f"총 {len(datasets)}개 데이터셋 발견")
        
        return datasets
        
    def _analyze_dataset_file(self, file_path: Path) -> Optional[DatasetInfo]:
        """데이터셋 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 파일 구조에 따른 분석
            if isinstance(data, dict):
                if 'integrated_scenarios' in data:
                    # 통합 데이터셋
                    scenarios = data['integrated_scenarios']
                    total_scenarios = len(scenarios)
                    
                    categories = set()
                    ethical_themes = set()
                    
                    for scenario in scenarios[:100]:  # 샘플링으로 성능 향상
                        if 'category' in scenario:
                            categories.add(scenario['category'])
                        if 'ethical_themes' in scenario:
                            ethical_themes.update(scenario['ethical_themes'])
                            
                    return DatasetInfo(
                        name=file_path.stem,
                        file_path=str(file_path),
                        total_scenarios=total_scenarios,
                        categories=list(categories),
                        ethical_themes=list(ethical_themes),
                        last_updated=data.get('integration_date', 'unknown')
                    )
                    
                elif 'scenarios' in data:
                    # 일반 시나리오 데이터셋
                    scenarios = data['scenarios']
                    return DatasetInfo(
                        name=file_path.stem,
                        file_path=str(file_path),
                        total_scenarios=len(scenarios),
                        categories=[],
                        ethical_themes=[],
                        last_updated='unknown'
                    )
                    
            elif isinstance(data, list):
                # 리스트 형태 데이터셋
                return DatasetInfo(
                    name=file_path.stem,
                    file_path=str(file_path),
                    total_scenarios=len(data),
                    categories=[],
                    ethical_themes=[],
                    last_updated='unknown'
                )
                
        except Exception as e:
            self.logger.error(f"데이터셋 파일 분석 실패 {file_path}: {e}")
            return None
            
        return None
        
    def _analyze_korean_literature_file(self, file_path: Path) -> Optional[DatasetInfo]:
        """한국 문학 데이터 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                total_items = len(data)
            elif isinstance(data, list):
                total_items = len(data)
            else:
                total_items = 1
                
            return DatasetInfo(
                name=f"korean_literature_{file_path.stem}",
                file_path=str(file_path),
                total_scenarios=total_items,
                categories=['korean_literature'],
                ethical_themes=['cultural_context'],
                last_updated='unknown'
            )
            
        except Exception as e:
            self.logger.error(f"한국 문학 데이터 분석 실패 {file_path}: {e}")
            return None
            
    def load_dataset(self, dataset_name: str, 
                    include_embeddings: bool = True,
                    sample_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """특정 데이터셋 로드"""
        
        if dataset_name not in self.dataset_info:
            self.logger.error(f"데이터셋을 찾을 수 없습니다: {dataset_name}")
            return None
            
        dataset_info = self.dataset_info[dataset_name]
        
        try:
            self.logger.info(f"데이터셋 로딩 시작: {dataset_name}")
            start_time = time.time()
            
            # JSON 파일 로드
            with open(dataset_info.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            # 데이터 구조 정규화
            normalized_data = self._normalize_data_structure(raw_data, dataset_name)
            
            # 샘플링 (필요한 경우)
            if sample_size and len(normalized_data) > sample_size:
                normalized_data = np.random.choice(
                    normalized_data, sample_size, replace=False
                ).tolist()
                self.logger.info(f"데이터 샘플링: {len(normalized_data)} 항목")
                
            # 고급 전처리
            processed_data = self._advanced_preprocessing(normalized_data, dataset_name)
            
            # 임베딩 생성 (선택적)
            if include_embeddings and self.sentence_transformer:
                embeddings = self._generate_embeddings(processed_data)
                processed_data['embeddings'] = embeddings
                
            # 캐시에 저장
            self.datasets[dataset_name] = processed_data
            
            loading_time = time.time() - start_time
            self.logger.info(f"데이터셋 로딩 완료: {dataset_name} ({loading_time:.2f}초)")
            
            # 메트릭 업데이트
            self.processing_metrics.update_loading_stats(dataset_name, loading_time, len(normalized_data))
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"데이터셋 로딩 실패 {dataset_name}: {e}")
            return None
            
    def _normalize_data_structure(self, raw_data: Any, dataset_name: str) -> List[Dict[str, Any]]:
        """데이터 구조 정규화"""
        normalized = []
        
        try:
            if isinstance(raw_data, dict):
                if 'integrated_scenarios' in raw_data:
                    # 통합 시나리오 데이터
                    scenarios = raw_data['integrated_scenarios']
                    for scenario in scenarios:
                        normalized_scenario = {
                            'id': scenario.get('id', ''),
                            'title': scenario.get('title', ''),
                            'description': scenario.get('description', ''),
                            'category': scenario.get('category', 'unknown'),
                            'ethical_themes': scenario.get('ethical_themes', []),
                            'stakeholders': scenario.get('stakeholders', []),
                            'options': scenario.get('options', []),
                            'source_type': scenario.get('source_type', 'unknown'),
                            'source_file': scenario.get('source_file', ''),
                            'metadata': {
                                'dataset': dataset_name,
                                'original_structure': 'integrated_scenarios'
                            }
                        }
                        normalized.append(normalized_scenario)
                        
                elif 'scenarios' in raw_data:
                    # 일반 시나리오 데이터
                    scenarios = raw_data['scenarios']
                    for scenario in scenarios:
                        normalized_scenario = {
                            'id': scenario.get('id', ''),
                            'title': scenario.get('title', ''),
                            'description': scenario.get('description', scenario.get('text', '')),
                            'category': scenario.get('category', 'unknown'),
                            'ethical_themes': scenario.get('themes', scenario.get('ethical_themes', [])),
                            'stakeholders': scenario.get('stakeholders', []),
                            'options': scenario.get('options', []),
                            'source_type': 'scenario_collection',
                            'metadata': {
                                'dataset': dataset_name,
                                'original_structure': 'scenarios'
                            }
                        }
                        normalized.append(normalized_scenario)
                        
                else:
                    # 한국 문학이나 기타 구조
                    for key, value in raw_data.items():
                        if isinstance(value, dict):
                            normalized_item = {
                                'id': key,
                                'title': value.get('title', key),
                                'description': value.get('content', value.get('text', str(value))),
                                'category': 'korean_literature',
                                'ethical_themes': value.get('themes', []),
                                'stakeholders': [],
                                'options': [],
                                'source_type': 'korean_literature',
                                'metadata': {
                                    'dataset': dataset_name,
                                    'original_structure': 'key_value'
                                }
                            }
                            normalized.append(normalized_item)
                            
            elif isinstance(raw_data, list):
                # 리스트 형태 데이터
                for i, item in enumerate(raw_data):
                    if isinstance(item, dict):
                        normalized_item = {
                            'id': item.get('id', f"{dataset_name}_{i}"),
                            'title': item.get('title', f"Item {i+1}"),
                            'description': item.get('description', item.get('text', str(item))),
                            'category': item.get('category', 'unknown'),
                            'ethical_themes': item.get('ethical_themes', []),
                            'stakeholders': item.get('stakeholders', []),
                            'options': item.get('options', []),
                            'source_type': 'list_item',
                            'metadata': {
                                'dataset': dataset_name,
                                'original_structure': 'list',
                                'index': i
                            }
                        }
                        normalized.append(normalized_item)
                        
        except Exception as e:
            self.logger.error(f"데이터 구조 정규화 실패: {e}")
            
        return normalized
        
    def _advanced_preprocessing(self, data: List[Dict[str, Any]], 
                              dataset_name: str) -> Dict[str, Any]:
        """고급 전처리"""
        try:
            # 텍스트 정리 및 통계 생성
            texts = []
            categories = []
            ethical_themes_flat = []
            
            for item in data:
                # 텍스트 결합 및 정리
                combined_text = f"{item.get('title', '')} {item.get('description', '')}"
                cleaned_text = self._clean_text(combined_text)
                texts.append(cleaned_text)
                
                # 카테고리 수집
                categories.append(item.get('category', 'unknown'))
                
                # 윤리적 테마 수집
                themes = item.get('ethical_themes', [])
                if isinstance(themes, list):
                    ethical_themes_flat.extend(themes)
                    
            # 통계 생성
            category_counts = pd.Series(categories).value_counts().to_dict()
            theme_counts = pd.Series(ethical_themes_flat).value_counts().to_dict()
            
            # TF-IDF 벡터화 (샘플링으로 성능 향상)
            sample_texts = texts[:min(1000, len(texts))]
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(sample_texts)
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # 상위 키워드 추출
                mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = np.argsort(mean_scores)[::-1][:50]
                top_keywords = [feature_names[i] for i in top_indices]
            except Exception as e:
                self.logger.warning(f"TF-IDF 처리 실패: {e}")
                top_keywords = []
                
            return {
                'scenarios': data,
                'texts': texts,
                'statistics': {
                    'total_scenarios': len(data),
                    'category_distribution': category_counts,
                    'theme_distribution': theme_counts,
                    'top_keywords': top_keywords,
                    'avg_text_length': np.mean([len(text) for text in texts]),
                    'total_unique_categories': len(category_counts),
                    'total_unique_themes': len(theme_counts)
                },
                'metadata': {
                    'dataset_name': dataset_name,
                    'preprocessing_version': '2.0',
                    'processed_at': time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"고급 전처리 실패: {e}")
            return {
                'scenarios': data,
                'texts': [item.get('description', '') for item in data],
                'statistics': {'total_scenarios': len(data)},
                'metadata': {'dataset_name': dataset_name, 'error': str(e)}
            }
            
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
            
        # 기본 정리
        cleaned = text.strip()
        
        # 불필요한 문자 제거
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)  # 공백 정리
        cleaned = re.sub(r'[^\w\s가-힣.,!?]', '', cleaned)  # 특수문자 제거 (한글 유지)
        
        return cleaned
        
    def _generate_embeddings(self, processed_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """임베딩 생성"""
        try:
            texts = processed_data['texts']
            self.logger.info(f"임베딩 생성 시작: {len(texts)} 텍스트")
            
            # 배치 처리로 임베딩 생성
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                try:
                    batch_embeddings = self.sentence_transformer.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        show_progress_bar=False
                    )
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    self.logger.error(f"배치 임베딩 실패: {e}")
                    # 실패한 배치는 0으로 채움
                    embeddings.extend([np.zeros(384) for _ in batch_texts])
                    
            embeddings_array = np.array(embeddings)
            self.logger.info(f"임베딩 생성 완료: {embeddings_array.shape}")
            
            return {
                'text_embeddings': embeddings_array,
                'embedding_model': 'paraphrase-multilingual-mpnet-base-v2',
                'embedding_dim': embeddings_array.shape[1] if len(embeddings_array) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            return {}
            
    def load_all_datasets(self, include_embeddings: bool = False,
                         max_scenarios_per_dataset: int = 1000) -> Dict[str, Any]:
        """모든 데이터셋 로드"""
        
        self.logger.info("모든 데이터셋 로딩 시작...")
        start_time = time.time()
        
        # 데이터셋 발견
        datasets_info = self.discover_datasets()
        
        self.progress.total_files = len(datasets_info)
        self.progress.start_time = start_time
        
        loaded_datasets = {}
        total_scenarios = 0
        
        for dataset_name, dataset_info in datasets_info.items():
            try:
                self.progress.current_file = dataset_name
                
                dataset = self.load_dataset(
                    dataset_name,
                    include_embeddings=include_embeddings,
                    sample_size=max_scenarios_per_dataset
                )
                
                if dataset:
                    loaded_datasets[dataset_name] = dataset
                    scenario_count = dataset['statistics']['total_scenarios']
                    total_scenarios += scenario_count
                    
                    self.logger.info(f"데이터셋 로드 완료: {dataset_name} ({scenario_count} 시나리오)")
                    
                self.progress.processed_files += 1
                
            except Exception as e:
                self.logger.error(f"데이터셋 로딩 실패 {dataset_name}: {e}")
                self.progress.errors.append(f"{dataset_name}: {str(e)}")
                
        total_time = time.time() - start_time
        
        # 전체 통계 생성
        overall_stats = self._generate_overall_statistics(loaded_datasets)
        
        result = {
            'datasets': loaded_datasets,
            'overall_statistics': overall_stats,
            'loading_summary': {
                'total_datasets': len(datasets_info),
                'loaded_datasets': len(loaded_datasets),
                'failed_datasets': len(datasets_info) - len(loaded_datasets),
                'total_scenarios': total_scenarios,
                'loading_time': total_time,
                'errors': self.progress.errors
            },
            'metadata': {
                'loaded_at': time.time(),
                'loader_version': '2.0',
                'include_embeddings': include_embeddings
            }
        }
        
        self.logger.info(f"전체 데이터셋 로딩 완료: {len(loaded_datasets)}개 데이터셋, "
                        f"{total_scenarios}개 시나리오 ({total_time:.2f}초)")
        
        return result
        
    def _generate_overall_statistics(self, loaded_datasets: Dict[str, Any]) -> Dict[str, Any]:
        """전체 통계 생성"""
        
        all_categories = []
        all_themes = []
        all_text_lengths = []
        dataset_sizes = {}
        
        for dataset_name, dataset in loaded_datasets.items():
            stats = dataset.get('statistics', {})
            
            # 카테고리 수집
            if 'category_distribution' in stats:
                all_categories.extend(stats['category_distribution'].keys())
                
            # 테마 수집
            if 'theme_distribution' in stats:
                all_themes.extend(stats['theme_distribution'].keys())
                
            # 텍스트 길이
            if 'avg_text_length' in stats:
                all_text_lengths.append(stats['avg_text_length'])
                
            # 데이터셋 크기
            dataset_sizes[dataset_name] = stats.get('total_scenarios', 0)
            
        # 통계 계산
        unique_categories = list(set(all_categories))
        unique_themes = list(set(all_themes))
        
        return {
            'total_unique_categories': len(unique_categories),
            'total_unique_themes': len(unique_themes),
            'categories': unique_categories,
            'themes': unique_themes,
            'dataset_sizes': dataset_sizes,
            'avg_text_length_overall': np.mean(all_text_lengths) if all_text_lengths else 0,
            'largest_dataset': max(dataset_sizes.items(), key=lambda x: x[1]) if dataset_sizes else None,
            'smallest_dataset': min(dataset_sizes.items(), key=lambda x: x[1]) if dataset_sizes else None
        }
        
    def get_dataset_summary(self) -> DatasetSummary:
        """데이터셋 요약 정보 반환"""
        
        summary = DatasetSummary(
            total_datasets=len(self.dataset_info),
            loaded_datasets=len(self.datasets),
            total_scenarios=sum(info.total_scenarios for info in self.dataset_info.values()),
            available_categories=list(set().union(*[info.categories for info in self.dataset_info.values()])),
            available_themes=list(set().union(*[info.ethical_themes for info in self.dataset_info.values()])),
            processing_metrics=self.processing_metrics
        )
        
        return summary
        
    def clear_cache(self):
        """캐시 클리어"""
        self.datasets.clear()
        self.embeddings_cache.clear()
        self.logger.info("데이터 캐시가 클리어되었습니다.")


def test_advanced_data_loader():
    """고급 데이터 로더 테스트"""
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 데이터 로더 초기화
        loader = AdvancedDataLoader()
        loader.initialize_models()
        
        print("=== 고급 데이터 로더 테스트 ===")
        
        # 1. 데이터셋 발견
        print("\n📂 데이터셋 발견 중...")
        datasets_info = loader.discover_datasets()
        
        print(f"발견된 데이터셋: {len(datasets_info)}개")
        for name, info in datasets_info.items():
            print(f"  • {name}: {info.total_scenarios} 시나리오")
            print(f"    카테고리: {len(info.categories)}개")
            print(f"    윤리 테마: {len(info.ethical_themes)}개")
            
        # 2. 특정 데이터셋 로드 테스트
        if datasets_info:
            first_dataset_name = list(datasets_info.keys())[0]
            print(f"\n📊 데이터셋 로드 테스트: {first_dataset_name}")
            
            dataset = loader.load_dataset(
                first_dataset_name, 
                include_embeddings=True,
                sample_size=100
            )
            
            if dataset:
                stats = dataset['statistics']
                print(f"  로드된 시나리오: {stats['total_scenarios']}개")
                print(f"  평균 텍스트 길이: {stats['avg_text_length']:.1f}자")
                print(f"  카테고리: {stats['total_unique_categories']}개")
                print(f"  테마: {stats['total_unique_themes']}개")
                
                if 'embeddings' in dataset:
                    embeddings = dataset['embeddings']
                    if 'text_embeddings' in embeddings:
                        embed_shape = embeddings['text_embeddings'].shape
                        print(f"  임베딩: {embed_shape}")
                        
        # 3. 전체 데이터셋 로딩 (제한적)
        print(f"\n🔄 전체 데이터셋 로딩 (샘플링)...")
        all_data = loader.load_all_datasets(
            include_embeddings=False,
            max_scenarios_per_dataset=50
        )
        
        summary = all_data['loading_summary']
        print(f"  로드된 데이터셋: {summary['loaded_datasets']}/{summary['total_datasets']}")
        print(f"  총 시나리오: {summary['total_scenarios']}개")
        print(f"  로딩 시간: {summary['loading_time']:.2f}초")
        
        if summary['errors']:
            print(f"  오류: {len(summary['errors'])}개")
            
        # 4. 통계 정보
        overall_stats = all_data['overall_statistics']
        print(f"\n📈 전체 통계:")
        print(f"  고유 카테고리: {overall_stats['total_unique_categories']}개")
        print(f"  고유 테마: {overall_stats['total_unique_themes']}개")
        
        if overall_stats['largest_dataset']:
            largest = overall_stats['largest_dataset']
            print(f"  최대 데이터셋: {largest[0]} ({largest[1]} 시나리오)")
            
        print("\n✅ 테스트 완료!")
        
        return loader
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_advanced_data_loader()