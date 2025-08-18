"""
유틸리티 함수 - 시스템 전반에서 사용되는 유틸리티 함수들을 정의합니다.
"""

import os
import json
import time
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import psutil
from tqdm import tqdm
import csv
import pickle
from pathlib import Path

from config import BASE_DIR, DATA_DIR, PLOTS_DIR, LOGS_DIR, SYSTEM_CONFIG
from data_models import EthicalSituation # 상단에 이미 typing에서 List 등을 import 했다고 가정

logger = logging.getLogger('RedHeart.Utils')

# 데이터 입출력 함수
def save_json(data: Dict, filepath: str, ensure_dir: bool = True) -> bool:
    """
    데이터를 JSON 파일로 저장합니다.

    Args:
        data: 저장할 데이터 딕셔너리
        filepath: 저장 파일 경로
        ensure_dir: 디렉토리가 없으면 생성할지 여부

    Returns:
        성공 여부
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"데이터가 {filepath}에 저장되었습니다.")
        return True
    except Exception as e:
        logger.error(f"JSON 저장 중 오류 발생: {e}")
        return False

def load_json(filepath: str, default: Dict = None) -> Optional[Dict]:
    """
    JSON 파일에서 데이터를 로드합니다.

    Args:
        filepath: 로드할 파일 경로
        default: 파일이 없거나 로드 실패 시 반환할 기본값

    Returns:
        로드된 데이터 또는 기본값
    """
    if not os.path.exists(filepath):
        logger.warning(f"파일이 존재하지 않습니다: {filepath}")
        return default
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"{filepath}에서 데이터를 로드했습니다.")
        return data
    except Exception as e:
        logger.error(f"JSON 로드 중 오류 발생: {e}")
        return default

def save_pickle(data: Any, filepath: str, ensure_dir: bool = True) -> bool:
    """
    데이터를 Pickle 파일로 저장합니다.

    Args:
        data: 저장할 데이터
        filepath: 저장 파일 경로
        ensure_dir: 디렉토리가 없으면 생성할지 여부

    Returns:
        성공 여부
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"데이터가 {filepath}에 저장되었습니다.")
        return True
    except Exception as e:
        logger.error(f"Pickle 저장 중 오류 발생: {e}")
        return False

def load_pickle(filepath: str, default: Any = None) -> Any:
    """
    Pickle 파일에서 데이터를 로드합니다.

    Args:
        filepath: 로드할 파일 경로
        default: 파일이 없거나 로드 실패 시 반환할 기본값

    Returns:
        로드된 데이터 또는 기본값
    """
    if not os.path.exists(filepath):
        logger.warning(f"파일이 존재하지 않습니다: {filepath}")
        return default
        
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.debug(f"{filepath}에서 데이터를 로드했습니다.")
        return data
    except Exception as e:
        logger.error(f"Pickle 로드 중 오류 발생: {e}")
        return default

def save_csv(data: List[Dict], filepath: str, ensure_dir: bool = True) -> bool:
    """
    데이터를 CSV 파일로 저장합니다.

    Args:
        data: 저장할 데이터 리스트 (딕셔너리의 리스트)
        filepath: 저장 파일 경로
        ensure_dir: 디렉토리가 없으면 생성할지 여부

    Returns:
        성공 여부
    """
    try:
        if not data:
            logger.warning("저장할 데이터가 없습니다.")
            return False
            
        if ensure_dir:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        fieldnames = data[0].keys()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.debug(f"데이터가 {filepath}에 저장되었습니다.")
        return True
    except Exception as e:
        logger.error(f"CSV 저장 중 오류 발생: {e}")
        return False

# 성능 관련 유틸리티
def check_resource_usage() -> Dict[str, float]:
    """
    시스템 리소스 사용량을 확인합니다.

    Returns:
        리소스 사용량 정보 (메모리, CPU 등)
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        usage = {
            'memory_mb': memory_info.rss / (1024 * 1024),  # MB 단위
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(interval=0.1),
            'threads': len(process.threads()),
            'open_files': len(process.open_files()),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return usage
    except Exception as e:
        logger.error(f"리소스 사용량 확인 중 오류 발생: {e}")
        return {
            'memory_mb': 0.0,
            'memory_percent': 0.0,
            'cpu_percent': 0.0,
            'threads': 0,
            'open_files': 0,
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(e)
        }

def wait_for_resources() -> bool:
    """
    리소스 사용량이 제한 이하로 내려갈 때까지 대기합니다.
    
    Returns:
        성공 여부
    """
    max_memory_usage = SYSTEM_CONFIG['performance']['max_memory_usage']
    processing_delay = SYSTEM_CONFIG['performance']['processing_delay']
    
    try:
        while True:
            usage = check_resource_usage()
            
            if usage['memory_percent'] < max_memory_usage * 100:  # max_memory_usage는 비율 (0.0-1.0)
                return True
                
            logger.info(f"메모리 사용량({usage['memory_percent']:.1f}%)이 제한({max_memory_usage*100:.1f}%)을 초과했습니다. 대기 중...")
            time.sleep(processing_delay)
    except Exception as e:
        logger.error(f"리소스 대기 중 오류 발생: {e}")
        return False

def measure_execution_time(func):
    """
    함수 실행 시간을 측정하는 데코레이터입니다.
    
    Args:
        func: 측정할 함수
        
    Returns:
        함수 실행 결과와 시간을 로깅하는 래퍼 함수
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"함수 {func.__name__} 실행 시간: {execution_time:.4f}초")
        
        return result, execution_time
    
    return wrapper

# 시각화 유틸리티
def plot_performance_metrics(metrics: List[Dict], save_path: str = None):
    """
    성능 메트릭을 시각화합니다.
    
    Args:
        metrics: 성능 메트릭 리스트
        save_path: 저장 경로 (None이면 저장하지 않음)
        
    Returns:
        matplotlib Figure 객체
    """
    plt.style.use('ggplot')
    
    # 데이터 준비
    epochs = [m.get('epoch', i) for i, m in enumerate(metrics)]
    accuracies = [m.get('accuracy', 0.0) for m in metrics]
    prediction_errors = [m.get('prediction_error', 0.0) for m in metrics]
    regret_ratios = [m.get('regret_ratio', 0.0) for m in metrics]
    learning_rates = [m.get('learning_rate', 0.01) for m in metrics]
    
    # 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('시스템 성능 지표', fontsize=16)
    
    # 정확도
    axes[0, 0].plot(epochs, accuracies, 'b-', marker='o')
    axes[0, 0].set_title('결정 정확도')
    axes[0, 0].set_xlabel('에포크')
    axes[0, 0].set_ylabel('정확도')
    axes[0, 0].grid(True)
    
    # 예측 오차
    axes[0, 1].plot(epochs, prediction_errors, 'r-', marker='x')
    axes[0, 1].set_title('예측 오차')
    axes[0, 1].set_xlabel('에포크')
    axes[0, 1].set_ylabel('오차')
    axes[0, 1].grid(True)
    
    # 후회 비율
    axes[1, 0].plot(epochs, regret_ratios, 'g-', marker='s')
    axes[1, 0].set_title('후회 비율')
    axes[1, 0].set_xlabel('에포크')
    axes[1, 0].set_ylabel('비율')
    axes[1, 0].grid(True)
    
    # 학습률
    axes[1, 1].plot(epochs, learning_rates, 'c-', marker='^')
    axes[1, 1].set_title('학습률')
    axes[1, 1].set_xlabel('에포크')
    axes[1, 1].set_ylabel('학습률')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"성능 지표 시각화가 {save_path}에 저장되었습니다.")
    
    return fig

def plot_surd_components(surd_result: Dict, save_path: str = None):
    """
    SURD 분석 결과를 시각화합니다.
    
    Args:
        surd_result: SURD 분석 결과 딕셔너리
        save_path: 저장 경로 (None이면 저장하지 않음)
        
    Returns:
        matplotlib Figure 객체
    """
    plt.style.use('ggplot')
    
    # 데이터 준비
    if 'surd_components' in surd_result:
        surd = surd_result['surd_components']
    else:
        surd = surd_result
    
    # 각 컴포넌트 추출
    synergistic = surd.get('S', {})
    unique = surd.get('U', {})
    redundant = surd.get('R', {})
    deterministic = surd.get('D', 0.0)
    
    # 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('SURD 인과 분석 결과', fontsize=16)
    
    # 고유(U) 효과
    if unique:
        var_names = list(unique.keys())
        values = list(unique.values())
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        axes[0, 0].bar(var_names, values, color=colors)
        axes[0, 0].set_title('고유(Unique) 효과')
        axes[0, 0].set_ylabel('효과 크기')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].grid(True)
    
    # 중복(R) 효과
    if redundant:
        pair_names = list(redundant.keys())
        values = list(redundant.values())
        colors = ['blue' if v >= 0 else 'purple' for v in values]
        
        axes[0, 1].bar(pair_names, values, color=colors)
        axes[0, 1].set_title('중복(Redundant) 효과')
        axes[0, 1].set_ylabel('효과 크기')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].grid(True)
    
    # 시너지(S) 효과
    if synergistic:
        group_names = list(synergistic.keys())
        values = list(synergistic.values())
        colors = ['orange' if v >= 0 else 'brown' for v in values]
        
        axes[1, 0].bar(group_names, values, color=colors)
        axes[1, 0].set_title('시너지(Synergistic) 효과')
        axes[1, 0].set_ylabel('효과 크기')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].grid(True)
    
    # 결정적(D) 효과와 요약
    axes[1, 1].bar(['결정적(D) 효과'], [deterministic], color='gray')
    axes[1, 1].set_title('결정적(Deterministic) 효과')
    axes[1, 1].set_ylabel('효과 크기')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"SURD 분석 시각화가 {save_path}에 저장되었습니다.")
    
    return fig

def plot_hedonic_values(hedonic_values: Dict, save_path: str = None) -> plt.Figure:
    """
    벤담의 쾌락 계산 값을 시각화합니다.
    
    Args:
        hedonic_values: 쾌락 계산 값 딕셔너리
        save_path: 저장 경로 (None이면 저장하지 않음)
        
    Returns:
        matplotlib Figure 객체
    """
    plt.style.use('ggplot')
    
    # 데이터 준비
    variables = ['intensity', 'duration', 'certainty', 'propinquity', 'fecundity', 'purity', 'extent']
    values = [hedonic_values.get(var, 0.0) for var in variables]
    
    # 변수 이름 한글화
    variable_names_kr = {
        'intensity': '강도',
        'duration': '지속성',
        'certainty': '확실성',
        'propinquity': '근접성',
        'fecundity': '다산성',
        'purity': '순수성',
        'extent': '범위'
    }
    
    kr_variables = [variable_names_kr.get(var, var) for var in variables]
    
    # 플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('벤담의 쾌락 계산법 결과', fontsize=16)
    
    # 막대 그래프
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(variables)))
    ax1.bar(kr_variables, values, color=colors)
    ax1.set_title('변수별 값')
    ax1.set_ylabel('값 (0-1 범위)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    
    # 레이더 차트
    values_normalized = values.copy()
    values_normalized.append(values_normalized[0])
    
    angles = np.linspace(0, 2*np.pi, len(variables), endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 형태로 만들기
    
    kr_variables.append(kr_variables[0])  # 닫힌 형태로 만들기
    
    ax2.set_ylim(0, 1)
    ax2.plot(angles, values_normalized, 'o-', linewidth=2)
    ax2.fill(angles, values_normalized, alpha=0.25)
    ax2.set_thetagrids(np.degrees(angles[:-1]), kr_variables[:-1])
    ax2.set_title('쾌락 계산법 레이더 차트')
    ax2.grid(True)
    
    # 종합 쾌락 값
    hedonic_total = hedonic_values.get('hedonic_total', 0.0)
    plt.figtext(0.5, 0.01, f'종합 쾌락 값: {hedonic_total:.3f}', ha='center', fontsize=14,
               bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 저장
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"쾌락 계산법 시각화가 {save_path}에 저장되었습니다.")
    
    return fig

# 파일 관리 유틸리티
def list_files(directory: str, pattern: str = None) -> List[str]:
    """
    디렉토리 내의 파일 목록을 반환합니다.
    
    Args:
        directory: 검색할 디렉토리 경로
        pattern: 파일 패턴 (예: *.json)
        
    Returns:
        파일 경로 리스트
    """
    if not os.path.exists(directory):
        logger.warning(f"디렉토리가 존재하지 않습니다: {directory}")
        return []
    
    if pattern:
        return sorted(str(p) for p in Path(directory).glob(pattern))
    else:
        return sorted(str(p) for p in Path(directory).iterdir() if p.is_file())

def get_file_list_by_date(directory: str, pattern: str = None, n_days: int = None) -> List[str]:
    """
    디렉토리 내의 파일 목록을 날짜순으로 정렬하여 반환합니다.
    
    Args:
        directory: 검색할 디렉토리 경로
        pattern: 파일 패턴 (예: *.json)
        n_days: 최근 n일 이내의 파일만 반환 (None이면 모든 파일)
        
    Returns:
        파일 경로 리스트 (최신순)
    """
    files = list_files(directory, pattern)
    
    if not files:
        return []
    
    # 최종 수정 시간 기준으로 정렬
    files_with_dates = [(f, os.path.getmtime(f)) for f in files]
    
    # n_days가 지정된 경우 필터링
    if n_days is not None:
        cutoff_time = time.time() - (n_days * 24 * 60 * 60)
        files_with_dates = [(f, t) for f, t in files_with_dates if t >= cutoff_time]
    
    # 최신순 정렬
    files_with_dates.sort(key=lambda x: x[1], reverse=True)
    
    return [f for f, _ in files_with_dates]

def clean_old_files(directory: str, pattern: str = None, max_age_days: int = 30, max_files: int = None) -> int:
    """
    오래된 파일을 정리합니다.
    
    Args:
        directory: 정리할 디렉토리 경로
        pattern: 파일 패턴 (예: *.json)
        max_age_days: 최대 보관 기간(일)
        max_files: 최대 보관 파일 수
        
    Returns:
        삭제된 파일 수
    """
    if not os.path.exists(directory):
        logger.warning(f"디렉토리가 존재하지 않습니다: {directory}")
        return 0
    
    files = list_files(directory, pattern)
    
    if not files:
        return 0
    
    # 최종 수정 시간 기준으로 정렬
    files_with_dates = [(f, os.path.getmtime(f)) for f in files]
    files_with_dates.sort(key=lambda x: x[1], reverse=True)
    
    deleted_count = 0
    current_time = time.time()
    
    # 최대 파일 수 제한
    if max_files is not None and len(files_with_dates) > max_files:
        files_to_delete = files_with_dates[max_files:]
        
        for file_path, _ in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
                logger.debug(f"파일 삭제: {file_path}")
            except Exception as e:
                logger.error(f"파일 삭제 중 오류 발생: {e}")
    
    # 최대 보관 기간 제한
    if max_age_days is not None:
        cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
        
        for file_path, mtime in files_with_dates:
            if mtime < cutoff_time:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"파일 삭제: {file_path}")
                except Exception as e:
                    logger.error(f"파일 삭제 중 오류 발생: {e}")
    
    return deleted_count

# 진행 상황 추적 유틸리티
class ProgressTracker:
    """
    진행 상황을 추적하고 로깅하는 클래스입니다.
    """
    
    def __init__(self, total: int, desc: str = "진행 중", log_interval: int = 10):
        self.total = total
        self.desc = desc
        self.progress = 0
        self.start_time = time.time()
        self.log_interval = log_interval
        self.last_log_time = self.start_time
        self.logger = logging.getLogger('RedHeart.Progress')
        
        self.tqdm = tqdm(total=total, desc=desc)
    
    def update(self, n: int = 1):
        """진행 상황을 업데이트합니다."""
        self.progress += n
        self.tqdm.update(n)
        
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        
        if elapsed >= self.log_interval:
            percentage = (self.progress / self.total) * 100 if self.total > 0 else 0
            elapsed_total = current_time - self.start_time
            
            self.logger.info(f"{self.desc}: {self.progress}/{self.total} ({percentage:.1f}%) - "
                           f"경과 시간: {elapsed_total:.1f}초")
            
            self.last_log_time = current_time
    
    def close(self):
        """추적을 종료합니다."""
        self.tqdm.close()
        
        elapsed = time.time() - self.start_time
        percentage = (self.progress / self.total) * 100 if self.total > 0 else 0
        
        self.logger.info(f"{self.desc} 완료: {self.progress}/{self.total} ({percentage:.1f}%) - "
                       f"총 소요 시간: {elapsed:.1f}초")

def load_situations_from_csv(filepath: str, encoding: str = 'utf-8') -> List[EthicalSituation]:
    """
    CSV 파일에서 윤리적 상황 데이터를 로드합니다.
    CSV 파일은 'title', 'description', 'context', 'variables', 'options' 등의 헤더를 가져야 합니다.
    'context', 'variables', 'options' 컬럼은 JSON 형식의 문자열이어야 합니다.

    Args:
        filepath: 로드할 CSV 파일 경로
        encoding: 파일 인코딩

    Returns:
        EthicalSituation 객체 리스트
    """
    situations = []
    if not os.path.exists(filepath):
        logger.error(f"CSV 파일을 찾을 수 없습니다: {filepath}")
        return situations

    try:
        with open(filepath, 'r', newline='', encoding=encoding) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # JSON 형식의 문자열을 파싱
                    context = json.loads(row.get('context', '{}')) if row.get('context') else {}
                    variables = json.loads(row.get('variables', '{}')) if row.get('variables') else {}
                    options = json.loads(row.get('options', '[]')) if row.get('options') else []

                    # EthicalSituation 객체 생성 시도
                    situation_data = {
                        'title': row.get('title', ''),
                        'description': row.get('description', ''),
                        'context': context,
                        'variables': variables,
                        'options': options,
                        'source': row.get('source', 'csv')
                    }
                    # 누락된 필수 키가 있는지 확인하고 기본값 설정 (예시)
                    # 실제 데이터 형식에 맞게 조정 필요
                    situation = EthicalSituation.from_dict(situation_data)
                    situations.append(situation)

                except json.JSONDecodeError as e:
                    logger.error(f"CSV 파일의 행 처리 중 JSON 파싱 오류 발생: {row}, 오류: {e}")
                except KeyError as e:
                     logger.error(f"CSV 파일의 행 처리 중 필수 키 누락: {row}, 누락된 키: {e}")
                except Exception as e:
                    logger.error(f"CSV 파일의 행 처리 중 예외 발생: {row}, 오류: {e}")

        logger.info(f"{filepath}에서 {len(situations)}개의 윤리적 상황을 로드했습니다.")
        return situations

    except Exception as e:
        logger.error(f"CSV 파일 로드 중 오류 발생: {filepath}, {e}")
        return []

def load_situations_from_json(filepath: str) -> List[EthicalSituation]:
    """
    JSON 파일(객체 리스트)에서 윤리적 상황 데이터를 로드합니다.

    Args:
        filepath: 로드할 JSON 파일 경로

    Returns:
        EthicalSituation 객체 리스트
    """
    situations_data = load_json(filepath, default=[])
    situations = []
    if not isinstance(situations_data, list):
        logger.error(f"JSON 파일 형식이 잘못되었습니다. 객체 리스트여야 합니다: {filepath}")
        return situations

    for data in situations_data:
        try:
            situation = EthicalSituation.from_dict(data)
            situations.append(situation)
        except Exception as e:
            logger.error(f"JSON 데이터 변환 중 오류 발생: {data}, 오류: {e}")

    logger.info(f"{filepath}에서 {len(situations)}개의 윤리적 상황을 로드했습니다.")
    return situations 


# conda 환경 패키지 접근을 위한 sys.path 조정
def setup_conda_path():
    """conda 환경의 site-packages를 sys.path에 추가 (fallback 없음)"""
    import sys
    
    # conda faiss-test 환경의 site-packages 경로
    conda_site_packages = '/home/kkj/miniconda3/envs/faiss-test/lib/python3.12/site-packages'
    
    # 경로가 존재하고 sys.path에 없으면 추가
    if os.path.exists(conda_site_packages) and conda_site_packages not in sys.path:
        sys.path.insert(1, conda_site_packages)
        logger.info(f"conda site-packages 경로 추가: {conda_site_packages}")
    
    return conda_site_packages in sys.path


def run_faiss_subprocess(operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """FAISS 작업을 별도 conda 프로세스에서 실행"""
    import subprocess
    import tempfile
    
    # 임시 파일로 데이터 전달
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'operation': operation, 'data': data}, f)
        input_file = f.name
    
    # 결과 파일
    output_file = input_file.replace('.json', '_output.json')
    
    # conda run으로 FAISS 스크립트 실행 (권장 방법)
    worker_script = os.path.join(os.getcwd(), 'faiss_worker.py')
    cmd = ['conda', 'run', '-n', 'faiss-test', 'python', worker_script, input_file, output_file]
    
    try:
        # 깨끗한 환경에서 conda run 실행
        clean_env = os.environ.copy()
        # venv 관련 환경 변수 제거
        clean_env.pop('VIRTUAL_ENV', None)
        clean_env.pop('VIRTUAL_ENV_PROMPT', None)
        
        # PATH에서 venv 경로 제거
        if 'PATH' in clean_env:
            current_path = clean_env['PATH']
            paths = current_path.split(':')
            clean_paths = [p for p in paths if 'red_heart_env' not in p]
            clean_env['PATH'] = ':'.join(clean_paths)
        
        # Python path 관련 정리
        if 'PYTHONPATH' in clean_env:
            pythonpath = clean_env['PYTHONPATH']
            # venv 경로 제거
            paths = pythonpath.split(':')
            clean_paths = [p for p in paths if 'red_heart_env' not in p]
            if clean_paths:
                clean_env['PYTHONPATH'] = ':'.join(clean_paths)
            else:
                clean_env.pop('PYTHONPATH', None)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=clean_env)
        
        if result.returncode != 0:
            logger.error(f"FAISS subprocess 실패: {result.stderr}")
            raise RuntimeError(f"FAISS 작업 실패: {result.stderr}")
        
        # 결과 로딩
        with open(output_file, 'r') as f:
            return json.load(f)
            
    finally:
        # 임시 파일 정리
        for file_path in [input_file, output_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def run_spacy_subprocess(operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """spacy 작업을 별도 conda 프로세스에서 실행"""
    import subprocess
    import tempfile
    
    # 임시 파일로 데이터 전달
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'operation': operation, 'data': data}, f)
        input_file = f.name
    
    # 결과 파일
    output_file = input_file.replace('.json', '_output.json')
    
    # conda run으로 spacy 스크립트 실행
    worker_script = os.path.join(os.getcwd(), 'spacy_worker.py')
    cmd = ['conda', 'run', '-n', 'faiss-test', 'python', worker_script, input_file, output_file]
    
    try:
        # 깨끗한 환경에서 conda run 실행
        clean_env = os.environ.copy()
        # venv 관련 환경 변수 제거
        clean_env.pop('VIRTUAL_ENV', None)
        clean_env.pop('VIRTUAL_ENV_PROMPT', None)
        
        # PATH에서 venv 경로 제거
        if 'PATH' in clean_env:
            current_path = clean_env['PATH']
            paths = current_path.split(':')
            clean_paths = [p for p in paths if 'red_heart_env' not in p]
            clean_env['PATH'] = ':'.join(clean_paths)
        
        # Python path 관련 정리
        if 'PYTHONPATH' in clean_env:
            pythonpath = clean_env['PYTHONPATH']
            # venv 경로 제거
            paths = pythonpath.split(':')
            clean_paths = [p for p in paths if 'red_heart_env' not in p]
            if clean_paths:
                clean_env['PYTHONPATH'] = ':'.join(clean_paths)
            else:
                clean_env.pop('PYTHONPATH', None)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=clean_env)
        
        if result.returncode != 0:
            logger.error(f"spacy subprocess 실패: {result.stderr}")
            raise RuntimeError(f"spacy 작업 실패: {result.stderr}")
        
        # 결과 로딩
        with open(output_file, 'r') as f:
            return json.load(f)
            
    finally:
        # 임시 파일 정리
        for file_path in [input_file, output_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def safe_import_sentence_transformers():
    """SentenceTransformers를 안전하게 import (conda 경로 설정 후)"""
    setup_conda_path()
    
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("SentenceTransformers import 성공")
        return SentenceTransformer
    except ImportError as e:
        logger.error(f"SentenceTransformers import 실패: {e}")
        raise


# GPU 메모리 관리 함수들
def get_gpu_memory_info():
    """GPU 메모리 사용량 정보 반환"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {'available': False}
        
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - allocated_memory
        
        return {
            'available': True,
            'total_gb': total_memory / 1024**3,
            'allocated_gb': allocated_memory / 1024**3,
            'cached_gb': cached_memory / 1024**3,
            'free_gb': free_memory / 1024**3,
            'usage_percent': (allocated_memory / total_memory) * 100
        }
    except Exception as e:
        logger.warning(f"GPU 메모리 정보 확인 실패: {e}")
        return {'available': False}


def check_gpu_memory_safe(required_gb=1.0, safety_margin=1.0):
    """GPU 메모리 사용이 안전한지 확인"""
    memory_info = get_gpu_memory_info()
    
    if not memory_info['available']:
        return False
    
    # 8GB 총 메모리에서 안전 마진 확인
    max_safe_usage = 8.0 - safety_margin  # 7GB까지만 사용
    current_usage = memory_info['allocated_gb']
    
    return (current_usage + required_gb) <= max_safe_usage


def cleanup_gpu_memory():
    """GPU 메모리 정리"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU 메모리 정리 완료")
            return True
    except Exception as e:
        logger.warning(f"GPU 메모리 정리 실패: {e}")
    return False


def safe_gpu_operation(operation_func, *args, required_memory_gb=0.5, **kwargs):
    """안전한 GPU 작업 실행 (메모리 모니터링 포함)"""
    # GPU 메모리 확인
    if not check_gpu_memory_safe(required_memory_gb):
        logger.warning("GPU 메모리 부족. CPU로 대체 실행합니다.")
        # CPU 모드로 강제 전환
        import torch
        device = torch.device('cpu')
        kwargs['device'] = device
    
    try:
        # 작업 실행 전 메모리 상태
        memory_before = get_gpu_memory_info()
        
        # 실제 작업 실행
        result = operation_func(*args, **kwargs)
        
        # 작업 실행 후 메모리 상태
        memory_after = get_gpu_memory_info()
        
        if memory_after['available']:
            logger.debug(f"GPU 메모리 사용량: {memory_after['allocated_gb']:.2f}GB / {memory_after['total_gb']:.1f}GB")
            
            # 메모리 사용량이 너무 높으면 경고
            if memory_after['usage_percent'] > 85:
                logger.warning(f"GPU 메모리 사용량 높음: {memory_after['usage_percent']:.1f}%")
                cleanup_gpu_memory()
        
        return result
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU 메모리 부족 오류 발생. 메모리 정리 후 CPU로 재시도합니다.")
            cleanup_gpu_memory()
            # CPU로 재시도
            import torch
            device = torch.device('cpu')
            kwargs['device'] = device
            return operation_func(*args, **kwargs)
        else:
            raise


# 동적 모델 로딩 관리
class ModelManager:
    """모델의 동적 로딩/언로딩을 관리하는 클래스"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_configs = {}
        self.access_count = {}
        self.last_access_time = {}
        self.max_loaded_models = 3  # 동시에 메모리에 유지할 최대 모델 수
        
    def register_model(self, model_name: str, model_class, init_args=None, init_kwargs=None):
        """모델 등록 (아직 로딩하지 않음)"""
        self.model_configs[model_name] = {
            'class': model_class,
            'init_args': init_args or [],
            'init_kwargs': init_kwargs or {}
        }
        self.access_count[model_name] = 0
        logger.info(f"모델 등록: {model_name}")
    
    def get_model(self, model_name: str, device=None):
        """모델을 가져오기 (필요시 로딩)"""
        import time
        
        # 이미 로딩된 모델이면 반환
        if model_name in self.loaded_models:
            self.access_count[model_name] += 1
            self.last_access_time[model_name] = time.time()
            logger.debug(f"모델 재사용: {model_name}")
            return self.loaded_models[model_name]
        
        # 메모리 공간 확보
        self._ensure_memory_space()
        
        # 모델 로딩
        if model_name not in self.model_configs:
            raise ValueError(f"등록되지 않은 모델: {model_name}")
        
        config = self.model_configs[model_name]
        
        logger.info(f"모델 로딩 중: {model_name}")
        model = config['class'](*config['init_args'], **config['init_kwargs'])
        
        # GPU로 이동
        if device is None:
            device = get_device()
        model = model.to(device)
        
        # 메모리에 저장
        self.loaded_models[model_name] = model
        self.access_count[model_name] += 1
        self.last_access_time[model_name] = time.time()
        
        logger.info(f"모델 로딩 완료: {model_name}")
        return model
    
    def unload_model(self, model_name: str):
        """특정 모델을 메모리에서 해제"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            cleanup_gpu_memory()
            logger.info(f"모델 언로딩: {model_name}")
            return True
        return False
    
    def _ensure_memory_space(self):
        """메모리 공간 확보 (오래된 모델 언로딩)"""
        if len(self.loaded_models) >= self.max_loaded_models:
            # 가장 오래 사용되지 않은 모델 찾기
            oldest_model = min(
                self.loaded_models.keys(),
                key=lambda x: self.last_access_time.get(x, 0)
            )
            self.unload_model(oldest_model)
    
    def get_status(self):
        """현재 로딩된 모델 상태 반환"""
        memory_info = get_gpu_memory_info()
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'model_count': len(self.loaded_models),
            'gpu_memory': memory_info,
            'access_counts': self.access_count.copy()
        }
    
    def cleanup_all(self):
        """모든 모델을 메모리에서 해제"""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        logger.info("모든 모델 정리 완료")


# 전역 모델 매니저 인스턴스
model_manager = ModelManager()


def get_device():
    """최적 디바이스 반환"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    except ImportError:
        return 'cpu'


def load_model_dynamically(model_name: str, force_reload=False):
    """동적으로 모델 로딩"""
    if force_reload and model_name in model_manager.loaded_models:
        model_manager.unload_model(model_name)
    
    return model_manager.get_model(model_name)

