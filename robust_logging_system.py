#!/usr/bin/env python3
"""
Robust Logging System for Red Heart AI
견고한 로깅 시스템 - 안정적인 결과 파일 생성 및 GPU 메모리 추적

Features:
- 강제 결과 파일 생성 보장
- 정확한 GPU 메모리 사용량 추적
- 실시간 로그 버퍼링 및 백업
- 시스템 안정성 모니터링
- 오류 복구 및 롤백 시스템
"""

import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import queue
import asyncio
import torch
import os
from contextlib import contextmanager

# psutil 대체 - 시스템 메모리 추적용
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil 미설치 - 시스템 메모리 추적 제한")

@dataclass
class LogEntry:
    """로그 엔트리 데이터 클래스"""
    timestamp: str
    level: str
    source: str
    message: str
    gpu_memory_mb: float
    system_memory_mb: float
    metadata: Dict[str, Any]

@dataclass
class TestResult:
    """테스트 결과 데이터 클래스"""
    test_id: str
    test_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    status: str  # success, failed, error
    gpu_peak_memory_mb: float
    gpu_avg_memory_mb: float
    system_metrics: Dict[str, Any]
    performance_data: Dict[str, Any]
    error_info: Optional[Dict[str, Any]] = None

class RobustLogger:
    """견고한 로깅 시스템"""
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(exist_ok=True)
        
        # 세션별 로그 디렉토리
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 로그 큐 및 스레드
        self.log_queue = queue.Queue()
        self.running = True
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
        
        # 메모리 추적
        self.gpu_memory_tracker = []
        self.system_memory_tracker = []
        
        # 테스트 결과 저장
        self.test_results: List[TestResult] = []
        self.current_test: Optional[Dict[str, Any]] = None
        
        # 백업 및 안정성
        self.backup_enabled = True
        self.force_write_interval = 30  # 30초마다 강제 저장
        self.last_force_write = time.time()
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        print(f"🔧 견고한 로깅 시스템 초기화 - 세션: {self.session_id}")
        print(f"📁 로그 디렉토리: {self.session_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"RobustLogger_{self.session_id}")
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        log_file = self.session_dir / "system.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _log_worker(self):
        """로그 워커 스레드"""
        while self.running:
            try:
                # 큐에서 로그 엔트리 처리
                try:
                    entry = self.log_queue.get(timeout=1.0)
                    self._write_log_entry(entry)
                    self.log_queue.task_done()
                except queue.Empty:
                    pass
                
                # 주기적 강제 저장
                current_time = time.time()
                if current_time - self.last_force_write > self.force_write_interval:
                    self._force_write_all()
                    self.last_force_write = current_time
                    
            except Exception as e:
                print(f"⚠️ 로그 워커 오류: {e}")
    
    def _write_log_entry(self, entry: LogEntry):
        """로그 엔트리 기록"""
        try:
            # 시스템 로그에 기록
            self.logger.info(f"{entry.source}: {entry.message}")
            
            # JSON 로그 파일에 추가
            json_log_file = self.session_dir / "detailed_logs.jsonl"
            with open(json_log_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(entry), f, ensure_ascii=False)
                f.write('\n')
                f.flush()  # 강제 플러시
                
        except Exception as e:
            print(f"⚠️ 로그 기록 오류: {e}")
    
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 조회"""
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        # 시스템 메모리 추적 - psutil 사용 가능시에만
        system_memory = 0.0
        if PSUTIL_AVAILABLE:
            try:
                system_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            except:
                system_memory = 0.0
        
        return {
            "gpu_memory_mb": gpu_memory,
            "system_memory_mb": system_memory
        }
    
    def log(self, level: str, source: str, message: str, metadata: Dict[str, Any] = None):
        """로그 메시지 기록"""
        if metadata is None:
            metadata = {}
        
        memory_usage = self._get_current_memory_usage()
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            source=source,
            message=message,
            gpu_memory_mb=memory_usage["gpu_memory_mb"],
            system_memory_mb=memory_usage["system_memory_mb"],
            metadata=metadata
        )
        
        # 메모리 추적 업데이트
        self.gpu_memory_tracker.append(memory_usage["gpu_memory_mb"])
        self.system_memory_tracker.append(memory_usage["system_memory_mb"])
        
        # 큐에 추가
        try:
            self.log_queue.put(entry, timeout=5.0)
        except queue.Full:
            print("⚠️ 로그 큐 가득참 - 직접 기록")
            self._write_log_entry(entry)
    
    @contextmanager
    def test_session(self, test_name: str, test_metadata: Dict[str, Any] = None):
        """테스트 세션 컨텍스트 매니저"""
        if test_metadata is None:
            test_metadata = {}
        
        test_id = f"{test_name}_{int(time.time())}"
        start_time = datetime.now()
        start_memory = self._get_current_memory_usage()
        
        self.current_test = {
            "test_id": test_id,
            "test_name": test_name,
            "start_time": start_time.isoformat(),
            "start_memory": start_memory,
            "metadata": test_metadata,
            "gpu_memory_samples": [],
            "performance_samples": []
        }
        
        self.log("INFO", "TestSession", f"테스트 시작: {test_name}", {"test_id": test_id})
        
        try:
            yield test_id
            
            # 성공 완료
            self._finalize_test_session("success")
            
        except Exception as e:
            # 오류 완료
            self._finalize_test_session("error", {"error": str(e), "error_type": type(e).__name__})
            raise
    
    def _finalize_test_session(self, status: str, error_info: Dict[str, Any] = None):
        """테스트 세션 완료 처리"""
        if not self.current_test:
            return
        
        end_time = datetime.now()
        end_memory = self._get_current_memory_usage()
        
        # 지속 시간 계산
        start_dt = datetime.fromisoformat(self.current_test["start_time"])
        duration = (end_time - start_dt).total_seconds()
        
        # GPU 메모리 통계
        gpu_samples = self.current_test["gpu_memory_samples"]
        if gpu_samples:
            gpu_peak = max(gpu_samples)
            gpu_avg = sum(gpu_samples) / len(gpu_samples)
        else:
            gpu_peak = end_memory["gpu_memory_mb"]
            gpu_avg = end_memory["gpu_memory_mb"]
        
        # 테스트 결과 생성
        test_result = TestResult(
            test_id=self.current_test["test_id"],
            test_name=self.current_test["test_name"],
            start_time=self.current_test["start_time"],
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            status=status,
            gpu_peak_memory_mb=gpu_peak,
            gpu_avg_memory_mb=gpu_avg,
            system_metrics={
                "start_memory": self.current_test["start_memory"],
                "end_memory": end_memory,
                "peak_system_memory": max(self.system_memory_tracker[-100:]) if self.system_memory_tracker else 0
            },
            performance_data=self.current_test.get("performance_data", {}),
            error_info=error_info
        )
        
        self.test_results.append(test_result)
        
        # 로그 기록
        self.log("INFO", "TestSession", 
                f"테스트 완료: {self.current_test['test_name']} - {status} ({duration:.2f}초)",
                {"test_id": self.current_test["test_id"], "status": status})
        
        # 즉시 저장
        self._save_test_result(test_result)
        
        self.current_test = None
    
    def add_performance_sample(self, sample_data: Dict[str, Any]):
        """성능 샘플 데이터 추가"""
        if self.current_test:
            memory_usage = self._get_current_memory_usage()
            self.current_test["gpu_memory_samples"].append(memory_usage["gpu_memory_mb"])
            self.current_test["performance_samples"].append({
                "timestamp": datetime.now().isoformat(),
                "memory": memory_usage,
                "data": sample_data
            })
    
    def _save_test_result(self, test_result: TestResult):
        """개별 테스트 결과 저장"""
        try:
            # 개별 테스트 결과 파일
            result_file = self.session_dir / f"test_result_{test_result.test_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(test_result), f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # 강제 디스크 동기화
            
            # 전체 테스트 결과 업데이트
            self._save_all_results()
            
        except Exception as e:
            print(f"⚠️ 테스트 결과 저장 오류: {e}")
    
    def _save_all_results(self):
        """전체 결과 저장"""
        try:
            # 모든 테스트 결과 요약
            summary = {
                "session_id": self.session_id,
                "total_tests": len(self.test_results),
                "successful_tests": len([r for r in self.test_results if r.status == "success"]),
                "failed_tests": len([r for r in self.test_results if r.status in ["failed", "error"]]),
                "total_duration": sum(r.duration_seconds for r in self.test_results),
                "avg_gpu_memory": sum(r.gpu_avg_memory_mb for r in self.test_results) / len(self.test_results) if self.test_results else 0,
                "peak_gpu_memory": max(r.gpu_peak_memory_mb for r in self.test_results) if self.test_results else 0,
                "timestamp": datetime.now().isoformat(),
                "individual_results": [asdict(r) for r in self.test_results]
            }
            
            # 요약 파일 저장
            summary_file = self.session_dir / "session_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # 백업 파일도 생성
            if self.backup_enabled:
                backup_file = self.base_log_dir / f"backup_session_{self.session_id}.json"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            
        except Exception as e:
            print(f"⚠️ 전체 결과 저장 오류: {e}")
    
    def _force_write_all(self):
        """모든 대기 중인 로그 강제 기록"""
        try:
            # 큐의 모든 엔트리 처리
            while not self.log_queue.empty():
                try:
                    entry = self.log_queue.get_nowait()
                    self._write_log_entry(entry)
                    self.log_queue.task_done()
                except queue.Empty:
                    break
            
            # 현재 테스트 결과 저장
            if self.test_results:
                self._save_all_results()
                
        except Exception as e:
            print(f"⚠️ 강제 저장 오류: {e}")
    
    def generate_continuous_test_report(self, test_count: int = 10) -> Dict[str, Any]:
        """연속 테스트 보고서 생성"""
        continuous_results = self.test_results[-test_count:] if len(self.test_results) >= test_count else self.test_results
        
        if not continuous_results:
            return {"error": "No test results available"}
        
        # 통계 계산
        successful = [r for r in continuous_results if r.status == "success"]
        failed = [r for r in continuous_results if r.status != "success"]
        
        report = {
            "continuous_test_summary": {
                "total_tests": len(continuous_results),
                "successful_tests": len(successful),
                "failed_tests": len(failed),
                "success_rate": len(successful) / len(continuous_results) * 100,
                "total_time": sum(r.duration_seconds for r in continuous_results),
                "avg_test_time": sum(r.duration_seconds for r in continuous_results) / len(continuous_results)
            },
            "performance_metrics": {
                "avg_gpu_memory_mb": sum(r.gpu_avg_memory_mb for r in continuous_results) / len(continuous_results),
                "peak_gpu_memory_mb": max(r.gpu_peak_memory_mb for r in continuous_results),
                "memory_efficiency": "high" if max(r.gpu_peak_memory_mb for r in continuous_results) < 2000 else "medium"
            },
            "detailed_results": [asdict(r) for r in continuous_results],
            "generated_at": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        # 보고서 저장
        report_file = self.session_dir / f"continuous_test_report_{test_count}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        return report
    
    def shutdown(self):
        """로깅 시스템 종료"""
        self.log("INFO", "System", "로깅 시스템 종료 시작")
        
        # 모든 대기 로그 처리
        self._force_write_all()
        
        # 워커 스레드 종료
        self.running = False
        if self.log_thread.is_alive():
            self.log_thread.join(timeout=5.0)
        
        # 최종 저장
        self._save_all_results()
        
        print(f"✅ 로깅 시스템 종료 완료 - {len(self.test_results)}개 테스트 결과 저장됨")

# 전역 로거 인스턴스
_robust_logger = None

def get_robust_logger() -> RobustLogger:
    """전역 견고한 로거 인스턴스 반환"""
    global _robust_logger
    if _robust_logger is None:
        _robust_logger = RobustLogger()
    return _robust_logger

# 편의 함수들
def log_info(source: str, message: str, metadata: Dict[str, Any] = None):
    """INFO 레벨 로그"""
    get_robust_logger().log("INFO", source, message, metadata)

def log_error(source: str, message: str, metadata: Dict[str, Any] = None):
    """ERROR 레벨 로그"""
    get_robust_logger().log("ERROR", source, message, metadata)

def test_session(test_name: str, metadata: Dict[str, Any] = None):
    """테스트 세션 컨텍스트 매니저"""
    return get_robust_logger().test_session(test_name, metadata)

def add_performance_sample(sample_data: Dict[str, Any]):
    """성능 샘플 추가"""
    get_robust_logger().add_performance_sample(sample_data)

def generate_test_report(test_count: int = 10) -> Dict[str, Any]:
    """테스트 보고서 생성"""
    return get_robust_logger().generate_continuous_test_report(test_count)