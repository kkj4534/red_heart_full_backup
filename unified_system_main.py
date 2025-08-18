#!/usr/bin/env python3
"""
Red Heart AI 통합 시스템 메인 진입점
Red Heart AI Unified System Main Entry Point

run_learning.sh와 연동하는 메인 실행 파일:
- 커맨드라인 인자 처리
- 800M 파라미터 통합 시스템 실행
- 실시간 모니터링 및 로깅
- 자동 복구 및 최적화
- 시스템 상태 리포트 생성
"""

import asyncio
import argparse
import sys
import signal
import json
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# 핵심 시스템 임포트
from unified_system_orchestrator import UnifiedSystemOrchestrator, SystemStatus
from unified_learning_system import TrainingStrategy
from config import ADVANCED_CONFIG

def setup_argument_parser() -> argparse.ArgumentParser:
    """커맨드라인 인자 파서 설정"""
    
    parser = argparse.ArgumentParser(
        description="Red Heart AI 통합 시스템 (800M 파라미터)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python unified_system_main.py --mode auto --samples 1000
  python unified_system_main.py --mode training --epochs 5 --batch-size 8
  python unified_system_main.py --mode monitoring --duration 3600
  python unified_system_main.py --mode test --samples 50 --debug
        """
    )
    
    # 기본 옵션
    parser.add_argument('--mode', '-m', 
                       choices=['auto', 'training', 'test', 'monitoring', 'validate', 'dashboard'],
                       default='auto',
                       help='실행 모드 (기본: auto)')
    
    # 훈련 관련 옵션
    training_group = parser.add_argument_group('훈련 옵션')
    training_group.add_argument('--samples', type=int, default=1000,
                               help='처리할 샘플 수 (기본: 1000)')
    training_group.add_argument('--epochs', type=int, default=3,
                               help='훈련 에포크 수 (기본: 3)')
    training_group.add_argument('--batch-size', type=int, default=4,
                               help='배치 크기 (기본: 4)')
    training_group.add_argument('--learning-rate', type=float, default=1e-4,
                               help='학습률 (기본: 1e-4)')
    training_group.add_argument('--strategy',
                               choices=['round_robin', 'parallel', 'priority_based', 'adaptive'],
                               default='round_robin',
                               help='훈련 전략 (기본: round_robin)')
    
    # 시스템 옵션
    system_group = parser.add_argument_group('시스템 옵션')
    system_group.add_argument('--timeout', type=int, default=3600,
                             help='최대 실행 시간 (초, 기본: 3600)')
    system_group.add_argument('--duration', type=int, default=1800,
                             help='모니터링 지속 시간 (초, 기본: 1800)')
    system_group.add_argument('--dashboard-port', type=int, default=8080,
                             help='대시보드 포트 (기본: 8080)')
    system_group.add_argument('--auto-recovery', action='store_true', default=True,
                             help='자동 복구 활성화')
    system_group.add_argument('--no-auto-recovery', dest='auto_recovery', action='store_false',
                             help='자동 복구 비활성화')
    
    # 출력 옵션
    output_group = parser.add_argument_group('출력 옵션')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='상세 로그 출력')
    output_group.add_argument('--debug', action='store_true',
                             help='디버그 모드')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='최소한의 출력')
    output_group.add_argument('--output-dir', type=str, default='./outputs',
                             help='출력 디렉토리 (기본: ./outputs)')
    output_group.add_argument('--report', action='store_true',
                             help='실행 완료 후 리포트 생성')
    
    # 고급 옵션
    advanced_group = parser.add_argument_group('고급 옵션')
    advanced_group.add_argument('--config-file', type=str,
                               help='설정 파일 경로')
    advanced_group.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                               help='체크포인트 디렉토리')
    advanced_group.add_argument('--force-cpu', action='store_true',
                               help='GPU 사용 강제 비활성화')
    advanced_group.add_argument('--memory-limit', type=int,
                               help='메모리 사용량 제한 (MB)')
    
    return parser

class UnifiedSystemRunner:
    """통합 시스템 실행기"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.orchestrator = None
        self.start_time = datetime.now()
        self.is_running = False
        
        # 출력 디렉토리 설정
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 설정 로드
        self.config = self._load_config()
        
        # 신호 처리 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        config = ADVANCED_CONFIG.copy()
        
        # 커맨드라인 인자로 설정 업데이트
        config.update({
            'dashboard_port': self.args.dashboard_port,
            'auto_recovery': self.args.auto_recovery,
            'enable_dashboard': self.args.mode == 'dashboard',
            'force_cpu': self.args.force_cpu,
            'memory_limit': self.args.memory_limit,
            'checkpoint_dir': self.args.checkpoint_dir,
            'fast_init_mode': self.args.mode == 'validate'  # validation 모드에서만 빠른 초기화
        })
        
        # 설정 파일이 있으면 로드
        if self.args.config_file:
            config_file = self.args.config_file
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    config.update(file_config)
        
        return config
    
    def _signal_handler(self, signum, frame):
        """신호 처리기"""
        print(f"\n신호 {signum} 수신 - 시스템 종료 중...")
        self.is_running = False
        if self.orchestrator:
            asyncio.create_task(self.orchestrator.shutdown_system())
    
    async def run(self) -> int:
        """메인 실행"""
        
        try:
            if not self.args.quiet:
                self._print_header()
            
            # 오케스트레이터 초기화
            self.orchestrator = UnifiedSystemOrchestrator(self.config)
            
            # 시스템 초기화
            if not await self.orchestrator.initialize_system():
                self._log("❌ 시스템 초기화 실패")
                return 1
            
            if not self.args.quiet:
                self._log("✅ 시스템 초기화 완료")
            
            self.is_running = True
            
            # 모드별 실행
            if self.args.mode == 'auto':
                return await self._run_auto_mode()
            elif self.args.mode == 'training':
                return await self._run_training_mode()
            elif self.args.mode == 'test':
                return await self._run_test_mode()
            elif self.args.mode == 'monitoring':
                return await self._run_monitoring_mode()
            elif self.args.mode == 'validate':
                return await self._run_validation_mode()
            elif self.args.mode == 'dashboard':
                return await self._run_dashboard_mode()
            else:
                self._log(f"❌ 알 수 없는 모드: {self.args.mode}")
                return 1
        
        except KeyboardInterrupt:
            self._log("\n사용자 중단 요청")
            return 130
        
        except Exception as e:
            self._log(f"❌ 실행 중 오류: {str(e)}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            return 1
        
        finally:
            if self.orchestrator and self.is_running:
                await self.orchestrator.shutdown_system()
    
    async def _run_auto_mode(self) -> int:
        """자동 모드 실행"""
        
        self._log("🚀 자동 통합 모드 시작...")
        
        # 훈련 파이프라인 실행
        training_kwargs = {
            'samples': self.args.samples,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.learning_rate,
            'epochs': self.args.epochs,
            'timeout': self.args.timeout,
            'verbose': self.args.verbose,
            'debug': self.args.debug
        }
        
        success = await self.orchestrator.run_training_pipeline(
            mode="auto", **training_kwargs
        )
        
        if success:
            self._log("✅ 자동 통합 모드 완료")
            await self._generate_report()
            return 0
        else:
            self._log("❌ 자동 통합 모드 실패")
            return 1
    
    async def _run_training_mode(self) -> int:
        """훈련 모드 실행"""
        
        self._log(f"📚 훈련 모드 시작 ({self.args.strategy} 전략)...")
        
        # 훈련 전략 매핑
        strategy_map = {
            'round_robin': TrainingStrategy.ROUND_ROBIN,
            'parallel': TrainingStrategy.PARALLEL,
            'priority_based': TrainingStrategy.PRIORITY_BASED,
            'adaptive': TrainingStrategy.ADAPTIVE
        }
        
        strategy = strategy_map.get(self.args.strategy, TrainingStrategy.ADAPTIVE)
        
        # 가상의 데이터 로더 생성
        class DummyDataLoader:
            def __init__(self, num_samples, batch_size):
                self.num_batches = num_samples // batch_size
                self.batch_size = batch_size
            
            def __iter__(self):
                for i in range(self.num_batches):
                    yield {
                        'text': f'Training sample batch {i}',
                        'batch_size': self.batch_size,
                        'labels': [i % 10] * self.batch_size  # 0-9 라벨
                    }
        
        train_loader = DummyDataLoader(self.args.samples, self.args.batch_size)
        val_loader = DummyDataLoader(self.args.samples // 5, self.args.batch_size)
        
        # 통합 학습 시스템으로 훈련
        try:
            await self.orchestrator.learning_system.train_unified_system(
                train_data_loader=train_loader,
                validation_data_loader=val_loader,
                num_epochs=self.args.epochs,
                training_strategy=strategy
            )
            
            self._log("✅ 훈련 모드 완료")
            await self._generate_report()
            return 0
            
        except Exception as e:
            self._log(f"❌ 훈련 중 오류: {str(e)}")
            return 1
    
    async def _run_test_mode(self) -> int:
        """테스트 모드 실행"""
        
        self._log("🧪 테스트 모드 시작...")
        
        # 빠른 테스트를 위한 소규모 설정
        test_kwargs = {
            'samples': min(self.args.samples, 100),  # 최대 100 샘플
            'batch_size': 2,
            'epochs': 1,
            'timeout': 300,  # 5분 제한
            'verbose': True,
            'debug': self.args.debug
        }
        
        success = await self.orchestrator.run_training_pipeline(
            mode="test", **test_kwargs
        )
        
        if success:
            self._log("✅ 테스트 모드 완료")
            # 테스트 결과 간단 리포트
            await self._generate_test_report()
            return 0
        else:
            self._log("❌ 테스트 모드 실패")
            return 1
    
    async def _run_monitoring_mode(self) -> int:
        """모니터링 모드 실행"""
        
        self._log(f"📊 모니터링 모드 시작 ({self.args.duration}초간)...")
        
        start_time = time.time()
        
        try:
            while self.is_running and (time.time() - start_time) < self.args.duration:
                # 시스템 상태 출력
                status = self.orchestrator.get_system_status()
                
                if not self.args.quiet:
                    self._print_monitoring_status(status)
                
                # 5초마다 업데이트
                await asyncio.sleep(5)
            
            self._log("✅ 모니터링 모드 완료")
            await self._generate_monitoring_report()
            return 0
            
        except Exception as e:
            self._log(f"❌ 모니터링 중 오류: {str(e)}")
            return 1
    
    async def _run_validation_mode(self) -> int:
        """검증 모드 실행"""
        
        self._log("🔍 시스템 검증 모드 시작...")
        
        # 시스템 상태 검증
        status = self.orchestrator.get_system_status()
        
        # 검증 결과 분석
        issues = []
        warnings = []
        
        # 시스템 상태 체크
        if status['system_status'] != 'ready':
            issues.append(f"시스템 상태 이상: {status['system_status']}")
        
        # 컴포넌트 건강도 체크
        monitor_summary = status.get('monitor_summary', {})
        overall_health = monitor_summary.get('overall_health', 0)
        
        if overall_health < 0.7:
            issues.append(f"전체 시스템 건강도 낮음: {overall_health:.1%}")
        elif overall_health < 0.9:
            warnings.append(f"시스템 건강도 주의: {overall_health:.1%}")
        
        # 메모리 사용량 체크
        current_metrics = monitor_summary.get('current_metrics', {})
        memory_usage = current_metrics.get('memory_usage', 0)
        
        if memory_usage > 90:
            issues.append(f"높은 메모리 사용률: {memory_usage:.1f}%")
        elif memory_usage > 80:
            warnings.append(f"메모리 사용률 주의: {memory_usage:.1f}%")
        
        # 결과 출력
        if issues:
            self._log("❌ 시스템 검증 실패:")
            for issue in issues:
                self._log(f"  - {issue}")
            return 1
        elif warnings:
            self._log("⚠️ 시스템 검증 경고:")
            for warning in warnings:
                self._log(f"  - {warning}")
            self._log("✅ 시스템 검증 완료 (경고 포함)")
            return 0
        else:
            self._log("✅ 시스템 검증 완료 - 모든 시스템 정상")
            return 0
    
    async def _run_dashboard_mode(self) -> int:
        """대시보드 모드 실행"""
        
        self._log(f"📈 웹 대시보드 모드 시작 (포트: {self.args.dashboard_port})...")
        
        # 대시보드 실행 (실제 구현에서는 웹 서버 시작)
        self._log(f"🌐 대시보드 접속: http://localhost:{self.args.dashboard_port}")
        self._log("Ctrl+C로 종료")
        
        try:
            # 대시보드가 실행되는 동안 시스템 상태 모니터링
            while self.is_running:
                await asyncio.sleep(10)
                if not self.args.quiet:
                    status = self.orchestrator.get_system_status()
                    uptime = status['uptime_seconds']
                    self._log(f"대시보드 가동 중... (가동시간: {uptime:.0f}초)")
            
            self._log("✅ 대시보드 모드 종료")
            return 0
            
        except Exception as e:
            self._log(f"❌ 대시보드 오류: {str(e)}")
            return 1
    
    def _print_header(self):
        """헤더 출력"""
        print("=" * 60)
        print("🚀 Red Heart AI 통합 시스템 (800M 파라미터)")
        print("=" * 60)
        print(f"모드: {self.args.mode}")
        print(f"시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.args.mode in ['training', 'auto']:
            print(f"샘플 수: {self.args.samples:,}")
            print(f"배치 크기: {self.args.batch_size}")
            print(f"에포크 수: {self.args.epochs}")
        print("-" * 60)
    
    def _print_monitoring_status(self, status: Dict[str, Any]):
        """모니터링 상태 출력"""
        monitor_summary = status.get('monitor_summary', {})
        current_metrics = monitor_summary.get('current_metrics', {})
        
        print(f"\r📊 CPU: {current_metrics.get('cpu_usage', 0):.1f}% | "
              f"메모리: {current_metrics.get('memory_usage', 0):.1f}% | "
              f"GPU: {current_metrics.get('gpu_usage', 0):.1f}% | "
              f"상태: {status.get('system_status', 'unknown')}", end='', flush=True)
    
    def _log(self, message: str):
        """로깅"""
        if not self.args.quiet:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] {message}")
    
    async def _generate_report(self):
        """실행 완료 리포트 생성"""
        if not self.args.report:
            return
        
        status = self.orchestrator.get_system_status()
        
        report = {
            'execution_summary': {
                'mode': self.args.mode,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                'success': True
            },
            'system_status': status,
            'configuration': {
                'samples': self.args.samples,
                'batch_size': self.args.batch_size,
                'epochs': self.args.epochs,
                'learning_rate': self.args.learning_rate,
                'strategy': self.args.strategy
            }
        }
        
        # 리포트 파일 저장 - pathlib 대신 os.path.join 사용
        report_file = os.path.join(self.output_dir, f"execution_report_{int(time.time())}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self._log(f"📄 실행 리포트 생성: {report_file}")
    
    async def _generate_test_report(self):
        """테스트 리포트 생성"""
        status = self.orchestrator.get_system_status()
        
        print("\n" + "=" * 50)
        print("🧪 테스트 결과 요약")
        print("=" * 50)
        print(f"시스템 상태: {status['system_status']}")
        print(f"전체 건강도: {status['monitor_summary'].get('overall_health', 0):.1%}")
        print(f"복구 시도: {status['recovery_attempts']}회")
        print("=" * 50)
    
    async def _generate_monitoring_report(self):
        """모니터링 리포트 생성"""
        status = self.orchestrator.get_system_status()
        
        print(f"\n📊 모니터링 완료 - 지속시간: {self.args.duration}초")
        print(f"최종 시스템 상태: {status['system_status']}")
        print(f"평균 건강도: {status['monitor_summary'].get('overall_health', 0):.1%}")

async def main():
    """메인 함수"""
    
    # 커맨드라인 인자 파싱
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 시스템 실행기 생성 및 실행
    runner = UnifiedSystemRunner(args)
    exit_code = await runner.run()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())