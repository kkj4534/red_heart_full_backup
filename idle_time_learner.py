#!/usr/bin/env python3
"""
유휴 시간 학습 시스템
Idle Time Learning System

대화가 없는 시간에 자동으로 배치 학습을 수행하는 시스템
"""

import asyncio
import time
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

logger = logging.getLogger('RedHeart.IdleLearner')


class IdleLevel(Enum):
    """유휴 레벨 정의"""
    ACTIVE = "active"           # 활동 중 (0-1분)
    IMMEDIATE = "immediate"     # 즉시 정리 (1-10분)
    SHORT = "short"            # 짧은 유휴 (10-30분)
    MEDIUM = "medium"          # 중간 유휴 (30분-1시간)
    LONG = "long"              # 긴 유휴 (1-8시간)
    OVERNIGHT = "overnight"    # 밤샘 유휴 (8시간+)


@dataclass
class RegretMemory:
    """후회 메모리 항목"""
    timestamp: datetime
    decision: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    predicted_outcome: Dict[str, Any]
    regret_score: float
    context: Dict[str, Any] = field(default_factory=dict)
    learned: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'decision': self.decision,
            'actual_outcome': self.actual_outcome,
            'predicted_outcome': self.predicted_outcome,
            'regret_score': self.regret_score,
            'context': self.context,
            'learned': self.learned
        }


@dataclass
class LearningSession:
    """학습 세션 정보"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    items_processed: int = 0
    total_items: int = 0
    learning_type: str = "regret"
    status: str = "pending"
    metrics: Dict[str, float] = field(default_factory=dict)


class HierarchicalIdleLearner:
    """계층적 유휴 시간 학습 시스템"""
    
    def __init__(self, 
                 regret_buffer_size: int = 1000,
                 experience_cache_size: int = 500):
        
        # 유휴 시간 임계값 (초)
        self.idle_thresholds = {
            IdleLevel.IMMEDIATE: 60,      # 1분
            IdleLevel.SHORT: 600,         # 10분
            IdleLevel.MEDIUM: 1800,       # 30분
            IdleLevel.LONG: 3600,         # 1시간
            IdleLevel.OVERNIGHT: 28800    # 8시간
        }
        
        # 상태 추적
        self.last_interaction_time = time.time()
        self.is_learning = False
        self.current_idle_level = IdleLevel.ACTIVE
        
        # 후회 버퍼
        self.regret_buffer: Deque[RegretMemory] = deque(maxlen=regret_buffer_size)
        self.priority_regrets: List[RegretMemory] = []  # 높은 후회 점수 항목
        
        # 경험 캐시
        self.experience_cache: Deque[Dict] = deque(maxlen=experience_cache_size)
        
        # 학습 세션 기록
        self.learning_sessions: List[LearningSession] = []
        self.current_session: Optional[LearningSession] = None
        
        # 학습 통계
        self.stats = {
            'total_sessions': 0,
            'total_items_learned': 0,
            'total_learning_time': 0.0,
            'regret_reduction': [],
            'last_learning': None
        }
        
        # 모니터링 태스크
        self.monitor_task: Optional[asyncio.Task] = None
        
        logger.info("HierarchicalIdleLearner 초기화 완료")
    
    def record_interaction(self):
        """사용자 상호작용 기록"""
        self.last_interaction_time = time.time()
        self.current_idle_level = IdleLevel.ACTIVE
    
    def add_regret(self, regret_memory: RegretMemory):
        """후회 메모리 추가"""
        self.regret_buffer.append(regret_memory)
        
        # 높은 후회 점수는 우선순위 큐에도 추가
        if regret_memory.regret_score > 0.7:
            self.priority_regrets.append(regret_memory)
            # 점수 기준 정렬
            self.priority_regrets.sort(key=lambda x: x.regret_score, reverse=True)
            # 상위 100개만 유지
            self.priority_regrets = self.priority_regrets[:100]
    
    def add_experience(self, experience: Dict):
        """경험 추가"""
        experience['timestamp'] = datetime.now()
        self.experience_cache.append(experience)
    
    def get_idle_level(self) -> IdleLevel:
        """현재 유휴 레벨 계산"""
        idle_time = time.time() - self.last_interaction_time
        
        for level in reversed(list(IdleLevel)):
            if level == IdleLevel.ACTIVE:
                continue
            if idle_time >= self.idle_thresholds[level]:
                return level
        
        return IdleLevel.ACTIVE
    
    async def start_monitoring(self):
        """유휴 모니터링 시작"""
        if self.monitor_task is not None:
            logger.warning("이미 모니터링 중")
            return
        
        self.monitor_task = asyncio.create_task(self._monitor_and_learn())
        logger.info("유휴 시간 모니터링 시작")
    
    async def stop_monitoring(self):
        """유휴 모니터링 중지"""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
            logger.info("유휴 시간 모니터링 중지")
    
    async def _monitor_and_learn(self):
        """유휴 상태를 모니터링하고 적절한 시점에 학습 수행"""
        
        while True:
            try:
                await asyncio.sleep(30)  # 30초마다 체크
                
                # 현재 유휴 레벨 확인
                current_level = self.get_idle_level()
                
                # 레벨 변경 시 로깅
                if current_level != self.current_idle_level:
                    logger.info(f"유휴 레벨 변경: {self.current_idle_level.value} → {current_level.value}")
                    self.current_idle_level = current_level
                
                # 이미 학습 중이면 스킵
                if self.is_learning:
                    continue
                
                # 레벨별 학습 수행
                if current_level == IdleLevel.IMMEDIATE:
                    # 1-10분: 캐시 정리
                    await self._immediate_cleanup()
                
                elif current_level == IdleLevel.SHORT:
                    # 10-30분: 경험 정리
                    await self._consolidate_experiences()
                
                elif current_level == IdleLevel.MEDIUM:
                    # 30분-1시간: 부분 학습
                    await self._partial_update()
                
                elif current_level == IdleLevel.LONG:
                    # 1시간+: 전체 배치 학습
                    await self._batch_regret_learning()
                
                elif current_level == IdleLevel.OVERNIGHT:
                    # 8시간+: 대규모 재학습
                    await self._deep_retrospective_learning()
                
            except asyncio.CancelledError:
                logger.info("모니터링 태스크 취소됨")
                break
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    async def _immediate_cleanup(self):
        """즉시 정리 (1-10분 유휴)"""
        logger.info("📧 즉시 정리 시작...")
        
        # 중복 제거
        unique_regrets = {}
        for regret in self.regret_buffer:
            key = json.dumps(regret.decision, sort_keys=True)
            if key not in unique_regrets or regret.regret_score > unique_regrets[key].regret_score:
                unique_regrets[key] = regret
        
        # 버퍼 업데이트
        self.regret_buffer = deque(unique_regrets.values(), maxlen=self.regret_buffer.maxlen)
        
        logger.info(f"   중복 제거 완료: {len(unique_regrets)}개 항목")
    
    async def _consolidate_experiences(self):
        """경험 정리 (10-30분 유휴)"""
        logger.info("💾 경험 정리 시작...")
        
        if not self.experience_cache:
            return
        
        # 유사한 경험 그룹화
        experience_groups = {}
        for exp in self.experience_cache:
            # 간단한 해시 키 생성
            key = f"{exp.get('emotion', {}).get('dominant', 'unknown')}_{exp.get('bentham', {}).get('intensity', 0):.1f}"
            if key not in experience_groups:
                experience_groups[key] = []
            experience_groups[key].append(exp)
        
        # 그룹별 대표 경험 추출
        consolidated = []
        for group in experience_groups.values():
            # 평균 또는 가장 최근 경험을 대표로
            if group:
                consolidated.append(group[-1])  # 가장 최근 것
        
        # 캐시 업데이트
        self.experience_cache = deque(consolidated, maxlen=self.experience_cache.maxlen)
        
        logger.info(f"   경험 정리 완료: {len(experience_groups)}개 그룹 → {len(consolidated)}개 대표")
    
    async def _partial_update(self):
        """부분 학습 (30분-1시간 유휴)"""
        logger.info("📚 부분 학습 시작...")
        
        self.is_learning = True
        session = LearningSession(
            session_id=f"partial_{int(time.time())}",
            start_time=datetime.now(),
            learning_type="partial",
            total_items=min(50, len(self.regret_buffer))
        )
        self.current_session = session
        
        try:
            # 최근 50개 항목만 학습
            items_to_learn = list(self.regret_buffer)[-50:]
            
            for i, regret_item in enumerate(items_to_learn):
                if not regret_item.learned:
                    # 간단한 학습 시뮬레이션
                    await self._learn_from_regret(regret_item)
                    regret_item.learned = True
                    session.items_processed += 1
                    
                    # 주기적으로 유휴 상태 체크
                    if i % 10 == 0:
                        if self.get_idle_level() == IdleLevel.ACTIVE:
                            logger.info("   사용자 활동 감지 - 학습 중단")
                            break
                    
                    await asyncio.sleep(0.1)  # 비동기 처리
            
            session.end_time = datetime.now()
            session.status = "completed"
            self.stats['total_items_learned'] += session.items_processed
            
            logger.info(f"   부분 학습 완료: {session.items_processed}개 항목")
            
        except Exception as e:
            logger.error(f"   부분 학습 실패: {e}")
            session.status = "failed"
        finally:
            self.is_learning = False
            self.learning_sessions.append(session)
            self.current_session = None
    
    async def _batch_regret_learning(self):
        """배치 후회 학습 (1시간+ 유휴)"""
        logger.info("🎓 배치 후회 학습 시작...")
        
        self.is_learning = True
        session = LearningSession(
            session_id=f"batch_{int(time.time())}",
            start_time=datetime.now(),
            learning_type="batch",
            total_items=len(self.regret_buffer)
        )
        self.current_session = session
        
        try:
            # 우선순위 항목부터 학습
            all_items = self.priority_regrets + list(self.regret_buffer)
            unique_items = {id(item): item for item in all_items}.values()
            
            batch_size = 32
            batches = [list(unique_items)[i:i+batch_size] 
                      for i in range(0, len(unique_items), batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                # 배치 학습
                await self._learn_batch(batch)
                session.items_processed += len(batch)
                
                # 유휴 상태 체크
                if self.get_idle_level() == IdleLevel.ACTIVE:
                    logger.info("   사용자 활동 감지 - 배치 학습 중단")
                    break
                
                # 진행상황 로깅
                if batch_idx % 5 == 0:
                    progress = (session.items_processed / session.total_items) * 100
                    logger.info(f"   진행률: {progress:.1f}% ({session.items_processed}/{session.total_items})")
                
                await asyncio.sleep(1)  # 배치 간 대기
            
            # 학습 후 버퍼 정리
            learned_items = [item for item in self.regret_buffer if item.learned]
            if len(learned_items) > len(self.regret_buffer) * 0.8:
                # 80% 이상 학습됨 - 버퍼 초기화
                self.regret_buffer.clear()
                logger.info("   후회 버퍼 초기화")
            
            session.end_time = datetime.now()
            session.status = "completed"
            self.stats['total_sessions'] += 1
            self.stats['total_items_learned'] += session.items_processed
            self.stats['last_learning'] = datetime.now()
            
            logger.info(f"   배치 학습 완료: {session.items_processed}개 항목")
            
        except Exception as e:
            logger.error(f"   배치 학습 실패: {e}")
            session.status = "failed"
        finally:
            self.is_learning = False
            self.learning_sessions.append(session)
            self.current_session = None
    
    async def _deep_retrospective_learning(self):
        """대규모 회고 학습 (8시간+ 유휴)"""
        logger.info("🌙 대규모 회고 학습 시작...")
        
        self.is_learning = True
        session = LearningSession(
            session_id=f"deep_{int(time.time())}",
            start_time=datetime.now(),
            learning_type="deep_retrospective",
            total_items=len(self.regret_buffer) + len(self.experience_cache)
        )
        self.current_session = session
        
        try:
            # 1. 모든 후회 재평가
            logger.info("   1단계: 후회 재평가...")
            for regret in self.regret_buffer:
                await self._reevaluate_regret(regret)
                session.items_processed += 1
            
            # 2. 경험 패턴 분석
            logger.info("   2단계: 경험 패턴 분석...")
            patterns = await self._analyze_experience_patterns()
            session.metrics['patterns_found'] = len(patterns)
            
            # 3. 메타 학습
            logger.info("   3단계: 메타 학습...")
            meta_insights = await self._meta_learning(patterns)
            session.metrics['meta_insights'] = len(meta_insights)
            
            # 4. 정책 업데이트
            logger.info("   4단계: 정책 업데이트...")
            await self._update_policies(meta_insights)
            
            session.end_time = datetime.now()
            session.status = "completed"
            
            # 통계 업데이트
            learning_time = (session.end_time - session.start_time).total_seconds()
            self.stats['total_learning_time'] += learning_time
            
            logger.info(f"   대규모 회고 학습 완료")
            logger.info(f"   - 처리 항목: {session.items_processed}")
            logger.info(f"   - 발견 패턴: {session.metrics.get('patterns_found', 0)}")
            logger.info(f"   - 메타 인사이트: {session.metrics.get('meta_insights', 0)}")
            logger.info(f"   - 소요 시간: {learning_time/60:.1f}분")
            
        except Exception as e:
            logger.error(f"   대규모 회고 학습 실패: {e}")
            session.status = "failed"
        finally:
            self.is_learning = False
            self.learning_sessions.append(session)
            self.current_session = None
    
    async def _learn_from_regret(self, regret_item: RegretMemory):
        """단일 후회 항목으로부터 학습"""
        # 간단한 학습 시뮬레이션
        await asyncio.sleep(0.01)
        
        # 후회 점수 감소 (학습 효과)
        regret_item.regret_score *= 0.9
        self.stats['regret_reduction'].append(regret_item.regret_score)
    
    async def _learn_batch(self, batch: List[RegretMemory]):
        """배치 학습"""
        # 배치 학습 시뮬레이션
        await asyncio.sleep(0.1 * len(batch))
        
        for item in batch:
            item.learned = True
            item.regret_score *= 0.85  # 배치 학습이 더 효과적
    
    async def _reevaluate_regret(self, regret_item: RegretMemory):
        """후회 재평가"""
        # 시간이 지난 후회는 중요도 감소
        age_days = (datetime.now() - regret_item.timestamp).days
        if age_days > 7:
            regret_item.regret_score *= 0.7
    
    async def _analyze_experience_patterns(self) -> List[Dict]:
        """경험 패턴 분석"""
        patterns = []
        
        # 간단한 패턴 추출 시뮬레이션
        if len(self.experience_cache) > 10:
            pattern = {
                'type': 'recurring_emotion',
                'confidence': 0.8,
                'description': '반복되는 감정 패턴 발견'
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _meta_learning(self, patterns: List[Dict]) -> List[Dict]:
        """메타 학습"""
        insights = []
        
        for pattern in patterns:
            insight = {
                'pattern': pattern,
                'learning': f"{pattern['type']}에 대한 개선 전략",
                'priority': pattern['confidence']
            }
            insights.append(insight)
        
        return insights
    
    async def _update_policies(self, insights: List[Dict]):
        """정책 업데이트"""
        # 정책 업데이트 시뮬레이션
        for insight in insights:
            logger.debug(f"정책 업데이트: {insight['learning']}")
    
    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'idle_level': self.current_idle_level.value,
            'idle_seconds': time.time() - self.last_interaction_time,
            'is_learning': self.is_learning,
            'current_session': self.current_session.to_dict() if self.current_session else None,
            'regret_buffer_size': len(self.regret_buffer),
            'priority_regrets': len(self.priority_regrets),
            'experience_cache_size': len(self.experience_cache),
            'stats': self.stats
        }
    
    def save_state(self, filepath: Path):
        """상태 저장"""
        state = {
            'regret_buffer': [r.to_dict() for r in self.regret_buffer],
            'priority_regrets': [r.to_dict() for r in self.priority_regrets],
            'experience_cache': list(self.experience_cache),
            'stats': self.stats,
            'learning_sessions': [s.__dict__ for s in self.learning_sessions[-100:]]  # 최근 100개
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"상태 저장: {filepath}")
    
    def load_state(self, filepath: Path):
        """상태 로드"""
        if not filepath.exists():
            logger.warning(f"상태 파일 없음: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # 복원
        self.regret_buffer = deque(
            [RegretMemory(**r) for r in state['regret_buffer']], 
            maxlen=self.regret_buffer.maxlen
        )
        self.priority_regrets = [RegretMemory(**r) for r in state['priority_regrets']]
        self.experience_cache = deque(state['experience_cache'], maxlen=self.experience_cache.maxlen)
        self.stats = state['stats']
        
        logger.info(f"상태 로드: {filepath}")


# 사용 예제
async def example_usage():
    """사용 예제"""
    
    # 학습기 생성
    learner = HierarchicalIdleLearner()
    
    # 모니터링 시작
    await learner.start_monitoring()
    
    # 후회 메모리 추가 예제
    for i in range(5):
        regret = RegretMemory(
            timestamp=datetime.now(),
            decision={'action': f'decision_{i}'},
            actual_outcome={'result': 'bad'},
            predicted_outcome={'result': 'good'},
            regret_score=0.5 + i * 0.1
        )
        learner.add_regret(regret)
    
    # 경험 추가 예제
    for i in range(3):
        experience = {
            'emotion': {'valence': 0.5, 'arousal': 0.3},
            'bentham': {'intensity': 0.6},
            'outcome': 'positive'
        }
        learner.add_experience(experience)
    
    # 상호작용 시뮬레이션
    print("시뮬레이션 시작...")
    for minute in range(10):
        print(f"\n분 {minute}:")
        
        # 처음 2분은 활동
        if minute < 2:
            learner.record_interaction()
            print("  사용자 활동 중...")
        
        # 상태 출력
        status = learner.get_status()
        print(f"  유휴 레벨: {status['idle_level']}")
        print(f"  유휴 시간: {status['idle_seconds']:.0f}초")
        print(f"  학습 중: {status['is_learning']}")
        
        await asyncio.sleep(1)  # 실제로는 60초
    
    # 모니터링 중지
    await learner.stop_monitoring()
    
    # 최종 통계
    print("\n최종 통계:")
    print(f"  총 세션: {learner.stats['total_sessions']}")
    print(f"  학습 항목: {learner.stats['total_items_learned']}")
    print(f"  총 학습 시간: {learner.stats['total_learning_time']:.1f}초")


if __name__ == '__main__':
    asyncio.run(example_usage())