#!/usr/bin/env python3
"""
Experience DB 자동 정리 스크립트 (Phase 1 개선)

상위 K% 후회 점수 데이터만 유지하여 DB 크기를 최적화합니다.
"""
import sqlite3
import argparse
import sys
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any
import shutil

class ExperienceDBPruner:
    """Experience DB 자동 정리기"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backup_path = None
        
    def create_backup(self) -> str:
        """백업 파일 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(self.db_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_filename = f"ethics_policy_backup_{timestamp}.db"
        self.backup_path = os.path.join(backup_dir, backup_filename)
        
        shutil.copy2(self.db_path, self.backup_path)
        print(f"📦 백업 생성: {self.backup_path}")
        return self.backup_path
    
    def analyze_database(self) -> Dict[str, Any]:
        """데이터베이스 현재 상태 분석"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 전체 경험 수
            cursor.execute("SELECT COUNT(*) FROM ethics_experiences")
            total_count = cursor.fetchone()[0]
            
            # 후회 점수 통계
            cursor.execute("""
                SELECT 
                    MIN(actual_regret) as min_regret,
                    MAX(actual_regret) as max_regret,
                    AVG(actual_regret) as avg_regret,
                    COUNT(CASE WHEN actual_regret > 0 THEN 1 END) as regret_experiences
                FROM ethics_experiences
            """)
            regret_stats = cursor.fetchone()
            
            # 시간 범위
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM ethics_experiences
            """)
            time_range = cursor.fetchone()
            
            # 사용자별 분포
            cursor.execute("""
                SELECT user_id, COUNT(*) as count 
                FROM ethics_experiences 
                GROUP BY user_id
            """)
            user_distribution = cursor.fetchall()
            
            conn.close()
            
            analysis = {
                'total_experiences': total_count,
                'regret_stats': {
                    'min': regret_stats[0],
                    'max': regret_stats[1], 
                    'avg': regret_stats[2],
                    'experiences_with_regret': regret_stats[3]
                },
                'time_range': {
                    'start': datetime.fromtimestamp(time_range[0]) if time_range[0] else None,
                    'end': datetime.fromtimestamp(time_range[1]) if time_range[1] else None
                },
                'user_distribution': dict(user_distribution)
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ 데이터베이스 분석 실패: {e}")
            return {}
    
    def get_pruning_candidates(self, top_k_percent: float, 
                             sort_by: str = "regret_score") -> List[str]:
        """정리 대상 experience_id 목록 반환"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if sort_by == "regret_score":
                # 후회 점수 기준 상위 K% 선택
                cursor.execute("""
                    SELECT experience_id, actual_regret 
                    FROM ethics_experiences 
                    ORDER BY actual_regret DESC
                """)
            elif sort_by == "timestamp":
                # 시간 기준 최신 K% 선택
                cursor.execute("""
                    SELECT experience_id, timestamp 
                    FROM ethics_experiences 
                    ORDER BY timestamp DESC
                """)
            elif sort_by == "outcome_rating":
                # 결과 평가 기준 상위 K% 선택
                cursor.execute("""
                    SELECT experience_id, outcome_rating 
                    FROM ethics_experiences 
                    ORDER BY outcome_rating DESC
                """)
            else:
                raise ValueError(f"지원하지 않는 정렬 기준: {sort_by}")
            
            all_experiences = cursor.fetchall()
            total_count = len(all_experiences)
            keep_count = max(1, int(total_count * top_k_percent / 100))
            
            # 상위 K%는 유지, 나머지는 삭제 대상
            to_keep = [exp[0] for exp in all_experiences[:keep_count]]
            to_delete = [exp[0] for exp in all_experiences[keep_count:]]
            
            conn.close()
            
            print(f"📊 총 {total_count}개 경험 중 상위 {top_k_percent}% ({keep_count}개) 유지")
            print(f"🗑️ {len(to_delete)}개 경험 삭제 예정")
            
            return to_delete
            
        except Exception as e:
            print(f"❌ 정리 대상 조회 실패: {e}")
            return []
    
    def prune_experiences(self, experience_ids_to_delete: List[str], 
                         dry_run: bool = False) -> Tuple[int, int]:
        """경험 데이터 정리 실행"""
        if not experience_ids_to_delete:
            print("ℹ️ 삭제할 데이터가 없습니다.")
            return 0, 0
        
        if dry_run:
            print(f"🧪 DRY RUN: {len(experience_ids_to_delete)}개 경험을 삭제할 예정입니다.")
            return len(experience_ids_to_delete), 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 삭제 전 개수
            cursor.execute("SELECT COUNT(*) FROM ethics_experiences")
            before_count = cursor.fetchone()[0]
            
            # 배치 삭제 (1000개씩)
            batch_size = 1000
            deleted_count = 0
            
            for i in range(0, len(experience_ids_to_delete), batch_size):
                batch = experience_ids_to_delete[i:i+batch_size]
                placeholders = ','.join(['?' for _ in batch])
                
                cursor.execute(f"""
                    DELETE FROM ethics_experiences 
                    WHERE experience_id IN ({placeholders})
                """, batch)
                
                deleted_count += cursor.rowcount
                print(f"🗑️ 배치 {i//batch_size + 1}: {cursor.rowcount}개 삭제")
            
            # 삭제 후 개수
            cursor.execute("SELECT COUNT(*) FROM ethics_experiences")
            after_count = cursor.fetchone()[0]
            
            # DB 최적화 (VACUUM)
            print("🔧 데이터베이스 최적화 중...")
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            print(f"✅ 정리 완료: {before_count} → {after_count} (총 {deleted_count}개 삭제)")
            return deleted_count, after_count
            
        except Exception as e:
            print(f"❌ 데이터 정리 실패: {e}")
            return 0, 0
    
    def generate_report(self, before_analysis: Dict[str, Any], 
                       after_analysis: Dict[str, Any], 
                       deleted_count: int) -> str:
        """정리 리포트 생성"""
        report = {
            'pruning_timestamp': datetime.now().isoformat(),
            'backup_path': self.backup_path,
            'before': before_analysis,
            'after': after_analysis,
            'deleted_count': deleted_count,
            'space_saved_percent': (
                (before_analysis.get('total_experiences', 0) - 
                 after_analysis.get('total_experiences', 0)) / 
                max(before_analysis.get('total_experiences', 1), 1) * 100
            )
        }
        
        # 리포트 파일 저장
        report_dir = os.path.join(os.path.dirname(self.db_path), "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"pruning_report_{timestamp}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 리포트 저장: {report_path}")
        return report_path


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Experience DB 자동 정리 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python prune_experience_db.py --top_k 20%                    # 상위 20% 후회 점수만 유지
  python prune_experience_db.py --top_k 30% --sort_by timestamp # 최신 30% 데이터만 유지
  python prune_experience_db.py --dry_run --top_k 15%          # 시뮬레이션 모드
        """
    )
    
    parser.add_argument('--db_path', 
                       default='/mnt/c/large_project/linux_red_heart/data/experience_db/ethics_policy.db',
                       help='Experience DB 경로')
    
    parser.add_argument('--top_k', 
                       default='20%',
                       help='유지할 상위 데이터 비율 (예: 20%)')
    
    parser.add_argument('--sort_by',
                       choices=['regret_score', 'timestamp', 'outcome_rating'],
                       default='regret_score',
                       help='정렬 기준 (regret_score: 후회점수, timestamp: 시간순)')
    
    parser.add_argument('--dry_run',
                       action='store_true',
                       help='실제 삭제 없이 시뮬레이션만 실행')
    
    parser.add_argument('--no_backup',
                       action='store_true', 
                       help='백업 생성 건너뛰기')
    
    args = parser.parse_args()
    
    # top_k 파라미터 파싱
    top_k_str = args.top_k.rstrip('%')
    try:
        top_k_percent = float(top_k_str)
        if not 0 < top_k_percent <= 100:
            raise ValueError("0과 100 사이의 값이어야 합니다")
    except ValueError as e:
        print(f"❌ 잘못된 top_k 값: {args.top_k} ({e})")
        return 1
    
    # DB 파일 존재 확인
    if not os.path.exists(args.db_path):
        print(f"❌ DB 파일을 찾을 수 없습니다: {args.db_path}")
        return 1
    
    print("🧹 Experience DB 정리 시작")
    print("=" * 60)
    print(f"DB 경로: {args.db_path}")
    print(f"유지 비율: 상위 {top_k_percent}%")
    print(f"정렬 기준: {args.sort_by}")
    print(f"모드: {'시뮬레이션' if args.dry_run else '실제 정리'}")
    print("")
    
    # 정리기 초기화
    pruner = ExperienceDBPruner(args.db_path)
    
    # 1. 현재 상태 분석
    print("📊 데이터베이스 현재 상태 분석...")
    before_analysis = pruner.analyze_database()
    if not before_analysis:
        return 1
    
    print(f"총 경험 수: {before_analysis['total_experiences']}")
    print(f"후회 점수 평균: {before_analysis['regret_stats']['avg']:.3f}")
    print(f"후회 경험 수: {before_analysis['regret_stats']['experiences_with_regret']}")
    print("")
    
    # 2. 백업 생성 (dry_run이 아닌 경우)
    if not args.dry_run and not args.no_backup:
        pruner.create_backup()
    
    # 3. 정리 대상 선별
    print("🔍 정리 대상 경험 선별...")
    to_delete = pruner.get_pruning_candidates(top_k_percent, args.sort_by)
    
    if not to_delete:
        print("ℹ️ 정리할 데이터가 없습니다.")
        return 0
    
    # 4. 정리 실행
    print("🗑️ 데이터 정리 실행...")
    deleted_count, remaining_count = pruner.prune_experiences(to_delete, args.dry_run)
    
    if not args.dry_run and deleted_count > 0:
        # 5. 정리 후 분석
        print("📊 정리 후 상태 분석...")
        after_analysis = pruner.analyze_database()
        
        # 6. 리포트 생성
        pruner.generate_report(before_analysis, after_analysis, deleted_count)
        
        print("\n🎉 Experience DB 정리 완료!")
        print(f"공간 절약: {before_analysis['total_experiences'] - after_analysis['total_experiences']}개 경험 제거")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())