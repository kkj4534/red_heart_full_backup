#!/usr/bin/env python3
"""
Experience DB 구조 분석 스크립트
"""
import sqlite3
import sys
import os

def analyze_experience_db(db_path):
    """Experience DB 구조 분석"""
    print(f"🔍 Experience DB 분석: {db_path}")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print('📋 테이블 목록:')
        for table in tables:
            print(f'  - {table[0]}')
        
        # 각 테이블 구조 확인
        for table in tables:
            table_name = table[0]
            print(f'\n🔍 {table_name} 테이블 구조:')
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                print(f'  {col[1]} ({col[2]})')
            
            # 레코드 수 확인
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f'  📊 레코드 수: {count}')
            
            # 샘플 데이터 (최대 2개)
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 2;")
                samples = cursor.fetchall()
                print(f'  📝 샘플 데이터:')
                for i, sample in enumerate(samples, 1):
                    print(f'    {i}: {sample}')
        
        conn.close()
        print('\n✅ DB 구조 분석 완료')
        return True
        
    except Exception as e:
        print(f'❌ DB 접근 오류: {e}')
        return False

if __name__ == "__main__":
    db_path = "/mnt/c/large_project/linux_red_heart/data/experience_db/ethics_policy.db"
    analyze_experience_db(db_path)