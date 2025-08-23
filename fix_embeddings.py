#!/usr/bin/env python3
"""손상된 임베딩 JSON 파일 정확한 복구"""

import os
import sys
import re

def fix_json_file(input_path, output_path):
    """JSON 파일의 마지막 완전한 객체까지만 추출하여 저장"""
    
    file_size = os.path.getsize(input_path)
    print(f"파일 크기: {file_size:,} bytes ({file_size/1024/1024/1024:.2f} GB)")
    
    # 파일 끝부분 읽기 (마지막 100MB)
    search_size = min(100 * 1024 * 1024, file_size)
    
    with open(input_path, 'rb') as f:
        # 끝부분 읽기
        f.seek(file_size - search_size)
        tail_data = f.read()
        
    tail_text = tail_data.decode('utf-8', errors='ignore')
    
    # 마지막 완전한 객체 찾기
    # "embedding": [...] 패턴 뒤에 오는 } 찾기
    last_object_end = -1
    
    # 방법 1: 마지막 '},' 패턴 찾기
    matches = list(re.finditer(r'\}\s*,', tail_text))
    if matches:
        last_match = matches[-1]
        last_object_end = file_size - search_size + last_match.end() - 1  # 콤마 제외
        print(f"마지막 완전한 객체 위치 (방법1): {last_object_end:,} bytes")
    
    # 방법 2: 더 정확한 패턴 - embedding 배열이 끝나고 객체가 닫히는 위치
    pattern = r'"embedding"\s*:\s*\[[^\]]*\]\s*\}'
    matches2 = list(re.finditer(pattern, tail_text))
    if matches2:
        last_match2 = matches2[-1]
        last_object_end2 = file_size - search_size + last_match2.end()
        print(f"마지막 완전한 객체 위치 (방법2): {last_object_end2:,} bytes")
        
        # 더 큰 값 사용 (더 안전)
        if last_object_end2 > last_object_end:
            last_object_end = last_object_end2
    
    if last_object_end == -1:
        print("완전한 객체를 찾을 수 없습니다.")
        return False
    
    print(f"최종 선택된 위치: {last_object_end:,} bytes")
    
    # 새 파일 생성
    print(f"\n복구 파일 생성 중: {output_path}")
    
    with open(input_path, 'rb') as src:
        with open(output_path, 'wb') as dst:
            # 처음부터 마지막 완전한 객체까지 복사
            remaining = last_object_end
            chunk_size = 10 * 1024 * 1024  # 10MB 청크
            copied = 0
            
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = src.read(to_read)
                if not data:
                    break
                dst.write(data)
                copied += len(data)
                remaining -= len(data)
                
                # 진행 상황
                progress = copied / last_object_end * 100
                sys.stdout.write(f"\r진행률: {progress:.1f}%")
                sys.stdout.flush()
            
            # JSON 배열 닫기
            dst.write(b']')
    
    print(f"\n✅ 복구 완료!")
    
    # 검증
    print("\n파일 끝 부분 확인...")
    with open(output_path, 'rb') as f:
        f.seek(-100, 2)
        tail = f.read().decode('utf-8', errors='ignore')
        print(f"마지막 100자:\n{tail}")
        
        if tail.strip().endswith(']'):
            print("✅ 파일이 정상적으로 ']'로 끝남")
            return True
        else:
            print("❌ 파일 종료 문자 확인 실패")
            return False

if __name__ == "__main__":
    # 백업 파일
    corrupted_file = "/mnt/c/large_project/linux_red_heart/claude_api_preprocessing/claude_preprocessed_complete.embedded.corrupted_20250822_102244.json"
    recovered_file = "/mnt/c/large_project/linux_red_heart/claude_api_preprocessing/claude_preprocessed_complete.embedded.json"
    
    if not os.path.exists(corrupted_file):
        print(f"백업 파일을 찾을 수 없습니다: {corrupted_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("임베딩 파일 정확한 복구 시작")
    print("=" * 60)
    
    if fix_json_file(corrupted_file, recovered_file):
        print("\n✅ 파일 복구 성공!")
    else:
        print("\n❌ 파일 복구 실패")
        sys.exit(1)