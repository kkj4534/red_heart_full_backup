#!/usr/bin/env python3
"""
손상된 임베딩 JSON 파일 복구 스크립트
절전모드로 인해 불완전하게 저장된 JSON 파일을 복구
"""

import json
import sys
import os
from pathlib import Path

def find_last_complete_object(file_path, chunk_size=1024*1024):
    """파일에서 마지막 완전한 JSON 객체 위치 찾기"""
    
    file_size = os.path.getsize(file_path)
    print(f"파일 크기: {file_size:,} bytes ({file_size/1024/1024/1024:.2f} GB)")
    
    # 파일 끝에서부터 역방향으로 검색
    with open(file_path, 'rb') as f:
        # 마지막 10MB 정도만 검색 (충분히 큰 범위)
        search_size = min(10 * 1024 * 1024, file_size)
        f.seek(file_size - search_size)
        tail_data = f.read()
        
        # 텍스트로 디코딩
        tail_text = tail_data.decode('utf-8', errors='ignore')
        
        # 마지막 완전한 객체 찾기 ("},")
        last_complete = tail_text.rfind('},')
        
        if last_complete == -1:
            print("완전한 객체를 찾을 수 없습니다.")
            return None
            
        # 실제 파일 위치 계산
        actual_position = file_size - search_size + last_complete + 1  # '}' 다음 위치
        
        print(f"마지막 완전한 객체 위치: {actual_position:,} bytes")
        return actual_position

def recover_json_file(corrupted_path, output_path):
    """손상된 JSON 파일 복구"""
    
    print(f"복구 시작: {corrupted_path}")
    
    # 마지막 완전한 객체 위치 찾기
    last_pos = find_last_complete_object(corrupted_path)
    
    if last_pos is None:
        print("복구 실패: 완전한 객체를 찾을 수 없습니다.")
        return False
    
    # 복구된 데이터 저장
    print(f"복구된 파일 생성 중: {output_path}")
    
    with open(corrupted_path, 'rb') as src:
        with open(output_path, 'wb') as dst:
            # 처음부터 마지막 완전한 객체까지 복사
            remaining = last_pos
            chunk_size = 1024 * 1024 * 10  # 10MB 청크
            
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = src.read(to_read)
                if not data:
                    break
                dst.write(data)
                remaining -= len(data)
                
                # 진행 상황 표시
                progress = (last_pos - remaining) / last_pos * 100
                sys.stdout.write(f"\r진행률: {progress:.1f}%")
                sys.stdout.flush()
            
            # JSON 배열 닫기
            dst.write(b']')
    
    print(f"\n복구 완료: {output_path}")
    
    # 복구된 파일 검증
    print("\n복구된 파일 검증 중...")
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
            
        total_items = len(data)
        embedded_count = sum(1 for item in data if item.get('embedding') is not None)
        
        # 마지막 임베딩된 인덱스 찾기
        last_embedded_idx = -1
        for i in range(len(data)-1, -1, -1):
            if data[i].get('embedding') is not None:
                last_embedded_idx = i
                break
        
        print(f"✅ 복구 성공!")
        print(f"  - 전체 데이터: {total_items:,}개")
        print(f"  - 임베딩 완료: {embedded_count:,}개")
        print(f"  - 마지막 임베딩 인덱스: {last_embedded_idx}")
        print(f"  - 진행률: {embedded_count/total_items*100:.2f}%")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ 검증 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    # 백업된 손상 파일 직접 지정
    backup_dir = Path("/mnt/c/large_project/linux_red_heart/claude_api_preprocessing")
    
    # 14GB 파일 사용
    corrupted_file = Path("/mnt/c/large_project/linux_red_heart/claude_api_preprocessing/claude_preprocessed_complete.embedded.corrupted_20250822_102244.json")
    
    # 파일 존재 확인
    import os
    if os.path.exists(str(corrupted_file)):
        print(f"파일 존재 확인: {os.path.getsize(str(corrupted_file))/1024/1024/1024:.2f} GB")
    else:
        print(f"백업된 손상 파일을 찾을 수 없습니다: {corrupted_file}")
        sys.exit(1)
    
    print(f"복구할 파일: {corrupted_file}")
    
    # 복구된 파일 경로
    recovered_file = backup_dir / "claude_preprocessed_complete.embedded.json"
    
    # 복구 실행
    if recover_json_file(corrupted_file, recovered_file):
        print(f"\n✅ 복구 완료: {recovered_file}")
        print("\n이제 중단된 지점부터 임베딩을 이어서 진행할 수 있습니다.")
    else:
        print("\n❌ 복구 실패")
        sys.exit(1)