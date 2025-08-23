#!/usr/bin/env python3
"""복구된 임베딩 파일 검증 스크립트"""

import json
import sys
import time

def verify_json_structure(file_path):
    """JSON 파일 구조 검증"""
    
    print("=" * 60)
    print("복구된 임베딩 파일 검증 시작")
    print("=" * 60)
    
    # 1. 파일 시작 부분 확인
    print("\n1. 파일 시작 부분 확인...")
    with open(file_path, 'r') as f:
        first_char = f.read(1)
        if first_char != '[':
            print(f"   ❌ 파일이 '['로 시작하지 않음: '{first_char}'")
            return False
        print("   ✅ 파일이 정상적으로 '['로 시작")
    
    # 2. 파일 끝 부분 확인
    print("\n2. 파일 끝 부분 확인...")
    with open(file_path, 'rb') as f:
        f.seek(-100, 2)  # 파일 끝에서 100바이트 전
        tail = f.read().decode('utf-8', errors='ignore').strip()
        if not tail.endswith(']'):
            print(f"   ❌ 파일이 ']'로 끝나지 않음")
            print(f"   마지막 100자: {tail}")
            return False
        print("   ✅ 파일이 정상적으로 ']'로 종료")
    
    # 3. JSON 파싱 테스트
    print("\n3. JSON 파싱 테스트...")
    start_time = time.time()
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        parse_time = time.time() - start_time
        print(f"   ✅ JSON 파싱 성공 (소요시간: {parse_time:.2f}초)")
        
        # 4. 데이터 구조 분석
        print("\n4. 데이터 구조 분석...")
        print(f"   - 전체 데이터 개수: {len(data):,}개")
        
        # 첫 번째 항목 확인
        if data and len(data) > 0:
            first_item = data[0]
            print(f"   - 첫 번째 항목 키: {list(first_item.keys())}")
            has_embedding = 'embedding' in first_item
            print(f"   - 첫 번째 항목 임베딩 여부: {has_embedding}")
        
        # 임베딩 카운트
        print("\n5. 임베딩 통계 계산 중...")
        embedded_count = 0
        last_embedded_idx = -1
        first_non_embedded_idx = -1
        
        for i, item in enumerate(data):
            if item.get('embedding') is not None:
                embedded_count += 1
                last_embedded_idx = i
            elif first_non_embedded_idx == -1 and i > 0:
                first_non_embedded_idx = i
            
            # 진행 상황 표시 (1000개마다)
            if i % 1000 == 0:
                sys.stdout.write(f"\r   처리 중: {i:,}/{len(data):,}")
                sys.stdout.flush()
        
        print(f"\r   처리 완료: {len(data):,}/{len(data):,}")
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("검증 결과 요약")
        print("=" * 60)
        print(f"✅ 파일 상태: 정상")
        print(f"📊 전체 데이터: {len(data):,}개")
        print(f"✅ 임베딩 완료: {embedded_count:,}개")
        print(f"⏸️  임베딩 대기: {len(data) - embedded_count:,}개")
        print(f"📍 마지막 임베딩 인덱스: {last_embedded_idx}")
        print(f"📍 첫 미완료 인덱스: {first_non_embedded_idx}")
        print(f"📈 진행률: {embedded_count/len(data)*100:.2f}%")
        
        # 다음 시작 지점 계산
        next_start_idx = last_embedded_idx + 1
        next_batch = next_start_idx // 30  # 배치 크기 30
        print(f"\n🔄 다음 시작 정보:")
        print(f"   - 시작 인덱스: {next_start_idx}")
        print(f"   - 시작 배치: {next_batch}/349")
        
        # 임베딩 연속성 확인
        print("\n6. 임베딩 연속성 확인...")
        has_gap = False
        for i in range(min(last_embedded_idx + 1, len(data))):
            if data[i].get('embedding') is None:
                print(f"   ⚠️ 임베딩 갭 발견: 인덱스 {i}")
                has_gap = True
                break
        
        if not has_gap and last_embedded_idx >= 0:
            print(f"   ✅ 0부터 {last_embedded_idx}까지 연속적으로 임베딩됨")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON 파싱 실패: {e}")
        print(f"   오류 위치: 라인 {e.lineno}, 컬럼 {e.colno}")
        return False
    except Exception as e:
        print(f"   ❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    file_path = '/mnt/c/large_project/linux_red_heart/claude_api_preprocessing/claude_preprocessed_complete.embedded.json'
    
    if verify_json_structure(file_path):
        print("\n✅ 파일 검증 완료: 임베딩을 이어서 진행할 수 있습니다.")
        sys.exit(0)
    else:
        print("\n❌ 파일 검증 실패: 파일 복구가 필요합니다.")
        sys.exit(1)