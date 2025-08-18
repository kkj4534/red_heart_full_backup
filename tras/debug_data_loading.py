"""
데이터 로딩 디버깅
"""
import json
from pathlib import Path

data_dir = Path("/mnt/c/large_project/linux_red_heart/processed_datasets")

# 스크러플 파일 하나만 열어서 구조 확인
scruples_files = list(data_dir.glob("scruples/scruples_batch_*.json"))
print(f"스크러플 파일 개수: {len(scruples_files)}")

if scruples_files:
    first_file = scruples_files[0]
    print(f"첫 번째 파일: {first_file}")
    
    with open(first_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"데이터 키들: {data.keys()}")
    print(f"시나리오 개수: {len(data['scenarios'])}")
    print(f"첫 번째 시나리오 타입: {type(data['scenarios'][0])}")
    print(f"첫 번째 시나리오 구조:")
    first_scenario = data['scenarios'][0]
    if isinstance(first_scenario, dict):
        print(f"  키들: {first_scenario.keys()}")
        print(f"  description 있음: {'description' in first_scenario}")
    else:
        print(f"  값: {first_scenario}")

# 실제 처리 테스트
valid_count = 0
for scenario in data['scenarios'][:10]:  # 처음 10개만
    if isinstance(scenario, dict) and 'description' in scenario:
        valid_count += 1
        print(f"유효한 시나리오: {scenario['id']}")
    else:
        print(f"무효한 시나리오: {type(scenario)}")

print(f"유효한 시나리오: {valid_count}/10")