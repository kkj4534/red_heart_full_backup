"""
특성 추출 디버깅
"""
import json
from pathlib import Path

data_dir = Path("/mnt/c/large_project/linux_red_heart/processed_datasets")

def _extract_bentham_features(scenario):
    """시나리오에서 벤담 특성 추출"""
    try:
        print(f"시나리오 타입: {type(scenario)}")
        print(f"시나리오 내용: {scenario}")
        
        text = scenario.get('description', '')
        if not text:
            print("텍스트 없음")
            return None
        
        print(f"텍스트 길이: {len(text)}")
        
        # 기본 특성들만
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'complexity': scenario.get('complexity', 0.5),
        }
        
        print(f"추출된 특성: {features}")
        return features
        
    except Exception as e:
        print(f"특성 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# 스크러플 파일 하나만 테스트
scruples_files = list(data_dir.glob("scruples/scruples_batch_*.json"))
first_file = scruples_files[0]

print(f"파일: {first_file}")

with open(first_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"시나리오 개수: {len(data['scenarios'])}")

training_data = []
for i, scenario in enumerate(data['scenarios'][:5]):  # 처음 5개만
    print(f"\n=== 시나리오 {i+1} ===")
    if isinstance(scenario, dict) and 'description' in scenario:
        features = _extract_bentham_features(scenario)
        if features:
            training_data.append(features)
            print("✅ 성공적으로 추가됨")
        else:
            print("❌ 특성 추출 실패")
    else:
        print(f"❌ 잘못된 형식: {type(scenario)}")

print(f"\n최종 훈련 데이터 개수: {len(training_data)}")