"""
테스트 데이터 생성기
윤리적 딜레마 시나리오 10개 생성
"""

import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from config import DATA_DIR

def generate_test_scenarios():
    """10개의 윤리적 딜레마 테스트 시나리오 생성"""
    
    scenarios = [
        # 1. 노숙자 도움 딜레마 (기존)
        {
            "title": "노숙자 돕기 딜레마",
            "description": "길을 걷다가 한 노숙자가 도움을 청하고 있습니다. 당신의 지갑에는 중요한 약속을 위해 사용할 예정이었던 돈이 있습니다.",
            "context": {
                "location": "도시 거리",
                "time": "오후",
                "weather": "춥고 비가 오는 날씨",
                "people_involved": ["나", "노숙자"]
            },
            "variables": {
                "money_amount": 30000,
                "urgency_of_appointment": 0.8,
                "homeless_person_condition": 0.7
            },
            "options": [
                {"id": "give_money", "text": "노숙자에게 돈을 준다", "affected_count": 2, "duration_seconds": 3600},
                {"id": "give_food", "text": "근처 가게에서 음식을 사서 준다", "affected_count": 2, "duration_seconds": 1800},
                {"id": "ignore", "text": "무시하고 지나간다", "affected_count": 1, "duration_seconds": 300},
                {"id": "call_shelter", "text": "노숙자 쉼터에 연락한다", "affected_count": 2, "duration_seconds": 1200}
            ],
            "outcomes": {
                "give_money": {"hedonic": 0.7, "emotion": "JOY"},
                "give_food": {"hedonic": 0.8, "emotion": "JOY"},
                "ignore": {"hedonic": -0.4, "emotion": "SADNESS"},
                "call_shelter": {"hedonic": 0.6, "emotion": "TRUST"}
            }
        },
        
        # 2. 자율주행차 윤리 딜레마
        {
            "title": "자율주행차 사고 회피 딜레마",
            "description": "자율주행차가 사고를 피할 수 없는 상황에서 보행자 1명과 승객 2명 중 누구를 보호할지 결정해야 합니다.",
            "context": {
                "location": "도심 교차로",
                "time": "저녁",
                "weather": "우천시",
                "people_involved": ["승객들", "보행자", "시스템"]
            },
            "variables": {
                "vehicle_speed": 0.6,
                "pedestrian_age": 0.3,  # 젊음
                "passenger_age": 0.7,   # 고령
                "traffic_density": 0.8
            },
            "options": [
                {"id": "protect_pedestrian", "text": "보행자를 보호한다", "affected_count": 3, "duration_seconds": 30},
                {"id": "protect_passengers", "text": "승객들을 보호한다", "affected_count": 3, "duration_seconds": 30},
                {"id": "minimize_harm", "text": "최소 피해를 목표로 한다", "affected_count": 3, "duration_seconds": 30},
                {"id": "random_choice", "text": "랜덤하게 결정한다", "affected_count": 3, "duration_seconds": 30}
            ],
            "outcomes": {
                "protect_pedestrian": {"hedonic": 0.6, "emotion": "TRUST"},
                "protect_passengers": {"hedonic": 0.4, "emotion": "GUILT"},
                "minimize_harm": {"hedonic": 0.8, "emotion": "RELIEF"},
                "random_choice": {"hedonic": 0.2, "emotion": "ANXIETY"}
            }
        },
        
        # 3. 의료 자원 배분 딜레마
        {
            "title": "의료 자원 배분 딜레마",
            "description": "병원에 인공호흡기가 1대밖에 없는데 2명의 환자가 모두 필요한 상황입니다. 한 명은 젊지만 회복 가능성이 낮고, 다른 한 명은 고령이지만 회복 가능성이 높습니다.",
            "context": {
                "location": "응급실",
                "time": "밤",
                "weather": "실내",
                "people_involved": ["젊은 환자", "고령 환자", "의료진"]
            },
            "variables": {
                "young_patient_recovery": 0.3,
                "old_patient_recovery": 0.8,
                "resource_scarcity": 0.9,
                "family_pressure": 0.7
            },
            "options": [
                {"id": "prioritize_young", "text": "젊은 환자를 우선한다", "affected_count": 4, "duration_seconds": 86400},
                {"id": "prioritize_recoverable", "text": "회복 가능성이 높은 환자를 우선한다", "affected_count": 4, "duration_seconds": 86400},
                {"id": "first_come_first_serve", "text": "먼저 온 순서대로 결정한다", "affected_count": 4, "duration_seconds": 86400},
                {"id": "committee_decision", "text": "의료 위원회에서 결정한다", "affected_count": 4, "duration_seconds": 7200}
            ],
            "outcomes": {
                "prioritize_young": {"hedonic": 0.3, "emotion": "GUILT"},
                "prioritize_recoverable": {"hedonic": 0.8, "emotion": "TRUST"},
                "first_come_first_serve": {"hedonic": 0.6, "emotion": "RELIEF"},
                "committee_decision": {"hedonic": 0.7, "emotion": "TRUST"}
            }
        },
        
        # 4. 환경 vs 경제 딜레마
        {
            "title": "환경보호 vs 경제발전 딜레마",
            "description": "지역의 마지막 남은 숲을 보존할지, 일자리 창출을 위해 공장을 건설할지 결정해야 합니다.",
            "context": {
                "location": "시의회",
                "time": "오전",
                "weather": "맑음",
                "people_involved": ["시민들", "환경단체", "기업", "정부"]
            },
            "variables": {
                "unemployment_rate": 0.8,
                "environmental_damage": 0.9,
                "economic_benefit": 0.7,
                "public_opinion": 0.5
            },
            "options": [
                {"id": "preserve_forest", "text": "숲을 보존한다", "affected_count": 10000, "duration_seconds": 31536000},
                {"id": "build_factory", "text": "공장을 건설한다", "affected_count": 10000, "duration_seconds": 31536000},
                {"id": "compromise_solution", "text": "절충안을 찾는다", "affected_count": 10000, "duration_seconds": 31536000},
                {"id": "public_referendum", "text": "주민투표로 결정한다", "affected_count": 10000, "duration_seconds": 2592000}
            ],
            "outcomes": {
                "preserve_forest": {"hedonic": 0.7, "emotion": "JOY"},
                "build_factory": {"hedonic": 0.5, "emotion": "ANXIETY"},
                "compromise_solution": {"hedonic": 0.6, "emotion": "TRUST"},
                "public_referendum": {"hedonic": 0.8, "emotion": "TRUST"}
            }
        },
        
        # 5. AI 개인정보 활용 딜레마
        {
            "title": "AI 개인정보 활용 딜레마",
            "description": "AI 의료 진단 시스템의 정확도를 높이기 위해 환자들의 민감한 개인정보를 수집해야 합니다.",
            "context": {
                "location": "연구소",
                "time": "오후",
                "weather": "실내",
                "people_involved": ["연구진", "환자들", "병원", "정부"]
            },
            "variables": {
                "diagnostic_accuracy_improvement": 0.9,
                "privacy_violation_level": 0.8,
                "consent_rate": 0.6,
                "potential_lives_saved": 0.9
            },
            "options": [
                {"id": "collect_data", "text": "개인정보를 수집한다", "affected_count": 1000, "duration_seconds": 31536000},
                {"id": "anonymize_data", "text": "익명화된 데이터만 사용한다", "affected_count": 1000, "duration_seconds": 31536000},
                {"id": "refuse_collection", "text": "개인정보 수집을 거부한다", "affected_count": 1000, "duration_seconds": 31536000},
                {"id": "opt_in_only", "text": "동의한 환자만 참여시킨다", "affected_count": 600, "duration_seconds": 31536000}
            ],
            "outcomes": {
                "collect_data": {"hedonic": 0.4, "emotion": "GUILT"},
                "anonymize_data": {"hedonic": 0.7, "emotion": "TRUST"},
                "refuse_collection": {"hedonic": 0.3, "emotion": "REGRET"},
                "opt_in_only": {"hedonic": 0.8, "emotion": "JOY"}
            }
        },
        
        # 6. 교육 기회 배분 딜레마
        {
            "title": "교육 기회 배분 딜레마",
            "description": "우수한 대학 추천서를 써줄 수 있는데, 성적이 우수한 학생과 가정형편이 어려운 학생 중 누구를 선택해야 합니다.",
            "context": {
                "location": "학교",
                "time": "오후",
                "weather": "맑음",
                "people_involved": ["우수 학생", "어려운 학생", "교사"]
            },
            "variables": {
                "student_a_grades": 0.9,
                "student_b_need": 0.9,
                "recommendation_impact": 0.8,
                "fairness_concern": 0.7
            },
            "options": [
                {"id": "choose_merit", "text": "성적이 우수한 학생을 선택한다", "affected_count": 3, "duration_seconds": 31536000},
                {"id": "choose_need", "text": "가정형편이 어려운 학생을 선택한다", "affected_count": 3, "duration_seconds": 31536000},
                {"id": "write_both", "text": "둘 다 추천서를 써준다", "affected_count": 3, "duration_seconds": 31536000},
                {"id": "transparent_criteria", "text": "공개적인 기준을 마련한다", "affected_count": 10, "duration_seconds": 31536000}
            ],
            "outcomes": {
                "choose_merit": {"hedonic": 0.5, "emotion": "GUILT"},
                "choose_need": {"hedonic": 0.7, "emotion": "JOY"},
                "write_both": {"hedonic": 0.9, "emotion": "JOY"},
                "transparent_criteria": {"hedonic": 0.8, "emotion": "TRUST"}
            }
        },
        
        # 7. 기업 내부 고발 딜레마
        {
            "title": "기업 내부 고발 딜레마",
            "description": "회사가 환경오염을 은폐하고 있다는 사실을 발견했습니다. 고발하면 회사와 동료들에게 피해가 가지만 환경을 보호할 수 있습니다.",
            "context": {
                "location": "회사",
                "time": "야간",
                "weather": "실내",
                "people_involved": ["나", "동료들", "회사", "지역주민"]
            },
            "variables": {
                "environmental_damage": 0.8,
                "job_security_risk": 0.9,
                "legal_protection": 0.3,
                "moral_obligation": 0.9
            },
            "options": [
                {"id": "whistleblow", "text": "내부 고발한다", "affected_count": 1000, "duration_seconds": 31536000},
                {"id": "internal_report", "text": "내부적으로만 보고한다", "affected_count": 100, "duration_seconds": 2592000},
                {"id": "stay_silent", "text": "침묵을 유지한다", "affected_count": 1000, "duration_seconds": 31536000},
                {"id": "collect_evidence", "text": "더 많은 증거를 수집한다", "affected_count": 1000, "duration_seconds": 7776000}
            ],
            "outcomes": {
                "whistleblow": {"hedonic": 0.6, "emotion": "ANXIETY"},
                "internal_report": {"hedonic": 0.4, "emotion": "ANXIETY"},
                "stay_silent": {"hedonic": 0.1, "emotion": "GUILT"},
                "collect_evidence": {"hedonic": 0.8, "emotion": "TRUST"}
            }
        },
        
        # 8. 소셜미디어 검열 딜레마
        {
            "title": "소셜미디어 검열 딜레마",
            "description": "온라인 플랫폼에서 가짜뉴스와 표현의 자유 사이에서 어떤 기준으로 콘텐츠를 관리할지 결정해야 합니다.",
            "context": {
                "location": "온라인 플랫폼",
                "time": "실시간",
                "weather": "가상공간",
                "people_involved": ["사용자들", "플랫폼", "정부", "언론"]
            },
            "variables": {
                "misinformation_harm": 0.8,
                "censorship_concern": 0.9,
                "user_autonomy": 0.8,
                "platform_responsibility": 0.7
            },
            "options": [
                {"id": "strict_moderation", "text": "엄격한 검열을 시행한다", "affected_count": 100000, "duration_seconds": 31536000},
                {"id": "user_flagging", "text": "사용자 신고 시스템을 운영한다", "affected_count": 100000, "duration_seconds": 31536000},
                {"id": "ai_detection", "text": "AI 기반 자동 탐지를 사용한다", "affected_count": 100000, "duration_seconds": 31536000},
                {"id": "transparency_focus", "text": "투명성과 팩트체크에 집중한다", "affected_count": 100000, "duration_seconds": 31536000}
            ],
            "outcomes": {
                "strict_moderation": {"hedonic": 0.3, "emotion": "ANXIETY"},
                "user_flagging": {"hedonic": 0.6, "emotion": "TRUST"},
                "ai_detection": {"hedonic": 0.7, "emotion": "TRUST"},
                "transparency_focus": {"hedonic": 0.8, "emotion": "JOY"}
            }
        },
        
        # 9. 장기이식 우선순위 딜레마
        {
            "title": "장기이식 우선순위 딜레마",
            "description": "심장이식이 필요한 두 환자가 있습니다. 한 명은 과거 생활습관이 좋지 않았고, 다른 한 명은 유전적 질환 환자입니다.",
            "context": {
                "location": "병원",
                "time": "응급상황",
                "weather": "실내",
                "people_involved": ["환자 A", "환자 B", "의료진", "가족들"]
            },
            "variables": {
                "patient_a_lifestyle": 0.3,  # 좋지 않음
                "patient_b_genetic": 0.9,    # 유전적 원인
                "survival_rate_a": 0.7,
                "survival_rate_b": 0.8
            },
            "options": [
                {"id": "prioritize_genetic", "text": "유전적 질환 환자를 우선한다", "affected_count": 6, "duration_seconds": 86400},
                {"id": "prioritize_survival", "text": "생존률이 높은 환자를 우선한다", "affected_count": 6, "duration_seconds": 86400},
                {"id": "first_registered", "text": "먼저 등록한 환자를 우선한다", "affected_count": 6, "duration_seconds": 86400},
                {"id": "committee_review", "text": "윤리위원회에서 검토한다", "affected_count": 6, "duration_seconds": 3600}
            ],
            "outcomes": {
                "prioritize_genetic": {"hedonic": 0.8, "emotion": "TRUST"},
                "prioritize_survival": {"hedonic": 0.6, "emotion": "ANXIETY"},
                "first_registered": {"hedonic": 0.7, "emotion": "TRUST"},
                "committee_review": {"hedonic": 0.9, "emotion": "TRUST"}
            }
        },
        
        # 10. 인공지능 대체 딜레마
        {
            "title": "인공지능 일자리 대체 딜레마",
            "description": "AI 도입으로 효율성은 크게 향상되지만 많은 직원들이 해고될 상황입니다. 회사의 경쟁력과 직원들의 생계 중 무엇을 우선해야 할까요?",
            "context": {
                "location": "회사 이사회",
                "time": "오전",
                "weather": "실내",
                "people_involved": ["경영진", "직원들", "주주들", "고객들"]
            },
            "variables": {
                "efficiency_gain": 0.9,
                "job_displacement": 0.8,
                "competitive_pressure": 0.8,
                "social_responsibility": 0.7
            },
            "options": [
                {"id": "full_automation", "text": "완전 자동화를 진행한다", "affected_count": 200, "duration_seconds": 31536000},
                {"id": "gradual_transition", "text": "점진적 전환을 진행한다", "affected_count": 200, "duration_seconds": 63072000},
                {"id": "human_ai_collaboration", "text": "인간-AI 협업 모델을 구축한다", "affected_count": 200, "duration_seconds": 31536000},
                {"id": "retraining_program", "text": "재교육 프로그램을 운영한다", "affected_count": 200, "duration_seconds": 15552000}
            ],
            "outcomes": {
                "full_automation": {"hedonic": 0.2, "emotion": "GUILT"},
                "gradual_transition": {"hedonic": 0.6, "emotion": "ANXIETY"},
                "human_ai_collaboration": {"hedonic": 0.8, "emotion": "JOY"},
                "retraining_program": {"hedonic": 0.9, "emotion": "JOY"}
            }
        }
    ]
    
    return scenarios

def create_decision_log(scenario_data, chosen_option_id=None):
    """시나리오를 기반으로 의사결정 로그 생성"""
    
    # 랜덤하게 선택할 옵션 결정 (제공되지 않은 경우)
    if chosen_option_id is None:
        import random
        chosen_option_id = random.choice([opt['id'] for opt in scenario_data['options']])
    
    # 기본 IDs 생성
    situation_id = str(uuid.uuid4())
    decision_id = str(uuid.uuid4())
    log_id = str(uuid.uuid4())
    
    # 선택된 옵션 찾기
    chosen_option = next(opt for opt in scenario_data['options'] if opt['id'] == chosen_option_id)
    
    # 결과 데이터
    outcome = scenario_data['outcomes'][chosen_option_id]
    
    # 의사결정 로그 구조 생성
    decision_log = {
        "id": log_id,
        "situation": {
            "id": situation_id,
            "title": scenario_data["title"],
            "description": scenario_data["description"],
            "context": scenario_data["context"],
            "variables": scenario_data["variables"],
            "options": scenario_data["options"],
            "created_at": datetime.now().isoformat(),
            "source": "generated"
        },
        "biosignals": {
            "eeg": {},
            "ecg": {},
            "gsr": {},
            "voice": {},
            "eye_tracking": {},
            "timestamp": datetime.now().isoformat()
        },
        "emotions": {
            "primary_emotion": "NEUTRAL",
            "intensity": "MODERATE",
            "arousal": 0.0,
            "valence": 0.0,
            "secondary_emotions": {},
            "confidence": 0.5,
            "timestamp": datetime.now().isoformat()
        },
        "hedonic_values": {
            "intensity": 0.5,
            "duration": 1.0,
            "certainty": 0.5,
            "propinquity": 0.8,
            "fecundity": 0.3,
            "purity": 0.6,
            "extent": chosen_option["affected_count"] / 100.0,
            "hedonic_total": 0.5
        },
        "decision": {
            "id": decision_id,
            "situation_id": situation_id,
            "choice": chosen_option_id,
            "reasoning": f"'{chosen_option['text']}'를 선택했습니다.",
            "confidence": 0.7,
            "reasoning_log": {
                "options_analyzed": [],
                "activated_experience_ids": [],
                "avg_past_hedonic": None,
                "avg_past_regret": None,
                "confidence_factors": {
                    "hedonic_diff": 0.2,
                    "emotion_pred": 0.6,
                    "past_exp": 0.5,
                    "bentham_certainty": 0.5
                }
            },
            "predicted_outcome": {
                "hedonic_value": 0.5,
                "adjusted_hedonic_value": 0.5,
                "primary_emotion": "NEUTRAL",
                "selected_option": chosen_option_id
            },
            "timestamp": datetime.now().isoformat()
        },
        "actual_outcome": {
            "hedonic_value": outcome["hedonic"],
            "primary_emotion": outcome["emotion"],
            "option_outcomes": {opt_id: scenario_data["outcomes"][opt_id] for opt_id in scenario_data["outcomes"]},
            "description": f"결정의 결과입니다. 선택한 옵션: {chosen_option['text']}"
        },
        "has_regret": abs(outcome["hedonic"] - 0.5) < 0.3,  # 중간 정도 결과면 후회 있음
        "regret_data": {
            "intensity": max(0, 0.8 - outcome["hedonic"]),  # 결과가 나쁠수록 후회 증가
            "threshold": 0.3,
            "was_optimal": outcome["hedonic"] > 0.7,
            "analysis": {
                "prediction_error": abs(0.5 - outcome["hedonic"]),
                "predicted_hedonic": 0.5,
                "actual_hedonic": outcome["hedonic"],
                "hedonic_difference": outcome["hedonic"] - 0.5,
                "better_options": [],
                "was_optimal": outcome["hedonic"] > 0.7
            },
            "timestamp": time.time()
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return decision_log

def generate_all_test_data():
    """모든 테스트 데이터 생성"""
    print("📊 테스트 데이터 생성 시작...")
    
    scenarios = generate_test_scenarios()
    logs_dir = DATA_DIR / 'decision_logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 파일 개수 확인
    existing_files = list(logs_dir.glob("*.json"))
    print(f"📁 기존 파일: {len(existing_files)}개")
    
    generated_count = 0
    for i, scenario in enumerate(scenarios):
        try:
            # 각 시나리오당 1-2개 변형 생성
            for variation in range(2):  # 2개씩 생성
                import random
                chosen_option = random.choice([opt['id'] for opt in scenario['options']])
                
                decision_log = create_decision_log(scenario, chosen_option)
                
                # 파일명 생성
                filename = f"test_scenario_{i+1}_var_{variation+1}_{decision_log['id']}.json"
                filepath = logs_dir / filename
                
                # JSON 파일로 저장
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(decision_log, f, ensure_ascii=False, indent=2)
                
                generated_count += 1
                print(f"✅ 생성됨: {filename}")
                
        except Exception as e:
            print(f"❌ 시나리오 {i+1} 생성 실패: {e}")
    
    print(f"🎯 총 {generated_count}개 테스트 데이터 생성 완료")
    print(f"📂 저장 위치: {logs_dir}")
    
    return generated_count

if __name__ == "__main__":
    generate_all_test_data()