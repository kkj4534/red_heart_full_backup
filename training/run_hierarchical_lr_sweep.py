#!/usr/bin/env python3
"""
독립적인 계층적 LR 스윕 실행 스크립트
5-5-5-5 전략으로 총 25개 포인트 테스트
각 LR마다 독립적으로 초기 가중치에서 시작
"""

import sys
import os
sys.path.append('/mnt/c/large_project/linux_red_heart')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from datetime import datetime
import time
import json
from pathlib import Path
import numpy as np
from sentence_transformer_singleton import get_sentence_transformer, SentenceTransformerManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training/lr_sweep_results/hierarchical_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# 모듈 임포트
from training.unified_training_final import UnifiedModel, UnifiedTrainingConfig
from training.hierarchical_lr_sweep import HierarchicalLRSweep


class RedHeartDataset(Dataset):
    """실제 Red Heart 데이터셋"""
    def __init__(self, data_list, preprocessed_path=None):
        self.data = data_list
        self.preprocessed_path = preprocessed_path
        self.embedding_manager = None  # 지연 초기화
        self.embeddings_modified = False
        
        # label 매핑
        self.label_to_idx = {
            'AUTHOR': 0,
            'EVERYBODY': 1,
            'INFO': 2,
            'NOBODY': 3,
            'OTHER': 4
        }
        # 감정 매핑
        self.emotion_keys = ['joy', 'anger', 'surprise', 'disgust', 'sadness', 'shame', 'fear']
        
        # 임베딩 상태 확인
        self._check_embeddings()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """실제 데이터 반환"""
        item = self.data[idx]
        text = item.get('text', '') + ' ' + item.get('title', '')
        
        # 임베딩 처리
        if 'embedding' in item:
            text_embedding = torch.tensor(item['embedding'], dtype=torch.float32)
            # 100x768 크기로 조정
            if text_embedding.shape[0] < 100:
                pad_size = 100 - text_embedding.shape[0]
                text_embedding = torch.cat([text_embedding, torch.zeros(pad_size, 768)], dim=0)
            elif text_embedding.shape[0] > 100:
                text_embedding = text_embedding[:100]
        else:
            # 임베딩이 없으면 SentenceTransformer로 생성
            if self.embedding_manager is None:
                try:
                    self.embedding_manager = get_sentence_transformer(
                        'sentence-transformers/all-MiniLM-L6-v2',
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        cache_folder=os.path.expanduser('~/.cache/huggingface/hub')
                    )
                except Exception as e:
                    logger.error(f"❌ SentenceTransformer 로드 실패: {e}")
                    logger.error("LR 스윕에 필수적인 임베딩 모듈 로드 실패")
                    raise RuntimeError(f"SentenceTransformer 필수 모듈 로드 실패: {e}")
            
            if self.embedding_manager:
                try:
                    embedding = self.embedding_manager.encode(text[:512])
                    text_embedding = torch.tensor(embedding, dtype=torch.float32)
                    if text_embedding.dim() == 1:
                        text_embedding = text_embedding.unsqueeze(0)
                    
                    # 384차원을 768차원으로 패딩 (all-MiniLM-L6-v2는 384차원 출력)
                    if text_embedding.shape[-1] == 384:
                        padding = torch.zeros(text_embedding.shape[0], 384, dtype=torch.float32)
                        text_embedding = torch.cat([text_embedding, padding], dim=-1)  # (1, 768)
                    
                    text_embedding = text_embedding.repeat(100, 1)
                    self.data[idx]['embedding'] = text_embedding.numpy().tolist()
                    self.embeddings_modified = True
                except Exception as e:
                    logger.error(f"❌ 임베딩 생성 실패: {e}")
                    raise RuntimeError(f"임베딩 생성 실패: {e}")
            else:
                logger.error("❌ SentenceTransformer 모델이 초기화되지 않았습니다.")
                raise RuntimeError("SentenceTransformer 모델 초기화 실패")
        
        # label 처리
        label_str = item.get('label', 'OTHER')
        label_idx = self.label_to_idx.get(label_str, 4)
        
        # emotions 처리
        emotions = item.get('emotions', {})
        if isinstance(emotions, dict):
            emotion_vector = [emotions.get(key, 0.0) for key in self.emotion_keys]
            emotion_label = torch.argmax(torch.tensor(emotion_vector)).item()
        else:
            emotion_label = 0
        
        # bentham_scores 처리
        bentham_keys = ['intensity', 'duration', 'certainty', 'propinquity',
                        'purity', 'extent', 'fecundity', 'remoteness', 
                        'succession', 'utility']
        bentham_scores = item.get('bentham_scores', {})
        if isinstance(bentham_scores, dict):
            bentham_vector = [bentham_scores.get(key, 0.5) for key in bentham_keys]
        else:
            bentham_vector = [0.5] * 10
        
        return {
            'input': text_embedding,
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
            'bentham_label': torch.tensor(bentham_vector, dtype=torch.float),
            'regret_label': torch.tensor(item.get('regret_factor', 0.0), dtype=torch.float),
            'surd_label': torch.tensor(label_idx, dtype=torch.long)
        }
    
    def _check_embeddings(self):
        """임베딩 상태 확인"""
        total_items = len(self.data)
        items_with_embedding = sum(1 for item in self.data if 'embedding' in item)
        items_without_embedding = total_items - items_with_embedding
        
        logger.info(f"📊 임베딩 상태:")
        logger.info(f"  - 전체 데이터: {total_items}개")
        logger.info(f"  - 임베딩 있음: {items_with_embedding}개 ({items_with_embedding/total_items*100:.1f}%)")
        logger.info(f"  - 임베딩 없음: {items_without_embedding}개 ({items_without_embedding/total_items*100:.1f}%)")
        
        if items_without_embedding > 0:
            logger.warning(f"⚠️ {items_without_embedding}개 항목에 임베딩이 없습니다. 자동 생성됩니다.")
    
    def save_embeddings(self):
        """생성된 임베딩을 파일에 저장"""
        if not self.embeddings_modified:
            return
        
        if self.preprocessed_path:
            embedded_path = Path(str(self.preprocessed_path).replace('.json', '.embedded.json'))
            try:
                with open(embedded_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ 임베딩이 저장되었습니다: {embedded_path}")
                self.embeddings_modified = False
            except Exception as e:
                logger.error(f"임베딩 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    
    logger.info("=" * 80)
    logger.info("🚀 독립적인 Hierarchical Learning Rate Sweep 시작")
    logger.info("=" * 80)
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # 초기 GPU 메모리 정리
        torch.cuda.empty_cache()
    
    # 모델 설정
    config = UnifiedTrainingConfig()
    config.d_model = 896
    config.num_heads = 16
    config.num_layers = 6
    config.d_ff = 3584
    config.dropout = 0.1
    config.vocab_size = 50000
    config.max_length = 512
    config.micro_batch_size = 2
    
    logger.info("\n📊 모델 설정:")
    logger.info(f"  - d_model: {config.d_model}")
    logger.info(f"  - num_heads: {config.num_heads}")
    logger.info(f"  - num_layers: {config.num_layers}")
    logger.info(f"  - dropout: {config.dropout}")
    
    # 모델 생성
    logger.info("\n🔧 모델 생성 중...")
    model = UnifiedModel(config, device=device).to(device)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  - 총 파라미터: {total_params/1e6:.1f}M")
    logger.info(f"  - 학습 가능 파라미터: {trainable_params/1e6:.1f}M")
    
    # 초기 가중치 저장 (각 LR 테스트마다 이 가중치로 리셋)
    initial_state = model.state_dict()
    torch.save(initial_state, 'training/lr_sweep_results/initial_weights.pth')
    logger.info("  - 초기 가중치 저장 완료")
    
    # 데이터 로더 생성
    logger.info("\n📁 데이터 로더 생성 중...")
    
    # 실제 데이터 로드
    preprocessed_path = Path("claude_api_preprocessing/claude_preprocessed_complete.json")
    
    if not preprocessed_path.exists():
        preprocessed_path = Path("for_learn_dataset/claude_preprocessed_complete.json")
        if not preprocessed_path.exists():
            logger.error(f"전처리된 데이터를 찾을 수 없습니다: {preprocessed_path}")
            raise FileNotFoundError(f"전처리된 데이터 파일이 없습니다")
    
    # 임베딩이 포함된 파일이 있는지 먼저 확인
    embedded_path = Path(str(preprocessed_path).replace('.json', '.embedded.json'))
    if embedded_path.exists():
        logger.info(f"🎯 임베딩 파일 발견: {embedded_path}")
        with open(embedded_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        logger.info(f"📂 기본 데이터 로드: {preprocessed_path}")
        with open(preprocessed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # 전체 데이터 임베딩 생성 (나중에 학습용으로 일부만 사용)
    total_data_count = len(data)
    logger.info(f"  - 전체 데이터: {total_data_count}개")
    
    # 학습/검증 데이터 분할 (90:10) - 임베딩은 전체에 대해 생성
    val_size = int(len(data) * 0.1)
    all_train_data = data[val_size:]  # 전체 학습 데이터
    all_val_data = data[:val_size]    # 전체 검증 데이터
    
    # LR 스윕용으로 일부만 샘플링 (임베딩은 전체에 대해 이미 생성됨)
    train_data = all_train_data[:1000]  # LR 스윕용 학습 데이터
    val_data = all_val_data[:100]       # LR 스윕용 검증 데이터
    
    logger.info(f"  - 전체 데이터: {len(data)}개")
    logger.info(f"  - LR 스윕용 학습 데이터: {len(train_data)}개")
    logger.info(f"  - LR 스윕용 검증 데이터: {len(val_data)}개")
    
    # 전체 데이터셋 생성 (임베딩 생성용)
    logger.info("\n🔄 전체 데이터 임베딩 생성 중...")
    full_dataset = RedHeartDataset(data, preprocessed_path)
    
    # 임베딩이 없는 데이터 확인 및 생성
    missing_embeddings = 0
    for idx, item in enumerate(data):
        if 'embedding' not in item or item.get('embedding') is None:
            missing_embeddings += 1
    
    if missing_embeddings > 0:
        logger.info(f"  - 임베딩 생성 필요: {missing_embeddings}개 / {len(data)}개")
        logger.info("  - 배치 처리 시작 (30개씩, 각 배치 타임아웃 13분)")
        
        # 진행 상황 로그 파일
        progress_log_path = Path(f'training/lr_sweep_results/embedding_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        batch_size = 30  # 30개씩 배치 처리
        total_batches = (len(full_dataset) + batch_size - 1) // batch_size
        successful_count = 0
        failed_count = 0
        total_retry_count = 0  # 전체 프로세스의 누적 재시도 횟수
        max_total_retries = 3  # 최대 허용 재시도 횟수
        total_skip_count = 0  # 전체 스킵 횟수
        max_total_skips = 5  # 최대 허용 스킵 횟수
        
        # 배치 단위로 처리
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(full_dataset))
            batch_items = batch_end - batch_start
            
            # 진행률 계산
            progress = (batch_start / len(full_dataset)) * 100
            eta_batches = total_batches - batch_idx
            eta_minutes = eta_batches * 0.5  # 배치당 평균 30초 예상
            
            logger.info(f"\n  📦 배치 {batch_idx + 1}/{total_batches} 처리 중...")
            logger.info(f"    - 범위: {batch_start} ~ {batch_end-1} ({batch_items}개)")
            logger.info(f"    - 진행률: {progress:.1f}%")
            logger.info(f"    - 예상 남은 시간: {eta_minutes:.1f}분")
            
            # 진행 상황 로그 파일에 기록
            with open(progress_log_path, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] 배치 {batch_idx + 1}/{total_batches}: {batch_start}-{batch_end-1} (진행률: {progress:.1f}%)\n")
            
            batch_success = 0
            batch_fail = 0
            batch_start_time = time.time()
            
            # 현재 배치의 아이템들 처리
            for idx in range(batch_start, batch_end):
                success = False
                for retry in range(2):  # 최대 2번 시도 (원본 + 재시도 1번)
                    try:
                        _ = full_dataset[idx]  # 이 과정에서 임베딩이 생성됨
                        batch_success += 1
                        successful_count += 1
                        success = True
                        break
                    except Exception as e:
                        logger.error(f"    ❌ 임베딩 생성 실패 (idx={idx}, 시도 {retry+1}/2): {e}")
                        
                        if retry == 0:  # 첫 번째 실패 시 서버 재시작
                            # 누적 재시도 횟수 확인
                            if total_retry_count >= max_total_retries:
                                logger.error(f"\n🔴 누적 재시도 횟수 {total_retry_count}회 초과. 프로세스 종료.")
                                logger.error(f"   - 성공: {successful_count}개")
                                logger.error(f"   - 실패: {failed_count}개")
                                with open(progress_log_path, 'a') as f:
                                    f.write(f"[{datetime.now().isoformat()}] 누적 재시도 초과\n")
                                    f.write(f"최종 상태 - 성공: {successful_count}, 실패: {failed_count}\n")
                                raise RuntimeError(f"누적 재시도 {total_retry_count}회 초과로 종료")
                            
                            logger.info(f"    🔄 서버 재시작 시도 중... (누적 {total_retry_count+1}/{max_total_retries})")
                            total_retry_count += 1  # 시도 자체를 카운트
                            
                            try:
                                # SentenceTransformer 서버 재시작
                                manager = SentenceTransformerManager()
                                manager.restart_server(
                                    'sentence-transformers/all-MiniLM-L6-v2',
                                    device='cuda' if torch.cuda.is_available() else 'cpu'
                                )
                                
                                # embedding_manager 재초기화
                                full_dataset.embedding_manager = None
                                logger.info(f"    ✅ 서버 재시작 완료, 재시도 중...")
                                time.sleep(2)  # 서버 안정화 대기
                            except Exception as restart_error:
                                logger.error(f"    ❌ 서버 재시작 실패: {restart_error}")
                                # break 제거 - 두 번째 시도 진행
                        else:  # 두 번째 실패 시
                            batch_fail += 1
                            failed_count += 1
                            logger.error(f"    ❌❌ 재시도 후에도 실패 (idx={idx})")
                            
                            # 로그 파일에 기록
                            with open(progress_log_path, 'a') as f:
                                f.write(f"[{datetime.now().isoformat()}] 최종 실패! idx={idx}, 에러: {e}\n")
                                f.write(f"현재 상태 - 성공: {successful_count}, 실패: {failed_count}\n")
                            
                            break
                
                if not success:
                    # 스킵 횟수 확인
                    total_skip_count += 1
                    if total_skip_count >= max_total_skips:
                        logger.error(f"\n🔴 누적 스킵 횟수 {total_skip_count}개 초과. 프로세스 종료.")
                        logger.error(f"   - 성공: {successful_count}개")
                        logger.error(f"   - 실패: {failed_count}개")
                        logger.error(f"   - 스킵: {total_skip_count}개")
                        with open(progress_log_path, 'a') as f:
                            f.write(f"[{datetime.now().isoformat()}] 누적 스킵 초과\n")
                            f.write(f"최종 상태 - 성공: {successful_count}, 실패: {failed_count}, 스킵: {total_skip_count}\n")
                        raise RuntimeError(f"누적 스킵 {total_skip_count}개 초과로 종료")
                    
                    # 실패한 항목 스킵하고 계속 진행
                    logger.warning(f"    ⚠️ 인덱스 {idx} 스킵하고 계속 진행 (누적 스킵: {total_skip_count}/{max_total_skips})")
            
            batch_elapsed = time.time() - batch_start_time
            logger.info(f"    ✅ 배치 완료: 성공 {batch_success}개 (소요시간: {batch_elapsed:.1f}초)")
            
            # 로그 파일에 배치 완료 기록
            with open(progress_log_path, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] 배치 {batch_idx + 1} 완료: 성공 {batch_success}개, 시간 {batch_elapsed:.1f}초\n")
            
            # 주기적으로 임베딩 저장 (10배치마다)
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  💾 중간 저장 중... (배치 {batch_idx + 1})")
                full_dataset.save_embeddings()
                logger.info(f"     저장 완료")
        
        # 최종 임베딩 저장
        full_dataset.save_embeddings()
        
        # 최종 통계
        logger.info(f"\n  ✅ 전체 데이터 임베딩 생성 완료")
        logger.info(f"     - 성공: {successful_count}개")
        logger.info(f"     - 실패: {failed_count}개")
        logger.info(f"     - 진행 로그: {progress_log_path}")
        
        with open(progress_log_path, 'a') as f:
            f.write(f"\n[{datetime.now().isoformat()}] === 임베딩 생성 완료 ===\n")
            f.write(f"총 성공: {successful_count}개\n")
            f.write(f"총 실패: {failed_count}개\n")
    else:
        logger.info("  ✅ 모든 데이터에 임베딩이 이미 존재")
    
    # LR 스윕용 데이터셋 (이미 생성된 임베딩 사용)
    train_dataset = RedHeartDataset(train_data, preprocessed_path)
    val_dataset = RedHeartDataset(val_data, preprocessed_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"  - Train batches: {len(train_loader)}")
    logger.info(f"  - Val batches: {len(val_loader)}")
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # Hierarchical LR Sweep 설정
    sweep_config = {
        'test_epochs': 3,      # 각 LR당 3 에폭 테스트
        'test_steps': 50,      # 각 에폭당 50 스텝
        'warmup_steps': 10,    # 10 스텝 워밍업
        'output_dir': 'training/lr_sweep_results'
    }
    
    logger.info("\n⚙️ Sweep 설정:")
    logger.info(f"  - 테스트 에폭: {sweep_config['test_epochs']}")
    logger.info(f"  - 테스트 스텝/에폭: {sweep_config['test_steps']}")
    logger.info(f"  - 예상 총 포인트: 25개 (5-5-5-5 전략)")
    logger.info(f"  - 각 LR은 독립적으로 초기 가중치에서 시작")
    
    # Hierarchical LR Sweep 실행
    sweep = HierarchicalLRSweep(**sweep_config)
    
    try:
        # 스윕 실행
        logger.info("\n" + "=" * 80)
        logger.info("📌 각 LR 테스트는 동일한 초기 가중치에서 독립적으로 시작됩니다")
        logger.info("=" * 80)
        
        results = sweep.run_hierarchical_sweep(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # 결과 요약
        logger.info("\n" + "=" * 80)
        logger.info("📊 최종 결과 요약")
        logger.info("=" * 80)
        logger.info(f"✅ 최적 Learning Rate: {results['best_lr']:.1e}")
        logger.info(f"✅ 최적 Validation Loss: {results['best_loss']:.4f}")
        logger.info(f"✅ 총 테스트 포인트: {results['total_points_tested']}")
        logger.info(f"✅ 효율성: {results['efficiency_gain']['vs_grid_search']}")
        logger.info(f"✅ 절약된 포인트: {results['efficiency_gain']['points_saved']}")
        
        # Stage별 최고 성능
        logger.info("\n📈 Stage별 최고 성능:")
        for stage, info in results['stage_results'].items():
            logger.info(f"  - {stage}: LR={info['lr']:.1e}, Loss={info['val_loss']:.4f}, Acc={info['accuracy']:.4f}")
        
        # 최적 LR을 별도 파일로 저장
        optimal_lr_path = Path('training/lr_sweep_results/optimal_lr.json')
        with open(optimal_lr_path, 'w') as f:
            json.dump({
                'optimal_lr': results['best_lr'],
                'optimal_loss': results['best_loss'],
                'timestamp': datetime.now().isoformat(),
                'strategy': '5-5-5-5 Hierarchical',
                'total_points': results['total_points_tested'],
                'stage_results': results['stage_results']
            }, f, indent=2)
        
        logger.info(f"\n💾 최적 LR 저장: {optimal_lr_path}")
        
        # Stage별 파일 확인
        logger.info("\n📁 생성된 파일 확인:")
        lr_sweep_dir = Path('training/lr_sweep_results')
        pattern = f"hierarchical_lr_sweep_stage*_{datetime.now().strftime('%Y%m%d')}*.json"
        stage_files = list(lr_sweep_dir.glob(pattern))
        for f in sorted(stage_files):
            logger.info(f"  - {f.name}")
        
        pattern = f"hierarchical_lr_sweep_stage*_{datetime.now().strftime('%Y%m%d')}*.png"
        stage_plots = list(lr_sweep_dir.glob(pattern))
        for f in sorted(stage_plots):
            logger.info(f"  - {f.name}")
        
        # 학습 권장사항
        logger.info("\n🎯 다음 단계:")
        logger.info(f"  1. 최적 LR ({results['best_lr']:.1e})로 본격 학습 시작")
        logger.info(f"  2. unified_training_final.py의 base_lr을 {results['best_lr']:.1e}로 설정")
        logger.info(f"  3. 60 에폭 전체 학습 실행")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 스윕 실행 중 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # 생성된 임베딩 저장 (전체 데이터셋)
        if 'full_dataset' in locals() and hasattr(full_dataset, 'save_embeddings'):
            logger.info("\n💾 전체 데이터셋 임베딩 저장 중...")
            full_dataset.save_embeddings()
        
        # GPU 메모리 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info(f"\n🧹 GPU 메모리 정리 완료")
        
        # 초기 가중치 파일 삭제
        initial_weights_path = Path('training/lr_sweep_results/initial_weights.pth')
        if initial_weights_path.exists():
            initial_weights_path.unlink()
            logger.info("  - 임시 초기 가중치 파일 삭제")


if __name__ == "__main__":
    results = main()
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("🎉 Hierarchical LR Sweep 완료!")
    print(f"   최적 LR: {results['best_lr']:.1e}")
    print(f"   최적 Loss: {results['best_loss']:.4f}")
    print(f"   각 Stage별 JSON/PNG 파일이 저장되었습니다")
    print("=" * 80)