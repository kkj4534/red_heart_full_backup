#!/usr/bin/env python3
"""
메트릭 데이터 구조 분석 및 문제점 파악
"""

import json
import numpy as np
from pathlib import Path

def analyze_metrics_structure():
    """메트릭 구조와 문제점 분석"""
    
    print("=" * 80)
    print("📊 메트릭 데이터 구조 분석")
    print("=" * 80)
    
    # 1. 메트릭 히스토리 로드
    with open('training/checkpoints_final/metrics_history.json', 'r') as f:
        metrics = json.load(f)
    
    print("\n1. 전체 구조:")
    print(f"   - Global epochs: {len(metrics.get('global', []))}")
    print(f"   - Modules: {list(metrics.keys())}")
    
    # 2. 첫 번째와 마지막 에폭 데이터 비교
    if 'global' in metrics and len(metrics['global']) > 0:
        first_epoch = metrics['global'][0]
        last_epoch = metrics['global'][-1]
        
        print("\n2. 첫 에폭 메트릭:")
        first_metrics = first_epoch['metrics']
        print(f"   - train_loss: {first_metrics['train_loss']:.6f}")
        print(f"   - val_loss: {first_metrics['val_loss']:.6f}")
        print(f"   - train_acc: {first_metrics['train_acc']:.6f}")
        print(f"   - val_acc: {first_metrics['val_acc']:.6f}")
        print(f"   - Train-Val Loss 차이: {abs(first_metrics['train_loss'] - first_metrics['val_loss']):.8f}")
        print(f"   - Train-Val Acc 차이: {abs(first_metrics['train_acc'] - first_metrics['val_acc']):.8f}")
        
        print("\n3. 마지막 에폭 메트릭:")
        last_metrics = last_epoch['metrics']
        print(f"   - train_loss: {last_metrics['train_loss']:.6f}")
        print(f"   - val_loss: {last_metrics['val_loss']:.6f}")
        print(f"   - train_acc: {last_metrics['train_acc']:.6f}")
        print(f"   - val_acc: {last_metrics['val_acc']:.6f}")
        print(f"   - Train-Val Loss 차이: {abs(last_metrics['train_loss'] - last_metrics['val_loss']):.8f}")
        print(f"   - Train-Val Acc 차이: {abs(last_metrics['train_acc'] - last_metrics['val_acc']):.8f}")
        
        print("\n4. 모듈별 정확도 분석:")
        print("   첫 에폭:")
        for module in ['emotion', 'bentham', 'regret', 'surd']:
            if f'{module}_acc' in first_metrics:
                print(f"     - {module}_acc: {first_metrics[f'{module}_acc']:.4f}")
        
        print("   마지막 에폭:")
        for module in ['emotion', 'bentham', 'regret', 'surd']:
            if f'{module}_acc' in last_metrics:
                print(f"     - {module}_acc: {last_metrics[f'{module}_acc']:.4f}")
        
        # 5. 문제점 분석
        print("\n5. 발견된 문제점:")
        problems = []
        
        # Train-Val 동일성 체크
        train_val_same = True
        for epoch_data in metrics['global']:
            m = epoch_data['metrics']
            if abs(m['train_loss'] - m['val_loss']) > 1e-8:
                train_val_same = False
                break
        
        if train_val_same:
            problems.append("⚠️ train_loss와 val_loss가 모든 에폭에서 동일 (validation 미실행 추정)")
        
        # backbone_acc 체크
        if 'backbone_acc' in first_metrics and first_metrics['backbone_acc'] == 0:
            all_zero = all(
                epoch['metrics'].get('backbone_acc', 0) == 0 
                for epoch in metrics['global']
            )
            if all_zero:
                problems.append("⚠️ backbone_acc가 항상 0 (계산 로직 문제)")
        
        # accuracy 범위 체크
        for module in ['emotion', 'bentham', 'regret', 'surd']:
            key = f'{module}_acc'
            if key in first_metrics:
                values = [epoch['metrics'][key] for epoch in metrics['global']]
                if max(values) > 1.0:
                    problems.append(f"⚠️ {module}_acc가 1.0 초과 (스케일링 문제)")
                if min(values) < 0:
                    problems.append(f"⚠️ {module}_acc가 음수 (계산 오류)")
        
        for p in problems:
            print(f"   {p}")
        
        # 6. SURD 특이점 분석
        print("\n6. SURD 모듈 분석 (30 에폭 전후):")
        if len(metrics['global']) >= 30:
            epoch_29 = metrics['global'][28]['metrics']
            epoch_30 = metrics['global'][29]['metrics']
            epoch_31 = metrics['global'][30]['metrics'] if len(metrics['global']) > 30 else epoch_30
            
            print(f"   Epoch 29: surd_acc = {epoch_29.get('surd_acc', 0):.4f}")
            print(f"   Epoch 30: surd_acc = {epoch_30.get('surd_acc', 0):.4f}")
            print(f"   Epoch 31: surd_acc = {epoch_31.get('surd_acc', 0):.4f}")
            
            drop = abs(epoch_30.get('surd_acc', 0) - epoch_29.get('surd_acc', 0))
            if drop > 0.05:
                print(f"   ⚠️ 30 에폭에서 급격한 변화 감지: {drop:.4f}")
                print(f"      (dynamic threshold 변경 추정: 0.25 → 0.20)")
    
    # 7. 모듈별 데이터 분석
    print("\n7. 모듈별 데이터 존재 여부:")
    for module in ['emotion_head', 'bentham_head', 'regret_head', 'surd_head']:
        if module in metrics:
            print(f"   - {module}: {len(metrics[module])} epochs")
            if len(metrics[module]) > 0:
                sample = metrics[module][0]
                print(f"     키: {list(sample.get('metrics', {}).keys())[:5]}...")
    
    return metrics

def calculate_real_accuracy(metrics_data):
    """실제 accuracy 재계산"""
    print("\n" + "=" * 80)
    print("📊 실제 Accuracy 재계산")
    print("=" * 80)
    
    # 각 모듈의 실제 accuracy 계산 방법 분석
    for epoch_idx in [0, 24, 49]:  # 첫, 중간, 마지막
        if epoch_idx >= len(metrics_data['global']):
            continue
            
        epoch_data = metrics_data['global'][epoch_idx]
        metrics = epoch_data['metrics']
        
        print(f"\nEpoch {epoch_idx + 1}:")
        
        # 모듈별 loss와 accuracy 관계 분석
        for module in ['emotion', 'bentham', 'regret', 'surd']:
            if f'{module}_loss' in metrics and f'{module}_acc' in metrics:
                loss = metrics[f'{module}_loss']
                acc = metrics[f'{module}_acc']
                
                # Loss 기반 accuracy 추정 (1 - loss 방식)
                estimated_acc_v1 = 1.0 - loss if loss < 1.0 else 0.0
                
                # Loss 기반 accuracy 추정 (exp(-loss) 방식)
                estimated_acc_v2 = np.exp(-loss)
                
                print(f"  {module}:")
                print(f"    - loss: {loss:.6f}")
                print(f"    - recorded_acc: {acc:.6f}")
                print(f"    - estimated_acc (1-loss): {estimated_acc_v1:.6f}")
                print(f"    - estimated_acc (exp): {estimated_acc_v2:.6f}")
                
                # 차이 분석
                if abs(acc - estimated_acc_v1) < 0.01:
                    print(f"    ✓ 1-loss 방식과 일치")
                elif abs(acc - estimated_acc_v2) < 0.01:
                    print(f"    ✓ exp(-loss) 방식과 일치")
                elif acc > 0.9 and loss < 0.1:
                    print(f"    ✓ 높은 정확도, 낮은 loss (정상)")
                else:
                    print(f"    ⚠️ 특이한 관계")

if __name__ == "__main__":
    metrics = analyze_metrics_structure()
    calculate_real_accuracy(metrics)
    
    print("\n" + "=" * 80)
    print("✅ 분석 완료")
    print("=" * 80)