#!/usr/bin/env python3
"""
Sweet Spot Visualization 재생성
- confidence 문제 해결
- SURD 이상 패턴 설명 추가
- 정확한 accuracy 표시
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """데이터 로드"""
    # Sweet spot analysis 데이터
    with open('training/sweet_spot_analysis/sweet_spot_analysis.json', 'r') as f:
        sweet_data = json.load(f)
    
    # 메트릭 히스토리
    with open('training/checkpoints_final/metrics_history.json', 'r') as f:
        metrics_data = json.load(f)
    
    return sweet_data, metrics_data

def extract_accuracies(metrics_data, module_name):
    """메트릭 데이터에서 accuracy 추출"""
    epochs = []
    accuracies = []
    
    for epoch_data in metrics_data['global']:
        epoch = epoch_data['epoch']
        metrics = epoch_data['metrics']
        
        # 모듈별 accuracy 키 매핑
        acc_key_map = {
            'neural_analyzers': 'analyzer_acc',
            'emotion_head': 'emotion_acc',
            'bentham_head': 'bentham_acc',
            'regret_head': 'regret_acc',
            'surd_head': 'surd_acc',
            'backbone': 'backbone_acc',
            'system': 'train_acc'
        }
        
        acc_key = acc_key_map.get(module_name)
        if acc_key and acc_key in metrics:
            epochs.append(epoch)
            accuracies.append(metrics[acc_key])
    
    return epochs, accuracies

def analyze_confidence(sweet_data, module_name):
    """confidence 분석 및 후보 epoch 추출"""
    module_data = sweet_data.get(module_name, {})
    analyses = module_data.get('analyses', {})
    
    # Ensemble voting 결과 분석
    voting = analyses.get('voting', {})
    candidates = voting.get('candidates', {})
    
    # 각 방법이 선택한 epoch들 (int로 변환)
    def safe_int(val, default=-1):
        try:
            return int(val) if val is not None else default
        except:
            return default
    
    selected_epochs = {
        'min_loss': safe_int(candidates.get('min_loss', -1)),
        'plateau': safe_int(analyses.get('plateau', {}).get('center_epoch', -1)),
        'elbow': safe_int(analyses.get('elbow', {}).get('epoch', -1)),
        'mcda': safe_int(analyses.get('mcda', {}).get('best_epoch', -1))
    }
    
    # confidence 계산 (투표 일치도)
    recommendation = module_data.get('recommendation', {})
    confidence = recommendation.get('confidence', 0)
    recommended_epoch = safe_int(recommendation.get('epoch', -1))
    
    return selected_epochs, confidence, recommended_epoch

def plot_module_analysis(ax, module_name, sweet_data, metrics_data, epochs, losses, accuracies):
    """개별 모듈 분석 플롯"""
    
    # 기본 Loss 플롯
    ax_loss = ax
    color = 'tab:blue'
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', color=color, fontsize=10)
    ax_loss.plot(epochs, losses, color=color, alpha=0.7, linewidth=1.5)
    ax_loss.tick_params(axis='y', labelcolor=color)
    ax_loss.grid(True, alpha=0.3)
    
    # Accuracy 플롯 (오른쪽 축)
    ax_acc = ax_loss.twinx()
    color = 'tab:orange'
    ax_acc.set_ylabel('Accuracy', color=color, fontsize=10)
    
    acc_epochs, acc_values = extract_accuracies(metrics_data, module_name)
    if acc_values:
        ax_acc.plot(acc_epochs, acc_values, color=color, alpha=0.7, linewidth=1.5)
        ax_acc.tick_params(axis='y', labelcolor=color)
    
    # 후보 epoch들 분석
    selected_epochs, confidence, recommended_epoch = analyze_confidence(sweet_data, module_name)
    
    # 추천 epoch 표시
    if recommended_epoch > 0 and recommended_epoch <= len(losses):
        ax_loss.axvline(x=recommended_epoch, color='red', linestyle='--', alpha=0.7)
        ax_loss.scatter([recommended_epoch], [losses[recommended_epoch-1]], 
                       color='red', s=100, zorder=5, marker='*')
        
        # Confidence와 후보들 텍스트 표시
        y_pos = max(losses) * 0.95
        ax_loss.text(recommended_epoch, y_pos, 
                    f'Best: E{recommended_epoch}\nConf: {confidence:.1%}',
                    fontsize=8, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 다른 후보 epoch들 표시 (겹치지 않게)
    candidate_colors = {'min_loss': 'green', 'plateau': 'purple', 'elbow': 'cyan', 'mcda': 'brown'}
    y_offset = 0
    for method, epoch in selected_epochs.items():
        if epoch > 0 and epoch <= len(losses) and epoch != recommended_epoch:
            ax_loss.scatter([epoch], [losses[epoch-1]], 
                           color=candidate_colors[method], s=50, alpha=0.6)
            # 텍스트 위치 조정 (겹치지 않게)
            y_text = min(losses) + (max(losses) - min(losses)) * (0.1 + y_offset * 0.05)
            ax_loss.text(epoch, y_text, f'{method[:3]}:{epoch}',
                        fontsize=6, ha='center', color=candidate_colors[method])
            y_offset += 1
    
    # SURD 특별 처리 - 30 에폭 문제 표시
    if module_name == 'surd_head':
        ax_loss.axvline(x=30, color='orange', linestyle=':', alpha=0.8, linewidth=2)
        ax_loss.text(30, max(losses) * 0.8, 
                    'Threshold\nChange\n(0.25→0.20)',
                    fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))
    
    # 타이틀과 범례
    ax_loss.set_title(f'{module_name}', fontsize=11, fontweight='bold')
    
    # MCDA 점수 표시 (있는 경우)
    mcda_data = sweet_data[module_name].get('analyses', {}).get('mcda', {})
    if mcda_data and 'scores' in mcda_data:
        scores_text = f"MCDA Scores:\n"
        for k, v in mcda_data.get('normalized_scores', {}).items():
            if k in ['loss', 'accuracy', 'stability']:
                scores_text += f"  {k}: {v:.3f}\n"
        ax_loss.text(0.02, 0.98, scores_text, 
                    transform=ax_loss.transAxes,
                    fontsize=7, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

def main():
    """메인 실행"""
    print("🎨 Sweet Spot Visualization 재생성 시작...")
    
    # 데이터 로드
    sweet_data, metrics_data = load_data()
    
    # 출력 디렉토리
    viz_dir = Path('training/sweet_spot_analysis/visualizations')
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # 모듈 목록
    modules = ['neural_analyzers', 'emotion_head', 'bentham_head', 
               'regret_head', 'surd_head', 'backbone', 'system']
    
    # 전체 플롯 생성
    fig = plt.figure(figsize=(20, 12))
    
    for i, module_name in enumerate(modules, 1):
        if module_name not in sweet_data:
            continue
            
        ax = plt.subplot(3, 3, i)
        
        # 모듈 데이터 추출
        module_data = sweet_data[module_name]
        metrics = module_data.get('metrics', {})
        
        # Loss 데이터
        losses = metrics.get('val_losses', metrics.get('train_losses', []))
        epochs = list(range(1, len(losses) + 1))
        
        # Accuracy 데이터
        acc_epochs, accuracies = extract_accuracies(metrics_data, module_name)
        
        if losses:
            plot_module_analysis(ax, module_name, sweet_data, metrics_data, 
                               epochs, losses, accuracies)
    
    # 전체 요약 플롯 (8번째 위치)
    ax_summary = plt.subplot(3, 3, 8)
    
    # 각 모듈의 추천 epoch 요약
    recommended_epochs = []
    confidences = []
    module_names = []
    
    for module in modules:
        if module in sweet_data:
            rec = sweet_data[module].get('recommendation', {})
            if rec:
                module_names.append(module.replace('_', '\n'))
                recommended_epochs.append(rec.get('epoch', 0))
                confidences.append(rec.get('confidence', 0))
    
    # 막대 그래프
    x_pos = np.arange(len(module_names))
    bars = ax_summary.bar(x_pos, recommended_epochs, color='steelblue', alpha=0.7)
    
    # confidence를 색상으로 표현
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        bar.set_facecolor(plt.cm.RdYlGn(conf))
        # 막대 위에 epoch과 confidence 표시
        height = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'E{int(height)}\n{conf:.0%}',
                       ha='center', va='bottom', fontsize=8)
    
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(module_names, fontsize=8)
    ax_summary.set_ylabel('Recommended Epoch', fontsize=10)
    ax_summary.set_title('Best Checkpoints Summary', fontsize=11, fontweight='bold')
    ax_summary.grid(True, alpha=0.3, axis='y')
    ax_summary.set_ylim(0, 55)
    
    # 추천 체크포인트 텍스트 (9번째 위치)
    ax_text = plt.subplot(3, 3, 9)
    ax_text.axis('off')
    
    recommendation_text = """
    📊 Checkpoint Recommendations:
    
    🏆 Primary (Best Overall):
    • Epoch 30: Pre-SURD drop
    • Epoch 48: Neural Sweet Spot  
    • Epoch 50: Final convergence
    
    📌 Alternative Options:
    • Epoch 35: Stable middle point
    • Epoch 39: Regret head optimal
    • Epoch 40: SURD recovery
    
    ⚠️ Notes:
    • SURD: Threshold change at E30
      causes accuracy drop
    • Confidence: Voting agreement
      (25%=1/4, 50%=2/4, etc.)
    • Loss continues decreasing
      despite accuracy changes
    """
    
    ax_text.text(0.1, 0.9, recommendation_text,
                transform=ax_text.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 전체 제목
    fig.suptitle('Sweet Spot Analysis - Enhanced Visualization', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 저장
    output_path = viz_dir / 'enhanced_sweet_spot_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization 저장: {output_path}")
    
    # 개별 모듈 상세 플롯도 생성
    for module_name in modules:
        if module_name not in sweet_data:
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        module_data = sweet_data[module_name]
        metrics = module_data.get('metrics', {})
        losses = metrics.get('val_losses', metrics.get('train_losses', []))
        epochs = list(range(1, len(losses) + 1))
        acc_epochs, accuracies = extract_accuracies(metrics_data, module_name)
        
        if losses:
            plot_module_analysis(ax, module_name, sweet_data, metrics_data,
                               epochs, losses, accuracies)
        
        plt.tight_layout()
        module_output = viz_dir / f'{module_name}_detailed.png'
        plt.savefig(module_output, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - {module_name}: {module_output}")
    
    print("\n✨ Visualization 재생성 완료!")
    return output_path

if __name__ == "__main__":
    main()