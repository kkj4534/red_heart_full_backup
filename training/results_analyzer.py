#!/usr/bin/env python3
"""
학습 결과 분석기 및 docs 자동 생성
Training Results Analyzer and Automatic Documentation Generator
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import markdown
from jinja2 import Template
import os

class TrainingResultsAnalyzer:
    """학습 결과 분석 및 문서 생성기"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.training_dir = self.project_root / 'training' / 'outputs'
        self.docs_dir = self.project_root / 'docs'
        self.docs_dir.mkdir(exist_ok=True)
        
        # 한글 폰트 설정 (matplotlib)
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
    def analyze_training_report(self, report_path: Path) -> Dict[str, Any]:
        """학습 리포트 분석"""
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        analysis = {
            'report_path': str(report_path),
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': self._extract_basic_metrics(report),
            'regret_analysis': self._analyze_regret_patterns(report),
            'performance_trends': self._analyze_performance_trends(report),
            'efficiency_metrics': self._calculate_efficiency_metrics(report),
            'recommendations': self._generate_recommendations(report)
        }
        
        return analysis
    
    def _extract_basic_metrics(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """기본 메트릭 추출"""
        training_summary = report.get('training_summary', {})
        model_info = report.get('model_info', {})
        
        return {
            'total_training_steps': training_summary.get('total_steps', 0),
            'total_regrets_generated': training_summary.get('total_regrets', 0),
            'total_bentham_calculations': training_summary.get('total_bentham_calculations', 0),
            'final_loss': training_summary.get('final_loss', 0),
            'training_duration_hours': training_summary.get('training_duration', 0) / 3600,
            'model_parameters': model_info.get('main_model_parameters', 0),
            'regrets_per_step': training_summary.get('average_regrets_per_step', 0),
            'benthams_per_step': training_summary.get('average_benthams_per_step', 0)
        }
    
    def _analyze_regret_patterns(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """후회 패턴 분석"""
        config = report.get('configuration', {})
        training_summary = report.get('training_summary', {})
        
        target_regrets_per_step = config.get('regrets_per_step', 7)
        actual_regrets_per_step = training_summary.get('average_regrets_per_step', 0)
        
        regret_efficiency = (actual_regrets_per_step / target_regrets_per_step * 100) if target_regrets_per_step > 0 else 0
        
        bentham_per_regret = (
            training_summary.get('total_bentham_calculations', 0) / 
            training_summary.get('total_regrets', 1)
        )
        
        return {
            'target_regrets_per_step': target_regrets_per_step,
            'actual_regrets_per_step': actual_regrets_per_step,
            'regret_generation_efficiency': regret_efficiency,
            'bentham_calculations_per_regret': bentham_per_regret,
            'expected_bentham_per_regret': config.get('bentham_calculations_per_regret', 3),
            'bentham_efficiency': (bentham_per_regret / 3 * 100) if bentham_per_regret > 0 else 0
        }
    
    def _analyze_performance_trends(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """성능 트렌드 분석"""
        training_stats = report.get('training_stats', {})
        
        if not training_stats:
            return {'trend_analysis': 'No training statistics available'}
        
        # 손실 트렌드
        losses = training_stats.get('total_loss', [])
        if len(losses) > 10:
            early_loss = np.mean(losses[:len(losses)//4])
            late_loss = np.mean(losses[-len(losses)//4:])
            improvement = ((early_loss - late_loss) / early_loss * 100) if early_loss > 0 else 0
        else:
            improvement = 0
        
        return {
            'total_training_steps': len(losses),
            'loss_improvement_percentage': improvement,
            'convergence_analysis': 'Converged' if improvement > 10 else 'Partially converged' if improvement > 0 else 'No clear convergence',
            'training_stability': 'Stable' if len(losses) > 100 else 'Short training'
        }
    
    def _calculate_efficiency_metrics(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """효율성 메트릭 계산"""
        training_summary = report.get('training_summary', {})
        storage_usage = report.get('storage_usage', {})
        model_info = report.get('model_info', {})
        
        duration_hours = training_summary.get('training_duration', 0) / 3600
        parameters = model_info.get('main_model_parameters', 1)
        storage_gb = storage_usage.get('final_size_gb', 0)
        
        return {
            'parameters_per_hour': parameters / duration_hours if duration_hours > 0 else 0,
            'regrets_per_hour': training_summary.get('total_regrets', 0) / duration_hours if duration_hours > 0 else 0,
            'storage_efficiency_mb_per_parameter': (storage_gb * 1024) / parameters if parameters > 0 else 0,
            'training_speed_steps_per_hour': training_summary.get('total_steps', 0) / duration_hours if duration_hours > 0 else 0,
            'bentham_calculations_per_hour': training_summary.get('total_bentham_calculations', 0) / duration_hours if duration_hours > 0 else 0
        }
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        training_summary = report.get('training_summary', {})
        regret_efficiency = self._analyze_regret_patterns(report).get('regret_generation_efficiency', 0)
        
        # 후회 생성 효율성 검토
        if regret_efficiency < 90:
            recommendations.append("후회 시나리오 생성 효율성이 낮습니다. 후회 계산 로직을 최적화해보세요.")
        
        # 학습 안정성 검토
        final_loss = training_summary.get('final_loss', 0)
        if final_loss > 1.0:
            recommendations.append("최종 손실이 높습니다. 학습률 조정이나 추가 학습을 고려해보세요.")
        
        # 스토리지 사용량 검토
        storage_usage = report.get('storage_usage', {})
        if storage_usage.get('final_size_gb', 0) > 180:
            recommendations.append("스토리지 사용량이 한계에 가깝습니다. 로그 정리 주기를 단축해보세요.")
        
        # 벤담 계산 효율성
        bentham_efficiency = self._analyze_regret_patterns(report).get('bentham_efficiency', 0)
        if bentham_efficiency < 95:
            recommendations.append("벤담 쾌락 계산 효율성을 개선할 수 있습니다.")
        
        if not recommendations:
            recommendations.append("학습이 성공적으로 완료되었습니다. 모든 메트릭이 목표 범위 내에 있습니다.")
        
        return recommendations
    
    def create_visualizations(self, analysis: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Path]:
        """시각화 생성"""
        viz_dir = self.docs_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        plots = {}
        
        # 1. 기본 메트릭 대시보드
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Red Heart XAI 학습 결과 대시보드', fontsize=16, fontweight='bold')
        
        # 후회 및 벤담 계산 통계
        metrics = analysis['basic_metrics']
        labels = ['총 후회', '총 벤담 계산', '총 학습 스텝']
        values = [
            metrics['total_regrets_generated'],
            metrics['total_bentham_calculations'],
            metrics['total_training_steps']
        ]
        
        axes[0, 0].bar(labels, values, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        axes[0, 0].set_title('학습 통계')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 효율성 메트릭
        efficiency = analysis['efficiency_metrics']
        eff_labels = ['후회/시간', '벤담/시간', '스텝/시간']
        eff_values = [
            efficiency['regrets_per_hour'],
            efficiency['bentham_calculations_per_hour'],
            efficiency['training_speed_steps_per_hour']
        ]
        
        axes[0, 1].bar(eff_labels, eff_values, color=['#d62728', '#9467bd', '#8c564b'])
        axes[0, 1].set_title('시간당 효율성')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 모델 정보
        model_data = [
            ['파라미터 수', f"{metrics['model_parameters']:,}"],
            ['학습 시간', f"{metrics['training_duration_hours']:.2f}h"],
            ['최종 손실', f"{metrics['final_loss']:.4f}"]
        ]
        
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=model_data, colLabels=['메트릭', '값'], 
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1, 0].set_title('모델 정보')
        
        # 후회 패턴 분석
        regret_analysis = analysis['regret_analysis']
        regret_data = [
            ['목표 후회/스텝', regret_analysis['target_regrets_per_step']],
            ['실제 후회/스텝', f"{regret_analysis['actual_regrets_per_step']:.2f}"],
            ['후회 효율성', f"{regret_analysis['regret_generation_efficiency']:.1f}%"],
            ['벤담 효율성', f"{regret_analysis['bentham_efficiency']:.1f}%"]
        ]
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        regret_table = axes[1, 1].table(cellText=regret_data, colLabels=['메트릭', '값'],
                                       cellLoc='center', loc='center')
        regret_table.auto_set_font_size(False)
        regret_table.set_fontsize(10)
        axes[1, 1].set_title('후회 분석')
        
        plt.tight_layout()
        dashboard_path = viz_dir / 'training_dashboard.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['dashboard'] = dashboard_path
        
        # 2. 손실 트렌드 (사용 가능한 경우)
        training_stats = report.get('training_stats', {})
        if 'total_loss' in training_stats and len(training_stats['total_loss']) > 10:
            plt.figure(figsize=(12, 6))
            losses = training_stats['total_loss']
            plt.plot(losses, linewidth=2, color='#1f77b4')
            plt.title('학습 손실 트렌드', fontsize=14, fontweight='bold')
            plt.xlabel('학습 스텝')
            plt.ylabel('손실')
            plt.grid(True, alpha=0.3)
            
            # 이동 평균 추가
            if len(losses) > 50:
                window = min(50, len(losses) // 10)
                moving_avg = pd.Series(losses).rolling(window=window).mean()
                plt.plot(moving_avg, linewidth=3, color='#ff7f0e', label=f'{window}-step 이동평균')
                plt.legend()
            
            loss_trend_path = viz_dir / 'loss_trend.png'
            plt.savefig(loss_trend_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['loss_trend'] = loss_trend_path
        
        return plots
    
    def generate_markdown_report(self, analysis: Dict[str, Any], 
                               report: Dict[str, Any], 
                               plots: Dict[str, Path]) -> Path:
        """마크다운 리포트 생성"""
        
        template_str = """
# Red Heart XAI 학습 결과 리포트

**생성일시**: {{ analysis.timestamp }}  
**학습 데이터**: {{ analysis.report_path }}

## 📊 학습 요약

### 기본 통계
- **총 학습 스텝**: {{ "{:,}".format(analysis.basic_metrics.total_training_steps) }}
- **총 후회 생성**: {{ "{:,}".format(analysis.basic_metrics.total_regrets_generated) }}
- **총 벤담 계산**: {{ "{:,}".format(analysis.basic_metrics.total_bentham_calculations) }}
- **학습 시간**: {{ "%.2f"|format(analysis.basic_metrics.training_duration_hours) }}시간
- **모델 파라미터**: {{ "{:,}".format(analysis.basic_metrics.model_parameters) }}개
- **최종 손실**: {{ "%.6f"|format(analysis.basic_metrics.final_loss) }}

### 🎯 후회 시스템 분석

#### 후회 생성 성능
- **목표 후회/스텝**: {{ analysis.regret_analysis.target_regrets_per_step }}회
- **실제 후회/스텝**: {{ "%.2f"|format(analysis.regret_analysis.actual_regrets_per_step) }}회
- **후회 생성 효율성**: {{ "%.1f"|format(analysis.regret_analysis.regret_generation_efficiency) }}%

#### 벤담 쾌락 계산
- **목표 벤담/후회**: {{ analysis.regret_analysis.expected_bentham_per_regret }}회
- **실제 벤담/후회**: {{ "%.2f"|format(analysis.regret_analysis.bentham_calculations_per_regret) }}회
- **벤담 계산 효율성**: {{ "%.1f"|format(analysis.regret_analysis.bentham_efficiency) }}%

### ⚡ 성능 효율성

- **시간당 후회 생성**: {{ "{:,.0f}".format(analysis.efficiency_metrics.regrets_per_hour) }}개
- **시간당 벤담 계산**: {{ "{:,.0f}".format(analysis.efficiency_metrics.bentham_calculations_per_hour) }}개  
- **시간당 학습 스텝**: {{ "{:,.0f}".format(analysis.efficiency_metrics.training_speed_steps_per_hour) }}개

### 📈 학습 트렌드

{{ analysis.performance_trends.convergence_analysis }}

{% if analysis.performance_trends.loss_improvement_percentage > 0 %}
- **손실 개선율**: {{ "%.1f"|format(analysis.performance_trends.loss_improvement_percentage) }}%
{% endif %}

## 🎨 시각화

{% if plots.dashboard %}
### 학습 대시보드
![학습 대시보드](visualizations/training_dashboard.png)
{% endif %}

{% if plots.loss_trend %}
### 손실 트렌드
![손실 트렌드](visualizations/loss_trend.png)
{% endif %}

## 💡 권장사항

{% for recommendation in analysis.recommendations %}
- {{ recommendation }}
{% endfor %}

## 🔧 기술적 세부사항

### 학습 설정
- **에포크 수**: {{ report.configuration.epochs }}
- **배치 크기**: {{ report.configuration.batch_size }}
- **학습률**: {{ report.configuration.learning_rate }}
- **후회/스텝**: {{ report.configuration.regrets_per_step }}
- **벤담/후회**: {{ report.configuration.bentham_calculations_per_regret }}

### 스토리지 사용량
- **최종 사용량**: {{ "%.2f"|format(report.storage_usage.final_size_gb) }}GB
- **허용 한계**: {{ "%.0f"|format(report.storage_usage.max_allowed_gb) }}GB
- **사용률**: {{ "%.1f"|format((report.storage_usage.final_size_gb / report.storage_usage.max_allowed_gb) * 100) }}%

### XAI 통합
- **XAI 로그 생성**: {{ report.xai_integration.xai_logs_generated }}개
- **세션 ID**: `{{ report.xai_integration.session_id }}`

---

*이 리포트는 Red Heart XAI 시스템에 의해 자동 생성되었습니다.*  
*생성 시각: {{ analysis.timestamp }}*
"""

        template = Template(template_str)
        content = template.render(analysis=analysis, report=report, plots=plots)
        
        # 마크다운 파일 저장
        report_path = self.docs_dir / f'training_report_{int(datetime.now().timestamp())}.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return report_path
    
    def generate_html_report(self, markdown_path: Path) -> Path:
        """HTML 리포트 생성"""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # CSS 스타일 추가
        html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Red Heart XAI 학습 결과 리포트</title>
    <style>
        body {
            font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 { color: #2c3e50; }
        h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }
        .metric { 
            background: #f8f9fa; 
            padding: 10px; 
            margin: 5px 0; 
            border-left: 4px solid #3498db; 
        }
        .recommendation {
            background: #e8f5e8;
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #27ae60;
        }
        img { max-width: 100%; height: auto; margin: 10px 0; }
        code { background: #f1f2f6; padding: 2px 4px; border-radius: 3px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
        }
    </style>
</head>
<body>
    <div class="container">
        {{ content }}
    </div>
    <script>
        // 이미지 경로 수정
        document.querySelectorAll('img').forEach(img => {
            if (img.src.includes('visualizations/')) {
                img.src = img.src.replace('visualizations/', 'visualizations/');
            }
        });
    </script>
</body>
</html>
"""
        
        # 마크다운을 HTML로 변환
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # 템플릿에 삽입
        final_html = html_template.replace('{{ content }}', html_content)
        
        # HTML 파일 저장
        html_path = markdown_path.with_suffix('.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        return html_path
    
    def analyze_and_generate_docs(self, report_path: Optional[Path] = None) -> Dict[str, Path]:
        """전체 분석 및 문서 생성 프로세스"""
        
        # 가장 최근 리포트 찾기
        if report_path is None:
            reports_dir = self.training_dir / 'reports'
            if not reports_dir.exists():
                raise FileNotFoundError("학습 리포트를 찾을 수 없습니다.")
            
            report_files = list(reports_dir.glob('regret_training_report_*.json'))
            if not report_files:
                raise FileNotFoundError("학습 리포트 파일이 없습니다.")
            
            report_path = max(report_files, key=lambda x: x.stat().st_mtime)
        
        print(f"📊 분석 대상: {report_path}")
        
        # 1. 리포트 분석
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        analysis = self.analyze_training_report(report_path)
        
        # 2. 시각화 생성
        plots = self.create_visualizations(analysis, report)
        
        # 3. 마크다운 리포트 생성
        markdown_path = self.generate_markdown_report(analysis, report, plots)
        
        # 4. HTML 리포트 생성
        html_path = self.generate_html_report(markdown_path)
        
        # 5. 요약 JSON 생성
        summary_path = self.docs_dir / 'latest_training_summary.json'
        summary = {
            'analysis_timestamp': analysis['timestamp'],
            'source_report': str(report_path),
            'key_metrics': analysis['basic_metrics'],
            'recommendations': analysis['recommendations'],
            'files_generated': {
                'markdown_report': str(markdown_path),
                'html_report': str(html_path),
                'visualizations': {k: str(v) for k, v in plots.items()}
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        result = {
            'analysis': analysis,
            'markdown_report': markdown_path,
            'html_report': html_path,
            'summary': summary_path,
            'visualizations': plots
        }
        
        print(f"✅ 분석 완료!")
        print(f"📄 마크다운 리포트: {markdown_path}")
        print(f"🌐 HTML 리포트: {html_path}")
        print(f"📊 시각화: {len(plots)}개 생성")
        
        return result

if __name__ == "__main__":
    # 테스트 실행
    project_root = Path(__file__).parent.parent
    analyzer = TrainingResultsAnalyzer(project_root)
    
    print("🔍 학습 결과 분석기 준비 완료")
    print("analyze_and_generate_docs()를 호출하여 최신 학습 결과를 분석하세요.")