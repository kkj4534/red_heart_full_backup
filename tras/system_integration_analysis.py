"""
시스템 통합 분석기 (System Integration Analysis)
Red Heart 시스템의 완전성과 연결성을 체계적으로 분석하는 도구

핵심 기능:
1. 모듈 간 의존성 분석
2. 임포트 체인 검증
3. 누락된 연결 탐지
4. 순환 참조 확인
5. 시스템 완전성 보고서 생성
"""

import ast
import os
import sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import json

class SystemIntegrationAnalyzer:
    """시스템 통합 분석기"""
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = Path(__file__).parent
        
        self.project_root = Path(project_root)
        self.modules_info = {}
        self.import_graph = defaultdict(set)
        self.issues = []
        self.missing_dependencies = set()
        self.circular_dependencies = []
        
        # 핵심 모듈 정의
        self.core_modules = {
            'config': '시스템 설정 및 환경 변수',
            'data_models': '데이터 구조 정의',
            'emotion_ethics_regret_circuit': '감정-윤리-후회 삼각 회로',
            'ethics_policy_updater': '윤리 정책 자동 조정기',
            'phase_controller': '페이즈 컨트롤러 (학습/실행/반성)',
            'xai_feedback_integrator': 'XAI 피드백 통합기',
            'fuzzy_emotion_ethics_mapper': '퍼지 로직 감정-윤리 매핑',
            'deep_multi_dimensional_ethics_system': '심층 다차원 윤리 추론',
            'temporal_event_propagation_analyzer': '시계열 사건 전파 분석기',
            'integrated_system_orchestrator': '통합 시스템 오케스트레이터'
        }
        
        # 필수 외부 의존성
        self.required_external_deps = {
            'torch', 'numpy', 'transformers', 'scikit-learn',
            'pandas', 'matplotlib', 'seaborn', 'tqdm',
            'logging', 'json', 'pathlib', 'dataclasses',
            'typing', 'collections', 'enum', 'time',
            'threading', 'asyncio', 'concurrent.futures'
        }
        
    def analyze_system(self) -> Dict[str, Any]:
        """전체 시스템 분석 실행"""
        print("🔍 Red Heart 시스템 통합성 분석 시작")
        
        analysis_result = {
            'overview': {},
            'module_analysis': {},
            'dependency_analysis': {},
            'integration_issues': [],
            'recommendations': [],
            'completeness_score': 0.0
        }
        
        try:
            # 1. 모듈 발견 및 기본 분석
            print("\n📁 모듈 발견 및 분석")
            self._discover_modules()
            analysis_result['module_analysis'] = self._analyze_modules()
            
            # 2. 의존성 분석
            print("\n🔗 의존성 분석")
            self._analyze_dependencies()
            analysis_result['dependency_analysis'] = self._get_dependency_report()
            
            # 3. 임포트 테스트
            print("\n🧪 임포트 테스트")
            import_results = self._test_imports()
            analysis_result['import_test_results'] = import_results
            
            # 4. 통합성 검증
            print("\n✅ 통합성 검증")
            integration_issues = self._check_integration_issues()
            analysis_result['integration_issues'] = integration_issues
            
            # 5. 권장사항 생성
            print("\n💡 권장사항 생성")
            recommendations = self._generate_recommendations()
            analysis_result['recommendations'] = recommendations
            
            # 6. 완전성 점수 계산
            completeness_score = self._calculate_completeness_score()
            analysis_result['completeness_score'] = completeness_score
            
            # 7. 개요 생성
            analysis_result['overview'] = self._generate_overview(analysis_result)
            
            print(f"\n📊 분석 완료 - 완전성 점수: {completeness_score:.1%}")
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            analysis_result['error'] = str(e)
            analysis_result['traceback'] = traceback.format_exc()
        
        return analysis_result
    
    def _discover_modules(self):
        """프로젝트 내 Python 모듈 발견"""
        python_files = list(self.project_root.glob("*.py"))
        
        for py_file in python_files:
            if py_file.name.startswith('__') or py_file.name.startswith('test_'):
                continue
                
            module_name = py_file.stem
            try:
                # AST를 사용한 기본 분석
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                self.modules_info[module_name] = {
                    'path': str(py_file),
                    'size': py_file.stat().st_size,
                    'imports': self._extract_imports(tree),
                    'classes': self._extract_classes(tree),
                    'functions': self._extract_functions(tree),
                    'is_core': module_name in self.core_modules,
                    'description': self.core_modules.get(module_name, '일반 모듈')
                }
                
            except Exception as e:
                self.issues.append(f"모듈 {module_name} 파싱 실패: {e}")
                self.modules_info[module_name] = {
                    'path': str(py_file),
                    'error': str(e),
                    'is_core': module_name in self.core_modules
                }
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """AST에서 임포트 문 추출"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname
                    })
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname
                    })
        
        return imports
    
    def _extract_classes(self, tree: ast.AST) -> List[str]:
        """AST에서 클래스 정의 추출"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    
    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """AST에서 함수 정의 추출"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions
    
    def _analyze_modules(self) -> Dict[str, Any]:
        """모듈 분석 결과"""
        total_modules = len(self.modules_info)
        core_modules_found = sum(1 for info in self.modules_info.values() if info.get('is_core', False))
        core_modules_total = len(self.core_modules)
        
        # 누락된 핵심 모듈
        found_core_modules = {name for name, info in self.modules_info.items() if info.get('is_core', False)}
        missing_core_modules = set(self.core_modules.keys()) - found_core_modules
        
        # 모듈 크기 통계
        sizes = [info.get('size', 0) for info in self.modules_info.values()]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        return {
            'total_modules': total_modules,
            'core_modules_found': core_modules_found,
            'core_modules_total': core_modules_total,
            'core_modules_coverage': core_modules_found / core_modules_total if core_modules_total > 0 else 0,
            'missing_core_modules': list(missing_core_modules),
            'average_module_size': avg_size,
            'module_details': self.modules_info
        }
    
    def _analyze_dependencies(self):
        """의존성 분석"""
        for module_name, module_info in self.modules_info.items():
            if 'imports' not in module_info:
                continue
                
            for import_info in module_info['imports']:
                imported_module = import_info['module']
                
                # 로컬 모듈 의존성
                if imported_module in self.modules_info:
                    self.import_graph[module_name].add(imported_module)
                
                # 외부 의존성 체크
                elif imported_module.split('.')[0] not in self.required_external_deps:
                    # 표준 라이브러리나 알려진 의존성이 아닌 경우
                    if not self._is_standard_library(imported_module):
                        self.missing_dependencies.add(imported_module)
        
        # 순환 의존성 탐지
        self.circular_dependencies = self._detect_circular_dependencies()
    
    def _is_standard_library(self, module_name: str) -> bool:
        """표준 라이브러리 모듈인지 확인"""
        standard_libs = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'dataclasses', 'enum', 'abc', 'logging',
            'threading', 'asyncio', 'concurrent', 'traceback', 'math',
            'random', 'itertools', 'functools', 'operator', 're'
        }
        
        root_module = module_name.split('.')[0]
        return root_module in standard_libs
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """순환 의존성 탐지"""
        def dfs(node, path, visited, rec_stack):
            if node in rec_stack:
                # 순환 발견
                cycle_start = path.index(node)
                return [path[cycle_start:] + [node]]
            
            if node in visited:
                return []
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            cycles = []
            for neighbor in self.import_graph.get(node, set()):
                cycles.extend(dfs(neighbor, path, visited, rec_stack))
            
            path.pop()
            rec_stack.remove(node)
            
            return cycles
        
        all_cycles = []
        visited = set()
        
        for module in self.import_graph:
            if module not in visited:
                cycles = dfs(module, [], visited, set())
                all_cycles.extend(cycles)
        
        return all_cycles
    
    def _get_dependency_report(self) -> Dict[str, Any]:
        """의존성 분석 보고서"""
        return {
            'total_dependencies': sum(len(deps) for deps in self.import_graph.values()),
            'modules_with_dependencies': len([m for m in self.import_graph if self.import_graph[m]]),
            'missing_dependencies': list(self.missing_dependencies),
            'circular_dependencies': self.circular_dependencies,
            'dependency_graph': {k: list(v) for k, v in self.import_graph.items()},
            'most_imported_modules': self._get_most_imported_modules()
        }
    
    def _get_most_imported_modules(self) -> List[Tuple[str, int]]:
        """가장 많이 임포트되는 모듈들"""
        import_counts = defaultdict(int)
        
        for deps in self.import_graph.values():
            for dep in deps:
                import_counts[dep] += 1
        
        return sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _test_imports(self) -> Dict[str, Dict[str, Any]]:
        """실제 임포트 테스트"""
        results = {}
        
        # 프로젝트 루트를 Python 경로에 추가
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        for module_name in self.core_modules:
            result = {
                'success': False,
                'error': None,
                'missing_deps': [],
                'time_taken': 0
            }
            
            try:
                import time
                start_time = time.time()
                
                # 임포트 시도
                if module_name in sys.modules:
                    # 이미 로드된 모듈 재로드
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
                
                result['success'] = True
                result['time_taken'] = time.time() - start_time
                
            except ModuleNotFoundError as e:
                result['error'] = f"ModuleNotFoundError: {e}"
                # 누락된 의존성 추출
                missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
                result['missing_deps'].append(missing_module)
                
            except ImportError as e:
                result['error'] = f"ImportError: {e}"
                
            except Exception as e:
                result['error'] = f"기타 오류: {e}"
            
            results[module_name] = result
        
        return results
    
    def _check_integration_issues(self) -> List[Dict[str, Any]]:
        """통합성 문제 체크"""
        issues = []
        
        # 1. 핵심 모듈 누락 체크
        missing_core = [name for name in self.core_modules if name not in self.modules_info]
        for module in missing_core:
            issues.append({
                'type': 'missing_core_module',
                'severity': 'critical',
                'module': module,
                'description': f'핵심 모듈 {module}이 누락되었습니다.',
                'recommendation': f'{module}.py 파일을 생성하거나 복구하세요.'
            })
        
        # 2. 임포트 실패 체크
        import_results = self._test_imports()
        for module, result in import_results.items():
            if not result['success']:
                issues.append({
                    'type': 'import_failure',
                    'severity': 'high',
                    'module': module,
                    'description': f'모듈 {module} 임포트 실패: {result["error"]}',
                    'missing_deps': result.get('missing_deps', []),
                    'recommendation': '누락된 의존성을 설치하거나 임포트 경로를 수정하세요.'
                })
        
        # 3. 순환 의존성 체크
        for cycle in self.circular_dependencies:
            issues.append({
                'type': 'circular_dependency',
                'severity': 'medium',
                'modules': cycle,
                'description': f'순환 의존성 발견: {" -> ".join(cycle)}',
                'recommendation': '모듈 구조를 재설계하여 순환 의존성을 제거하세요.'
            })
        
        # 4. 고립된 모듈 체크
        connected_modules = set()
        for module, deps in self.import_graph.items():
            if deps:
                connected_modules.add(module)
                connected_modules.update(deps)
        
        isolated_modules = set(self.modules_info.keys()) - connected_modules
        for module in isolated_modules:
            if self.modules_info[module].get('is_core', False):
                issues.append({
                    'type': 'isolated_core_module',
                    'severity': 'medium',
                    'module': module,
                    'description': f'핵심 모듈 {module}이 다른 모듈과 연결되지 않았습니다.',
                    'recommendation': '모듈 간 통합을 위한 임포트 관계를 추가하세요.'
                })
        
        return issues
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """시스템 개선 권장사항"""
        recommendations = []
        
        # 의존성 관련 권장사항
        if self.missing_dependencies:
            recommendations.append({
                'category': '의존성 관리',
                'priority': 'high',
                'title': '누락된 외부 의존성 설치',
                'description': f'다음 패키지들을 설치해야 합니다: {", ".join(self.missing_dependencies)}',
                'action': 'pip install ' + ' '.join(self.missing_dependencies)
            })
        
        # 순환 의존성 해결
        if self.circular_dependencies:
            recommendations.append({
                'category': '아키텍처',
                'priority': 'medium',
                'title': '순환 의존성 해결',
                'description': f'{len(self.circular_dependencies)}개의 순환 의존성이 발견되었습니다.',
                'action': '모듈 구조를 재설계하여 계층적 의존성을 만드세요.'
            })
        
        # 모듈 통합 개선
        import_results = self._test_imports()
        failed_imports = [name for name, result in import_results.items() if not result['success']]
        
        if failed_imports:
            recommendations.append({
                'category': '모듈 통합',
                'priority': 'high',
                'title': '임포트 실패 모듈 수정',
                'description': f'{len(failed_imports)}개 모듈의 임포트가 실패했습니다.',
                'action': '각 모듈의 의존성을 확인하고 missing_deps를 해결하세요.'
            })
        
        # 테스트 커버리지
        recommendations.append({
            'category': '품질 보증',
            'priority': 'medium',
            'title': '통합 테스트 추가',
            'description': '모듈 간 상호작용을 검증하는 통합 테스트가 필요합니다.',
            'action': 'integrated_system_orchestrator.py의 테스트 기능을 확장하세요.'
        })
        
        return recommendations
    
    def _calculate_completeness_score(self) -> float:
        """시스템 완전성 점수 계산 (0.0 ~ 1.0)"""
        total_score = 0.0
        weight_sum = 0.0
        
        # 1. 핵심 모듈 존재 여부 (40%)
        core_modules_found = sum(1 for name in self.core_modules if name in self.modules_info)
        core_score = core_modules_found / len(self.core_modules)
        total_score += core_score * 0.4
        weight_sum += 0.4
        
        # 2. 임포트 성공률 (30%)
        import_results = self._test_imports()
        successful_imports = sum(1 for result in import_results.values() if result['success'])
        import_score = successful_imports / len(import_results) if import_results else 0
        total_score += import_score * 0.3
        weight_sum += 0.3
        
        # 3. 의존성 건전성 (20%)
        dependency_score = 1.0
        if self.circular_dependencies:
            dependency_score -= len(self.circular_dependencies) * 0.1
        if self.missing_dependencies:
            dependency_score -= len(self.missing_dependencies) * 0.05
        dependency_score = max(0.0, dependency_score)
        total_score += dependency_score * 0.2
        weight_sum += 0.2
        
        # 4. 통합성 (10%)
        integration_issues = self._check_integration_issues()
        critical_issues = len([i for i in integration_issues if i['severity'] == 'critical'])
        high_issues = len([i for i in integration_issues if i['severity'] == 'high'])
        
        integration_score = 1.0 - (critical_issues * 0.3 + high_issues * 0.1)
        integration_score = max(0.0, integration_score)
        total_score += integration_score * 0.1
        weight_sum += 0.1
        
        return total_score / weight_sum if weight_sum > 0 else 0.0
    
    def _generate_overview(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """분석 개요 생성"""
        module_analysis = analysis_result.get('module_analysis', {})
        dependency_analysis = analysis_result.get('dependency_analysis', {})
        integration_issues = analysis_result.get('integration_issues', [])
        
        return {
            'total_modules': module_analysis.get('total_modules', 0),
            'core_modules_coverage': f"{module_analysis.get('core_modules_coverage', 0):.1%}",
            'missing_core_modules': len(module_analysis.get('missing_core_modules', [])),
            'total_dependencies': dependency_analysis.get('total_dependencies', 0),
            'circular_dependencies': len(dependency_analysis.get('circular_dependencies', [])),
            'critical_issues': len([i for i in integration_issues if i['severity'] == 'critical']),
            'high_issues': len([i for i in integration_issues if i['severity'] == 'high']),
            'medium_issues': len([i for i in integration_issues if i['severity'] == 'medium']),
            'completeness_score': analysis_result.get('completeness_score', 0.0)
        }
    
    def save_report(self, analysis_result: Dict[str, Any], filepath: str = None):
        """분석 보고서 저장"""
        if filepath is None:
            filepath = self.project_root / 'system_integration_report.json'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"📄 분석 보고서가 {filepath}에 저장되었습니다.")
    
    def print_summary(self, analysis_result: Dict[str, Any]):
        """분석 결과 요약 출력"""
        overview = analysis_result.get('overview', {})
        
        print("\n" + "="*60)
        print("🎯 RED HEART 시스템 통합성 분석 요약")
        print("="*60)
        
        print(f"📊 전체 모듈: {overview.get('total_modules', 0)}개")
        print(f"🎯 핵심 모듈 커버리지: {overview.get('core_modules_coverage', '0%')}")
        print(f"🔗 총 의존성: {overview.get('total_dependencies', 0)}개")
        print(f"🔄 순환 의존성: {overview.get('circular_dependencies', 0)}개")
        print(f"⚠️  심각한 문제: {overview.get('critical_issues', 0)}개")
        print(f"🚨 높은 문제: {overview.get('high_issues', 0)}개")
        print(f"📈 완전성 점수: {overview.get('completeness_score', 0.0):.1%}")
        
        # 상태 판정
        score = overview.get('completeness_score', 0.0)
        if score >= 0.9:
            status = "🟢 EXCELLENT"
        elif score >= 0.7:
            status = "🟡 GOOD"
        elif score >= 0.5:
            status = "🟠 NEEDS_IMPROVEMENT"
        else:
            status = "🔴 CRITICAL"
        
        print(f"\n🎭 시스템 상태: {status}")
        
        # 주요 문제점
        issues = analysis_result.get('integration_issues', [])
        critical_issues = [i for i in issues if i['severity'] == 'critical']
        
        if critical_issues:
            print(f"\n🚨 즉시 해결이 필요한 문제:")
            for issue in critical_issues[:3]:
                print(f"   • {issue['description']}")
        
        # 권장사항
        recommendations = analysis_result.get('recommendations', [])
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        
        if high_priority:
            print(f"\n💡 우선 권장사항:")
            for rec in high_priority[:3]:
                print(f"   • {rec['title']}")


def main():
    """메인 실행 함수"""
    print("🚀 Red Heart 시스템 통합성 분석 시작")
    
    # 분석기 초기화
    analyzer = SystemIntegrationAnalyzer()
    
    # 전체 시스템 분석
    analysis_result = analyzer.analyze_system()
    
    # 결과 요약 출력
    analyzer.print_summary(analysis_result)
    
    # 보고서 저장
    analyzer.save_report(analysis_result)
    
    return analysis_result


if __name__ == "__main__":
    main()