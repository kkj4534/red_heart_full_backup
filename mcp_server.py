"""
MCP (Model Context Protocol) Server for Red Heart Ethics System
Red Heart 윤리 시스템을 위한 MCP 서버 구현

MD 문서 사양에 따른 구현:
- Red Heart AI를 MCP 도구로 제공
- Claude와 통합하여 윤리적 의사결정 지원
- HEAVY 모드 기본 사용
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Red Heart 시스템 임포트
sys.path.append('/mnt/c/large_project/linux_red_heart')
from main_unified import UnifiedInferenceSystem, InferenceConfig, MemoryMode

logger = logging.getLogger('RedHeart.MCPServer')

# MCP 요청/응답 모델
class MCPRequest(BaseModel):
    """MCP 요청 모델"""
    text: str = Field(..., description="분석할 텍스트 (윤리적 딜레마 상황)")
    mode: str = Field(default="heavy", description="추론 모드 (auto/heavy)")
    scenarios: Optional[List[str]] = Field(None, description="LLM이 제시한 시나리오들")
    context: Optional[Dict[str, Any]] = Field(None, description="추가 컨텍스트")

class ScenarioAnalysis(BaseModel):
    """시나리오 분석 결과"""
    scenario: str
    score: float
    ethical_analysis: Dict[str, Any]
    utility_score: float
    regret_potential: float

class MCPResponse(BaseModel):
    """MCP 응답 모델"""
    top_scenarios: List[ScenarioAnalysis]
    recommendation: str
    total_evaluated: int
    processing_time: float
    metadata: Dict[str, Any]

class RedHeartMCPServer:
    """Red Heart MCP 서버
    
    MD 문서 사양:
    - 이름: red-heart-ethics
    - 설명: Red Heart AI 윤리적 의사결정 지원 시스템
    - 입력: 텍스트 (윤리적 딜레마 상황)
    - 출력: 상위 2개 시나리오 및 추천
    """
    
    def __init__(self):
        self.logger = logger
        self.inference_system = None
        self.is_initialized = False
        
        # 서버 설정
        self.config = {
            'host': '0.0.0.0',
            'port': 8765,
            'name': 'red-heart-ethics',
            'version': '1.0.0',
            'description': 'Red Heart AI 윤리적 의사결정 지원 시스템'
        }
        
        # FastAPI 앱
        self.app = FastAPI(
            title=self.config['name'],
            version=self.config['version'],
            description=self.config['description']
        )
        
        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 라우트 설정
        self._setup_routes()
        
        self.logger.info(f"MCP 서버 초기화: {self.config['name']} v{self.config['version']}")
    
    async def initialize(self):
        """Red Heart 시스템 초기화"""
        if self.is_initialized:
            return
        
        self.logger.info("Red Heart 시스템 초기화 중...")
        
        try:
            # HEAVY 모드로 시스템 초기화
            config = InferenceConfig(
                memory_mode=MemoryMode.HEAVY,
                auto_memory_mode=False,
                llm_mode="none",  # MCP에서는 LLM 직접 사용 안 함
                use_three_view_scenario=True,
                use_multi_ethics_system=True,
                use_counterfactual_reasoning=True,
                use_advanced_regret_learning=True,
                use_temporal_propagation=True,
                use_meta_integration=True
            )
            
            self.inference_system = UnifiedInferenceSystem(config)
            await self.inference_system.initialize()
            
            self.is_initialized = True
            self.logger.info("Red Heart 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"시스템 초기화 실패: {e}")
            raise RuntimeError(f"Red Heart 초기화 실패: {e}")
    
    def _setup_routes(self):
        """API 라우트 설정"""
        
        @self.app.get("/")
        async def root():
            """서버 상태 확인"""
            return {
                "service": self.config['name'],
                "version": self.config['version'],
                "status": "ready" if self.is_initialized else "initializing",
                "description": self.config['description']
            }
        
        @self.app.get("/health")
        async def health():
            """헬스 체크"""
            return {
                "status": "healthy" if self.is_initialized else "starting",
                "initialized": self.is_initialized
            }
        
        @self.app.post("/analyze", response_model=MCPResponse)
        async def analyze_ethics(request: MCPRequest):
            """윤리적 딜레마 분석
            
            MCP 도구로 노출되는 메인 엔드포인트
            """
            if not self.is_initialized:
                await self.initialize()
            
            return await self.handle_request(request)
        
        @self.app.get("/mcp/manifest")
        async def mcp_manifest():
            """MCP 매니페스트 - Claude가 도구를 발견하기 위한 정보"""
            return {
                "name": self.config['name'],
                "version": self.config['version'],
                "description": self.config['description'],
                "tools": [
                    {
                        "name": "analyze_ethics",
                        "description": "윤리적 딜레마를 분석하고 최선의 행동을 추천",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "분석할 윤리적 딜레마 상황"
                                },
                                "scenarios": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "검토할 시나리오 목록 (선택사항)"
                                }
                            },
                            "required": ["text"]
                        },
                        "outputSchema": {
                            "type": "object",
                            "properties": {
                                "top_scenarios": {
                                    "type": "array",
                                    "description": "상위 2개 시나리오"
                                },
                                "recommendation": {
                                    "type": "string",
                                    "description": "최종 추천"
                                }
                            }
                        }
                    }
                ]
            }
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """MCP 요청 처리
        
        MD 문서 워크플로우:
        1. 텍스트 입력 받기
        2. 시나리오가 없으면 기본 생성
        3. HEAVY 모드로 추론
        4. 상위 2개 시나리오 반환
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 시나리오 준비
            if request.scenarios:
                scenarios = request.scenarios
                self.logger.info(f"제공된 시나리오 {len(scenarios)}개 사용")
            else:
                # 시나리오가 없으면 기본 생성
                scenarios = self._generate_default_scenarios(request.text)
                self.logger.info(f"기본 시나리오 {len(scenarios)}개 생성")
            
            # 윤리적 딜레마 분석
            result = await self.inference_system.analyze_ethical_dilemma(scenarios)
            
            # 응답 구성
            top_scenarios = []
            for scenario_data in result.get('selected_scenarios', [])[:2]:
                top_scenarios.append(ScenarioAnalysis(
                    scenario=scenario_data.get('original_scenario', ''),
                    score=scenario_data['analysis'].get('integrated_score', 0),
                    ethical_analysis=scenario_data.get('ethics_analysis', {}),
                    utility_score=scenario_data.get('utility_score', 0),
                    regret_potential=scenario_data.get('regret_potential', 0)
                ))
            
            response = MCPResponse(
                top_scenarios=top_scenarios,
                recommendation=result.get('recommendation', '분석 실패'),
                total_evaluated=result.get('total_evaluated', 0),
                processing_time=asyncio.get_event_loop().time() - start_time,
                metadata=result.get('metadata', {})
            )
            
            self.logger.info(f"분석 완료: {response.processing_time:.2f}초")
            
            return response
            
        except Exception as e:
            self.logger.error(f"요청 처리 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_default_scenarios(self, text: str) -> List[str]:
        """기본 시나리오 생성
        
        LLM이 제공하지 않은 경우 기본 시나리오 생성
        """
        base_scenarios = [
            f"적극적 대응: {text} - 즉시 강력한 조치를 취한다",
            f"중도적 접근: {text} - 신중하게 상황을 평가하고 균형잡힌 대응을 한다",
            f"보수적 대응: {text} - 최소한의 개입으로 상황을 관찰한다"
        ]
        
        return base_scenarios
    
    async def start(self):
        """서버 시작"""
        # 시스템 초기화
        await self.initialize()
        
        # uvicorn 서버 시작
        config = uvicorn.Config(
            app=self.app,
            host=self.config['host'],
            port=self.config['port'],
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        self.logger.info(f"MCP 서버 시작: http://{self.config['host']}:{self.config['port']}")
        self.logger.info(f"MCP 매니페스트: http://{self.config['host']}:{self.config['port']}/mcp/manifest")
        
        await server.serve()


# MCP 설정 파일 생성
def create_mcp_config():
    """MCP 설정 파일 생성 (Claude 연동용)"""
    
    config = {
        "mcpServers": {
            "red-heart-ethics": {
                "command": "python",
                "args": ["/mnt/c/large_project/linux_red_heart/mcp_server.py"],
                "env": {
                    "PYTHONPATH": "/mnt/c/large_project/linux_red_heart"
                },
                "url": "http://localhost:8765",
                "description": "Red Heart AI 윤리적 의사결정 지원 시스템"
            }
        }
    }
    
    # 설정 파일 저장
    config_path = Path.home() / ".config" / "claude" / "mcp_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"MCP 설정 파일 생성: {config_path}")
    print("Claude Desktop에서 이 서버를 사용할 수 있습니다.")
    
    return config_path


# CLI 엔트리포인트
async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Red Heart MCP Server')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create MCP config file for Claude')
    parser.add_argument('--port', type=int, default=8765,
                       help='Server port (default: 8765)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if args.create_config:
        # MCP 설정 파일 생성
        create_mcp_config()
        print("\nMCP 서버를 시작하려면:")
        print("  python mcp_server.py")
    else:
        # 서버 시작
        server = RedHeartMCPServer()
        if args.port != 8765:
            server.config['port'] = args.port
        
        await server.start()


if __name__ == "__main__":
    # 서버 실행
    asyncio.run(main())