# Development Guidelines

## Core Principles
- All code modifications and updates must leverage the advanced thinking capabilities of Claude 4 Extended Thinking to the fullest extent.
- If a solution cannot be resolved at a basic knowledge level, immediately conduct web research to investigate related papers, latest open-source knowledge, and best practices for application.
- Never include fallback processing, graceful degradation, mock values, dummy data, or simplification logic that could cause structural impurity or learning contamination in the code.
- When fixing issues, thoroughly review not just a single line of code, but the entire context above and below the line, and meticulously examine all interacting modules and functions to ensure consistency and integration.
- Even for already implemented features, thoroughly verify that no omissions or deficiencies exist. Prevent functional gaps.

## Resource Constraints
- Optimize based on the GPU 8GB VRAM limit.
- If VRAM limit is exceeded, actively attempt stable processing through RAM swapping and memory optimization.

## Quality Considerations
- Prioritize code quality, performance, consistency, inter-module coherence, and computational accuracy.

## Language and Development Approach
- Always work in Korean, thinking deeply and thoroughly.
- Carefully check and adhere to project rules throughout the development process.

## Dependency Management Rules
- **절대 금지**: 사용자 허가 없이 패키지 설치 (`pip install`, `npm install`, `conda install` 등)
- 모든 의존성 설치는 반드시 사용자에게 먼저 확인받아야 함
- 패키지 설치가 필요한 경우, 설치 명령어를 제시하고 사용자의 명시적 승인을 기다릴 것
- venv 환경에 특히 주의 - 무단 설치 시 환경 파괴 위험