#!/usr/bin/env python3
"""
임베딩 데이터를 1000개 단위 청크로 분할 저장 및 로드
메모리 효율적이고 빠른 접근을 위한 인덱스 맵 구조
"""

import json
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingChunkManager:
    """임베딩 청크 관리 클래스"""
    
    def __init__(self, base_dir: str = "claude_api_preprocessing/embedded", chunk_size: int = 1000):
        self.base_dir = Path(base_dir)
        self.chunk_size = chunk_size
        self.chunks_dir = self.base_dir / "chunks"
        self.metadata_file = self.base_dir / "metadata.json"
        self.index_map_file = self.base_dir / "index_map.json"
        
        # 디렉토리 생성
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시
        self._cache = {}
        self._metadata = None
        self._index_map = None
    
    def create_chunks_from_embedded_data(self, data: List[Dict], rebuild: bool = False):
        """
        임베딩 데이터를 청크로 분할하여 저장
        
        Args:
            data: 전체 임베딩 데이터 리스트
            rebuild: 기존 청크 삭제하고 재생성
        """
        if rebuild:
            logger.info("기존 청크 삭제 중...")
            for chunk_file in self.chunks_dir.glob("chunk_*.json"):
                chunk_file.unlink()
        
        total_items = len(data)
        num_chunks = (total_items + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"청크 생성 시작: {total_items}개 아이템 → {num_chunks}개 청크")
        
        metadata = {
            "total_items": total_items,
            "chunk_size": self.chunk_size,
            "num_chunks": num_chunks,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": []
        }
        
        index_map = {}  # item_id -> (chunk_idx, position_in_chunk)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_items)
            
            chunk_data = data[start_idx:end_idx]
            chunk_file = self.chunks_dir / f"chunk_{chunk_idx:05d}.json"
            
            # 청크 저장
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False)
            
            # 메타데이터 업데이트
            chunk_info = {
                "chunk_idx": chunk_idx,
                "file": chunk_file.name,
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "num_items": len(chunk_data),
                "has_embeddings": sum(1 for item in chunk_data if item.get("embedding") is not None)
            }
            metadata["chunks"].append(chunk_info)
            
            # 인덱스 맵 업데이트
            for pos, item in enumerate(chunk_data):
                if "id" in item:
                    index_map[item["id"]] = {
                        "chunk_idx": chunk_idx,
                        "position": pos,
                        "global_idx": start_idx + pos
                    }
            
            logger.info(f"청크 {chunk_idx}/{num_chunks-1} 저장: {len(chunk_data)}개 아이템")
        
        # 메타데이터 저장
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # 인덱스 맵 저장
        with open(self.index_map_file, 'w', encoding='utf-8') as f:
            json.dump(index_map, f, indent=2)
        
        logger.info(f"✅ 청크 생성 완료: {num_chunks}개 청크, {len(index_map)}개 인덱스")
        
        self._metadata = metadata
        self._index_map = index_map
        
        return metadata
    
    def load_metadata(self):
        """메타데이터 로드"""
        if self._metadata is None:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self._metadata = json.load(f)
            else:
                raise FileNotFoundError(f"메타데이터 파일 없음: {self.metadata_file}")
        return self._metadata
    
    def load_index_map(self):
        """인덱스 맵 로드"""
        if self._index_map is None:
            if self.index_map_file.exists():
                with open(self.index_map_file, 'r') as f:
                    self._index_map = json.load(f)
            else:
                raise FileNotFoundError(f"인덱스 맵 파일 없음: {self.index_map_file}")
        return self._index_map
    
    def load_chunk(self, chunk_idx: int, use_cache: bool = True) -> List[Dict]:
        """특정 청크 로드"""
        if use_cache and chunk_idx in self._cache:
            return self._cache[chunk_idx]
        
        chunk_file = self.chunks_dir / f"chunk_{chunk_idx:05d}.json"
        if not chunk_file.exists():
            raise FileNotFoundError(f"청크 파일 없음: {chunk_file}")
        
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        
        if use_cache:
            # 캐시 크기 제한 (최대 5개 청크)
            if len(self._cache) >= 5:
                # LRU: 가장 오래된 것 제거
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[chunk_idx] = chunk_data
        
        return chunk_data
    
    def get_item_by_id(self, item_id: str) -> Optional[Dict]:
        """ID로 특정 아이템 로드"""
        index_map = self.load_index_map()
        
        if item_id not in index_map:
            return None
        
        info = index_map[item_id]
        chunk_data = self.load_chunk(info["chunk_idx"])
        return chunk_data[info["position"]]
    
    def get_items_by_range(self, start_idx: int, end_idx: int) -> List[Dict]:
        """인덱스 범위로 아이템들 로드"""
        metadata = self.load_metadata()
        items = []
        
        for chunk_info in metadata["chunks"]:
            # 범위에 해당하는 청크만 로드
            if chunk_info["end_idx"] < start_idx:
                continue
            if chunk_info["start_idx"] > end_idx:
                break
            
            chunk_data = self.load_chunk(chunk_info["chunk_idx"])
            
            # 청크 내에서 필요한 부분만 추출
            for local_idx, item in enumerate(chunk_data):
                global_idx = chunk_info["start_idx"] + local_idx
                if start_idx <= global_idx <= end_idx:
                    items.append(item)
        
        return items
    
    def get_batch_items(self, item_ids: List[str]) -> Dict[str, Dict]:
        """여러 ID에 대한 아이템들 배치 로드 (효율적)"""
        index_map = self.load_index_map()
        
        # 청크별로 그룹화
        chunks_to_load = {}
        for item_id in item_ids:
            if item_id in index_map:
                info = index_map[item_id]
                chunk_idx = info["chunk_idx"]
                if chunk_idx not in chunks_to_load:
                    chunks_to_load[chunk_idx] = []
                chunks_to_load[chunk_idx].append((item_id, info["position"]))
        
        # 청크별로 로드하여 결과 수집
        results = {}
        for chunk_idx, items_info in chunks_to_load.items():
            chunk_data = self.load_chunk(chunk_idx)
            for item_id, position in items_info:
                results[item_id] = chunk_data[position]
        
        return results
    
    def get_statistics(self) -> Dict:
        """청크 통계 정보"""
        metadata = self.load_metadata()
        
        total_embedded = sum(chunk["has_embeddings"] for chunk in metadata["chunks"])
        
        return {
            "total_items": metadata["total_items"],
            "total_chunks": metadata["num_chunks"],
            "chunk_size": metadata["chunk_size"],
            "total_embedded": total_embedded,
            "embedding_ratio": total_embedded / metadata["total_items"] if metadata["total_items"] > 0 else 0,
            "created_at": metadata["created_at"]
        }


def convert_existing_embedded_file(input_file: str, output_dir: str, chunk_size: int = 1000):
    """
    기존 대용량 임베딩 파일을 청크로 변환
    """
    logger.info(f"변환 시작: {input_file} → {output_dir}")
    
    manager = EmbeddingChunkManager(output_dir, chunk_size)
    
    # 스트리밍 방식으로 읽기
    logger.info("파일 읽기 시작...")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"로드 완료: {len(data)}개 아이템")
        
        # 청크로 변환
        manager.create_chunks_from_embedded_data(data, rebuild=True)
        
        # 통계 출력
        stats = manager.get_statistics()
        logger.info("변환 완료!")
        logger.info(f"  - 총 아이템: {stats['total_items']}")
        logger.info(f"  - 총 청크: {stats['total_chunks']}")
        logger.info(f"  - 임베딩 완료: {stats['total_embedded']} ({stats['embedding_ratio']*100:.1f}%)")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 에러: {e}")
        return False
    except Exception as e:
        logger.error(f"변환 실패: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # 테스트용
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "claude_api_preprocessing/embedded"
        convert_existing_embedded_file(input_file, output_dir)
    else:
        print("사용법: python embedding_chunker.py <input_json> [output_dir]")