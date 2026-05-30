import faiss
import numpy as np
import os
from pathlib import Path
import json

class RAGDatabase:
    def __init__(self, db_path="rag_database"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.index_path = self.db_path / "faiss_index.bin"
        self.metadata_path = self.db_path / "metadata.json"
        
        # 加载或创建索引
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(768)  # all-mpnet-base-v2输出维度
            self.metadata = []

    def add_document(self, chunks, doc_type, file_path):
        """添加文档块到向量库"""
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        self.index.add(embeddings)
        
        # 保存元数据
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "chunk_id": len(self.metadata) + i,
                "content": chunk,
                "doc_type": doc_type,
                "file_path": str(file_path),
                "embedding_idx": self.index.ntotal - len(chunks) + i
            })

    def save(self):
        """保存索引和元数据到磁盘"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def search(self, query, top_k=5):
        """搜索相似文档块"""
        query_emb = self.embedder.encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "content": self.metadata[idx]["content"],
                    "distance": distances[0][i],
                    "source": self.metadata[idx]["file_path"]
                })
        return results
