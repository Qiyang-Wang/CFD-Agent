import requests
import torch
# 测试从ollama安装的bge模型  ollama pull bge-m3
# Ollama API 配置（默认本地服务地址）
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "bge-m3"

def get_embeddings(sentences):
    """通过 Ollama API 获取文本嵌入（逐个处理句子）"""
    embeddings = []
    for sentence in sentences:
        payload = {
            "model": MODEL_NAME,
            "prompt": sentence  # 单个句子输入
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()  # 检查请求是否成功
        # 直接获取 embedding 字段，而非 embeddings 数组
        embeddings.append(response.json()["embedding"])
    return embeddings

# 测试嵌入生成
sentences = [
    "Turbulence model: SST k-omega with wall functions",
    "Finite volume discretization using second-order upwind scheme",
    "Mesh independence study with 1.2 million hexahedral cells"
]

# 获取嵌入
embeddings = get_embeddings(sentences)
embeddings = torch.tensor(embeddings)  # 转换为 tensor 方便计算相似度

# 快速相似度检查（修复1D张量转置问题）
# 通过 unsqueeze(0) 将1D张量转为2D矩阵后再转置
# 修复：添加.item()将张量转换为标量后再格式化
print(f"Similarity 0-1: {(embeddings[0].unsqueeze(0) @ embeddings[1].unsqueeze(0).mT).item():.3f}")
print(f"Similarity 0-2: {(embeddings[0].unsqueeze(0) @ embeddings[2].unsqueeze(0).mT).item():.3f}")