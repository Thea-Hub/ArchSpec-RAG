import json
import re
import torch
import chromadb
from pathlib import Path
from typing import List, Dict
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    PromptTemplate, Settings,
    VectorStoreIndex, StorageContext, load_index_from_storage
)
from transformers import BitsAndBytesConfig
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
import os
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

# ================== 配置 ==================
class Config:
    GENERATE_MODE = "qwen3api"
    EMBED_MODEL_PATH = r"G:\project2026\project3_architect_chat\demo_21\model\EMBED_MODEL"
    LLM_MODEL_PATH = r"G:\project2026\项目1_中医临床智能诊疗系统\model\Qwen1.5-7B-Chat"
    RERANK_MODEL_PATH = r"E:\PythonNotebook\juke ai\work project\l2\day20-RAG+微调实现智能专家系统（部署测试）\day20-RAG+微调实现智能专家系统（部署测试）\demo_20\model\RERANK_MODEL"

    DATA_DIR = r"E:\PythonNotebook\juke ai\work project\l2\day19-RAG+微调实现智能专家系统（方案数据篇）\day19-RAG+微调实现智能专家系统（方案数据篇）\demo_19\data03"
    VECTOR_DB_DIR = "./chroma_db_final"
    PERSIST_DIR = "./storage_final"

    COLLECTION_NAME = "building_code"
    TOP_K = 10
    RERANK_TOP_K = 3
    SIMILARITY_CUTOFF = 0.6
    MIN_RERANK_SCORE = 0.4
    HYBRID_ALPHA = 0.5

# ================== 提示词 ==================
QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是专业建筑防火规范专家，只用中文，禁止英文、特殊符号。\n"
    "【输出格式强制绑定，必须严格执行】：\n"
    "1. 结论：直接给出最终答案（必须包含数值+单位，无对应数值写“无明确数值”）\n"
    "2. 依据：引用规范全称、编号、表号（例：建筑设计防火规范GB50016-2018 表5.5.17）\n"
    "3. 匹配说明：简要写清建筑类型、耐火等级、走道位置如何对应条文，不扩展、不编造\n"
    "【严格约束】：\n"
    "- 只使用提供的法规条文，严禁编造、自创规则、估算、推导公式\n"
    "- 严格区分厂房/仓库/公共建筑/住宅，不混用条文\n"
    "- 只回答用户问题相关内容，不扩展疏散宽度、防火分区等无关信息\n"
    "- 数值严格按原文，无原文定值如实说明，不许脑补\n"
    "相关法规条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
response_template = PromptTemplate(QA_TEMPLATE)

# ===========================================================================
# 🔥 全局独立函数：init_models()  【必须有！】
# ===========================================================================
def init_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_PATH, device=device)

    llm = None
    if Config.GENERATE_MODE == "local":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        llm = HuggingFaceLLM(
            model_name=Config.LLM_MODEL_PATH,
            tokenizer_name=Config.LLM_MODEL_PATH,
            model_kwargs={"trust_remote_code": True, "quantization_config": quantization_config},
            tokenizer_kwargs={"trust_remote_code": True},
            generate_kwargs={"temperature": 0.1},
            device_map="auto" if device == "cuda" else None
        )

    elif Config.GENERATE_MODE == "qwen3api":
        llm = DashScope(
            model_name=DashScopeGenerationModels.QWEN_MAX,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
        )

    reranker = SentenceTransformerRerank(
        model=Config.RERANK_MODEL_PATH,
        top_n=Config.RERANK_TOP_K
    )

    Settings.embed_model = embed_model
    Settings.llm = llm
    return embed_model, llm, reranker

# ===========================================================================
# 🔥 RAG 类
# ===========================================================================
class BuildingCodeRAG:
    def __init__(self):
        print("🔹 初始化 RAG 系统...")
        self.embed_model, self.llm, self.reranker = init_models()
        self.raw_data = load_and_validate_json_files(Config.DATA_DIR)
        self.nodes = create_nodes(self.raw_data)
        self.index = init_vector_store(self.nodes)

        self.retriever = self.index.as_retriever(
            similarity_top_k=Config.TOP_K,
            vector_store_query_mode="hybrid",
            alpha=Config.HYBRID_ALPHA
        )

        from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
        self.response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            text_qa_template=response_template
        )

    def query(self, user_query: str) -> str:
        q = preprocess_query(user_query)
        q = enhance_query_with_reasoning(q)
        nodes = self.retriever.retrieve(q)
        nodes = filter_by_building_type(user_query, nodes)
        nodes = self.reranker.postprocess_nodes(nodes)
        context = "\n".join([n.get_content() for n in nodes])
        prompt = response_template.format(context_str=context, query_str=user_query)
        response = self.llm.complete(prompt)
        return str(response)

# ================== 文档解析（含表格转文本） ==================
def convert_table_to_text(table_dict: dict) -> str:
    parts = []
    if table_dict.get("标题"):
        parts.append(f"标题：{table_dict['标题']}")
    if table_dict.get("续表标题"):
        parts.append(f"续表标题：{table_dict['续表标题']}")
    content = table_dict.get("内容", [])
    for item in content:
        for row_label, row_data in item.items():
            parts.append(f"行标签：{row_label}")
            for col_header, value in row_data.items():
                parts.append(f"  {col_header}：{value}")
    if table_dict.get("注释"):
        parts.append(f"注释：{table_dict['注释']}")
    return "\n".join(parts)

def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    json_files = list(Path(data_dir).glob("*.json"))
    all_data = []
    clause_id_records = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                key = list(entry.keys())[0]
                value = entry[key]
                match = re.match(r'^(.*?)\s+(\d+\.\d+(?:\.\d+)?)$', key)
                spec_name = match.group(1) if match else "未知规范"
                clause_id = match.group(2) if match else key
                text_parts = []
                if "正文" in value:
                    text_parts.append(value["正文"])
                if "表格" in value and value["表格"]:
                    text_parts.append("\n" + convert_table_to_text(value["表格"]))
                full_text = "\n".join(text_parts).strip()
                if not full_text:
                    continue
                full_title = value.get("表格", {}).get("标题", spec_name)
                key_unique = f"{json_file.name}::{clause_id}"
                if key_unique in clause_id_records:
                    continue
                clause_id_records[key_unique] = True
                all_data.append({
                    "content": {"clause_id": clause_id, "full_title": full_title, "content": full_text},
                    "metadata": {"source": json_file.name}
                })
    return all_data

# ================== 文本切块 ==================
def create_nodes(raw_data: List[Dict], chunk_size=512, chunk_overlap=64) -> List[TextNode]:
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    nodes = []
    for entry in raw_data:
        c = entry["content"]
        full_text = c["content"]
        chunks = splitter.split_text(full_text)
        for idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10:
                continue
            node = TextNode(
                text=chunk.strip(),
                metadata={
                    "clause_id": c["clause_id"],
                    "full_title": c["full_title"],
                    "source": entry["metadata"]["source"],
                    "chunk_index": idx,
                    "chunk_total": len(chunks)
                }
            )
            nodes.append(node)
    return nodes

# ================== 向量库 ==================
def init_vector_store(nodes=None):
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    if Path(Config.PERSIST_DIR).exists() and any(Path(Config.PERSIST_DIR).iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR, vector_store=vector_store)
        index = load_index_from_storage(storage_context)
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        index.storage_context.persist(Config.PERSIST_DIR)
    return index

# ================== 建筑类型识别 ==================
def infer_building_type(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["厂房", "车间"]):
        return "厂房"
    if any(w in q for w in ["仓库", "库房", "储存"]):
        return "仓库"
    if any(w in q for w in ["住宅", "居住", "公寓", "宿舍"]):
        return "住宅"
    if any(w in q for w in ["医院", "疗养", "养老"]):
        return "医疗建筑"
    if any(w in q for w in ["学校", "教学", "教室"]):
        return "教育建筑"
    return "公共建筑"

# ================== 预处理 ==================
def preprocess_query(query: str) -> str:
    stop_words = ["请问", "我想知道", "帮我查", "你好", "谢谢"]
    q = query.strip()
    for w in stop_words:
        q = q.replace(w, "")
    # ✅ 修复：\u9fa5 而不是 \9fa5
    q = re.sub(r'[^\u4e00-\u9fa50-9\s]', '', q)
    return q.strip()

def enhance_query_with_reasoning(query: str) -> str:
    return f"建筑设计防火规范 {query}".strip()

# ================== 建筑类型过滤 ==================
def filter_by_building_type(query: str, context_nodes: list) -> list:
    if not context_nodes:
        return []
    build_type = infer_building_type(query)
    filtered_nodes = []
    for node in context_nodes:
        cid = node.metadata.get("clause_id", "")
        if build_type == "住宅" and "5.5.17" in cid:
            continue
        if build_type != "住宅" and "5.5.21" in cid:
            continue
        filtered_nodes.append(node)
    return filtered_nodes

# ================== 表格解析 ==================
def text_to_streamlit_table(text: str):
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return text

    rows = []
    current_row = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("行标签："):
            if current_row is not None:
                rows.append(current_row)
            row_content = stripped[len("行标签："):].strip()
            current_row = {}
            parts = row_content.split("，")
            for part in parts:
                if "：" in part:
                    key, val = part.split("：", 1)
                    key = key.strip()
                    val = val.strip()
                    if " " in key:
                        key = key.split()[-1].strip()
                    current_row[key] = val
        elif current_row is not None and (line.startswith(" ") or line.startswith("\t")):
            if "：" in stripped:
                key, val = stripped.split("：", 1)
                key = key.strip()
                val = val.strip()
                if " " in key:
                    key = key.split()[-1].strip()
                current_row[key] = val
    if current_row is not None:
        rows.append(current_row)

    if not rows:
        return text

    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())
    priority_cols = ["名称", "耐火等级", "位于两个安全出口之间", "位于袋形走道两侧或尽端"]
    ordered_cols = [col for col in priority_cols if col in all_columns] + sorted([col for col in all_columns if col not in priority_cols])

    df_dict = {}
    for col in ordered_cols:
        df_dict[col] = [row.get(col, "") for row in rows]
    return df_dict

# ================== 核心：获取RAG引擎 ==================
def get_rag_engine():
    embed_model, llm, reranker = init_models()
    raw = load_and_validate_json_files(Config.DATA_DIR)
    nodes = create_nodes(raw)
    index = init_vector_store(nodes)

    retriever = index.as_retriever(
        similarity_top_k=Config.TOP_K,
        vector_store_query_mode="hybrid",
        alpha=Config.HYBRID_ALPHA
    )

    from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=response_template
    )

    return retriever, reranker, response_synthesizer