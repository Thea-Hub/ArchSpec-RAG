# =====================含前端streamlit终版（第一阶段）======================
import logging
import sys
import os
import json
import time
import re
import shutil
from pathlib import Path
from typing import List, Dict
import torch
import chromadb
import streamlit as st
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, \
    StorageContext, QueryBundle, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from transformers import BitsAndBytesConfig
import bitsandbytes
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

# ================== Streamlit 页面配置（仅界面） ==================
st.set_page_config(page_title="建筑防火规范专家", page_icon="🏗️", layout="wide")
st.title("🏗️ 建筑设计防火规范智能查询系统")
st.markdown("---")

# 定义提示词
QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是专业建筑防火规范专家，只使用中文，禁止出现任何英文、符号。\n"
    "回答规则：\n"
    "1. 只根据下面的法规条文回答，**不许自己编造任何数据**。\n"
    "2. 严格区分【厂房】和【仓库】，绝不混淆。\n"
    "3. 只复述条文里的明确数值，不许解释、不许推导、不许自创数值。\n"
    "4. 如果问的是厂房，只看厂房条文；问仓库只看仓库条文。\n"
    "5. 不清楚就直接按原文回答，不乱说、不脑补。\n"
    "相关法规条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
response_template = PromptTemplate(QA_TEMPLATE)


# ================== 配置区（完全和你一样） ==================
class Config:
    EMBED_MODEL_PATH = r"G:\project2026\project3_architect_chat\demo_21\model\EMBED_MODEL"
    LLM_MODEL_PATH = r"G:\project2026\项目1_中医临床智能诊疗系统\model\Qwen1.5-7B-Chat"
    DATA_DIR = r"E:\PythonNotebook\juke ai\work project\l2\day19-RAG+微调实现智能专家系统（方案数据篇）\day19-RAG+微调实现智能专家系统（方案数据篇）\demo_19\data03"
    VECTOR_DB_DIR = "./chroma_db_2"
    PERSIST_DIR = "./storage_2"
    COLLECTION_NAME = "building_code"
    TOP_K = 3


# ================== 初始化模型（完全和你一样） ==================
@st.cache_resource(show_spinner="正在加载模型...")
def init_models():
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        device_map = "auto"
    else:
        print("使用 CPU")
        device = "cpu"
        device_map = None

    embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_PATH, device=device)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm_kwargs = {
        "model_name": Config.LLM_MODEL_PATH,
        "tokenizer_name": Config.LLM_MODEL_PATH,
        "model_kwargs": {"trust_remote_code": True, "quantization_config": quantization_config},
        "tokenizer_kwargs": {"trust_remote_code": True},
        "generate_kwargs": {"temperature": 0.1}
    }
    if device_map is not None:
        llm_kwargs["device_map"] = device_map

    llm = HuggingFaceLLM(**llm_kwargs)
    Settings.embed_model = embed_model
    Settings.llm = llm
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")
    return embed_model, llm


# ================== 数据处理（完全和你一样） ==================
def convert_table_to_text(table_dict: dict) -> str:
    parts = []
    if table_dict.get("标题"):
        parts.append(f"标题：{table_dict['标题']}")
    if table_dict.get("续表标题"):
        parts.append(f"续表标题：{table_dict['续表标题']}")
    content = table_dict.get("内容", [])
    if content:
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
    assert json_files, f"❌ 未找到 JSON 文件于 {data_dir}"
    all_data = []
    clause_id_records = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"JSON 文件 {json_file.name} 根节点应为列表")
                print(f"\n📄 处理文件：{json_file.name}，共 {len(data)} 个条文")
                for entry in data:
                    key = list(entry.keys())[0]
                    value = entry[key]
                    match = re.match(r'^(.*?)\s+(\d+\.\d+(?:\.\d+)?)$', key)
                    if match:
                        spec_name = match.group(1)
                        clause_id = match.group(2)
                    else:
                        spec_name = "未知规范"
                        clause_id = key
                    text_parts = []
                    if "正文" in value:
                        text_parts.append(value["正文"])
                    if "表格" in value and value["表格"]:
                        table_text = convert_table_to_text(value["表格"])
                        text_parts.append("\n" + table_text)
                    full_text = "\n".join(text_parts).strip()
                    if not full_text:
                        print(f"⚠️  条文 {clause_id} 内容为空，跳过")
                        continue
                    full_title = value.get("表格", {}).get("标题", spec_name)
                    if not full_title:
                        full_title = spec_name
                    key_unique = f"{json_file.name}::{clause_id}"
                    if key_unique in clause_id_records:
                        print(f"⚠️  跳过重复条文：{key_unique}")
                        continue
                    clause_id_records[key_unique] = True
                    all_data.append({
                        "content": {"clause_id": clause_id, "full_title": full_title, "content": full_text},
                        "metadata": {"source": json_file.name, "content_type": "building_code", "spec_name": spec_name}
                    })
            except Exception as e:
                raise RuntimeError(f"❌ 加载文件 {json_file} 失败: {str(e)}")
    print(f"✅ 成功加载 {len(all_data)} 个建筑法规条文条目")
    return all_data


def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    nodes = []
    id_counter = {}
    for entry in raw_data:
        code_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        base_node_id = f"{source_file}::{code_dict['clause_id']}"
        if base_node_id in id_counter:
            id_counter[base_node_id] += 1
            unique_node_id = f"{base_node_id}::{id_counter[base_node_id]}"
        else:
            id_counter[base_node_id] = 0
            unique_node_id = base_node_id
        node = TextNode(
            text=code_dict["content"], id_=unique_node_id,
            metadata={
                "clause_id": code_dict["clause_id"], "full_title": code_dict["full_title"],
                "source_file": source_file, "content_type": "building_code", "base_id": base_node_id
            }
        )
        nodes.append(node)
    if nodes:
        print(f"✅ 生成 {len(nodes)} 个建筑法规节点")
    else:
        print("❌ 警告：未生成任何节点！")
    return nodes


# ================== 向量存储（完全和你一样） ==================
def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(name=Config.COLLECTION_NAME,
                                                               metadata={"hnsw:space": "cosine"})
    persist_dir = Path(Config.PERSIST_DIR)
    docstore_file = persist_dir / "docstore.json"
    need_rebuild = False
    if chroma_collection.count() == 0 and nodes and len(nodes) > 0:
        need_rebuild = True
    elif persist_dir.exists() and not docstore_file.exists():
        print(f"⚠️  检测到持久化文件缺失，删除残缺目录后重新构建...")
        shutil.rmtree(Config.VECTOR_DB_DIR)
        shutil.rmtree(Config.PERSIST_DIR)
        need_rebuild = True
    storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=chroma_collection))
    if need_rebuild:
        print(f"🚀 创建新索引（{len(nodes)}个节点）...")
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        print("📂 加载已有索引...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR,
                                                           vector_store=ChromaVectorStore(
                                                               chroma_collection=chroma_collection))
        except FileNotFoundError:
            print("⚠️  持久化文件缺失，使用空上下文...")
        index = VectorStoreIndex.from_vector_store(storage_context.vector_store, storage_context=storage_context)
    print("\n📊 存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"文档节点数：{doc_count}")
    return index

# ================== 【终极修复：支持多行表格，正确解析每一行】 ==================
def text_to_streamlit_table(text: str):
    """
    将规范条文中的表格文本转换为 Pandas DataFrame 兼容的字典格式。
    支持多行表格（多个“行标签：”），每行可包含多个键值对（用中文逗号分隔）。
    返回格式：{列名: [值1, 值2, ...]} 或 原始文本（当无表格时）
    """
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return text

    # 提取标题/续表标题等元数据（可选，暂不处理，只做表格主体）
    # 识别所有以“行标签：”开头的行作为新行的起点
    rows = []  # 存储每行的字典
    current_row = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # 新行开始：以“行标签：”开头
        if stripped.startswith("行标签："):
            # 保存上一行
            if current_row is not None:
                rows.append(current_row)
            # 解析新行
            row_content = stripped[len("行标签："):].strip()
            current_row = {}
            # 按中文逗号分割多个键值对
            parts = row_content.split("，")
            for part in parts:
                if "：" in part:
                    key, val = part.split("：", 1)
                    key = key.strip()
                    val = val.strip()
                    # 简化列名（取最后一段，避免过长）
                    if " " in key:
                        key = key.split()[-1].strip()
                    current_row[key] = val
                else:
                    # 异常情况，整行作为值，列名用“描述”
                    current_row["描述"] = row_content
        # 缩进行：以空格或制表符开头（属于当前行的附加列）
        elif current_row is not None and (line.startswith(" ") or line.startswith("\t") or line.startswith("  ")):
            kv_line = stripped
            if "：" in kv_line:
                key, val = kv_line.split("：", 1)
                key = key.strip()
                val = val.strip()
                if " " in key:
                    key = key.split()[-1].strip()
                # 如果该列已存在（理论上不会，但以防覆盖）
                current_row[key] = val
        # 忽略空行或无关行
        i += 1
    # 循环结束后添加最后一行
    if current_row is not None:
        rows.append(current_row)

    # 如果没有解析到任何行，返回原始文本
    if not rows:
        return text

    # 收集所有列名（所有行字典的键的并集）
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())
    # 排序使列顺序固定（可选，将“规范条件”或“生产的火灾危险性类别”等放在前面）
    # 优先显示常见列
    priority_cols = ["生产的火灾危险性类别", "储存物品的火灾危险性类别", "耐火等级", "最多允许层数", "每个防火分区的最大允许建筑面积(㎡)"]
    ordered_cols = [col for col in priority_cols if col in all_columns] + sorted(
        [col for col in all_columns if col not in priority_cols])

    # 构建 DataFrame 字典：{列名: [值1, 值2, ...]}
    df_dict = {}
    for col in ordered_cols:
        df_dict[col] = [row.get(col, "") for row in rows]

    return df_dict


# 主程序中调用 st.dataframe 时使用 width='stretch' 替换 use_container_width
# 原代码：
# st.dataframe(table_data, use_container_width=True)
# 改为：
# st.dataframe(table_data, width='stretch')


# ================== 主程序（界面替换，逻辑不变） ==================
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    embed_model, llm = init_models()
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([llama_debug])

    vector_db_path = Path(Config.VECTOR_DB_DIR)
    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        st.success("✅ 向量库已存在，直接加载")
        nodes = None
    else:
        st.warning("🔍 向量库不存在，开始构建...")
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        nodes = create_nodes(raw_data)

    with st.spinner("📚 加载索引中..."):
        index = init_vector_store(nodes)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=Config.TOP_K, vector_store_query_mode="hybrid",
                                     alpha=0.7)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.SIMPLE_SUMMARIZE,
                                                    text_qa_template=response_template)
    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)]
    )

    # 界面输入框
    question = st.text_input("💬 请输入你的问题：", placeholder="例如：单层甲级厂房一级耐火等级的防火分区面积是多少？")

    if st.button("🚀 开始查询") and question:
        with st.spinner("🔍 正在检索法规条文..."):
            response = query_engine.query(question)

        st.subheader("✅ 智能回答（严格提取原文）")
        st.success(response.response)

        st.markdown("---")
        st.subheader("📖 参考依据（已修复显示错乱）")

        # 【关键修复】遍历前先打印，确认 source_nodes 本身是否正确
        import copy
        for idx, node_with_score in enumerate(copy.deepcopy(response.source_nodes), 1):
            # 解包：node_with_score = (node, score)
            node = node_with_score.node
            score = node_with_score.score
            meta = node.metadata

            # 调试打印（终端看）：确认检索到的真的是甲级条文
            print(f"\n=== 参考条文 {idx} ===")
            print(f"条款ID: {meta.get('clause_id')}")
            print(f"相似度: {score:.4f}")
            print(f"内容前100字: {node.text[:100]}...")

            # 在展示参考依据的循环内
            with st.expander(f"条文 {idx} | {meta['clause_id']} | 相关度 {score:.4f}"):
                st.write(f"**章节标题**：{meta.get('full_title', '未知')}")
                st.write(f"**来源文件**：{meta.get('source_file', '未知')}")
                table_data = text_to_streamlit_table(node.text)
                if isinstance(table_data, dict):
                    # 修复弃用警告：use_container_width -> width='stretch'
                    st.dataframe(table_data, width='stretch')
                else:
                    st.write(table_data)


if __name__ == "__main__":
    main()