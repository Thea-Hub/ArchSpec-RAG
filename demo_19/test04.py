# -*- coding: utf-8 -*-
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
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from transformers import BitsAndBytesConfig
import bitsandbytes
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

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

# ================== 配置区 ==================
class Config:
    EMBED_MODEL_PATH = r"G:\project2026\project3_architect_chat\demo_21\model\EMBED_MODEL"
    LLM_MODEL_PATH = r"G:\project2026\项目1_中医临床智能诊疗系统\model\Qwen1.5-7B-Chat"

    DATA_DIR = r"E:\PythonNotebook\juke ai\work project\l2\day19-RAG+微调实现智能专家系统（方案数据篇）\day19-RAG+微调实现智能专家系统（方案数据篇）\demo_19\data_final"  # 存放 JSON 文件的目录
    VECTOR_DB_DIR = "./chroma_db_2"
    PERSIST_DIR = "./storage_2"

    COLLECTION_NAME = "building_code"
    TOP_K = 3


# ================== 初始化模型 ==================
def init_models():
    # 检查 GPU
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        device_map = "auto"  # 仅在 GPU 可用时设置
    else:
        print("使用 CPU")
        device = "cpu"
        device_map = None  # 用于条件判断，不传给 LLM

    # Embedding 模型
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_PATH,
        device=device
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # 启用嵌套量化，在第一轮量化之后会进行第二轮量化，为每个参数额外节省 0.4 比特
        bnb_4bit_compute_dtype=torch.bfloat16,  # 更改量化模型的计算数据类型来加速训练
    )

    # LLM 构建参数（先构建基础参数）
    llm_kwargs = {
        "model_name": Config.LLM_MODEL_PATH,
        "tokenizer_name": Config.LLM_MODEL_PATH,
        "model_kwargs": {"trust_remote_code": True,
                         "quantization_config": quantization_config},  # 压缩LLM时打开
        "tokenizer_kwargs": {"trust_remote_code": True},
        "generate_kwargs": {"temperature": 0.1}
    }
    # 仅在 GPU 可用时添加 device_map
    if device_map is not None:
        llm_kwargs["device_map"] = device_map

    llm = HuggingFaceLLM(**llm_kwargs)

    Settings.embed_model = embed_model
    Settings.llm = llm

    # 验证
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")

    return embed_model, llm


# ================== 数据处理（从 JSON 加载，整合表格内容） ==================
def convert_table_to_text(table_dict: dict) -> str:
    """
    将表格内容（嵌套 JSON）转换为可读文本，供 RAG 检索使用。
    表格结构：
        {
            "标题": "...",
            "续表标题": "...",
            "内容": [
                {"行标签1": {"列头1": "值1", "列头2": "值2", ...}},
                ...
            ],
            "注释": "..."
        }
    输出文本格式：
        标题：xxx
        行标签1：
            列头1：值1
            列头2：值2
        ...
        注释：xxx
    """
    parts = []
    if table_dict.get("标题"):
        parts.append(f"标题：{table_dict['标题']}")
    if table_dict.get("续表标题"):
        parts.append(f"续表标题：{table_dict['续表标题']}")

    content = table_dict.get("内容", [])
    if content:
        for item in content:
            # 每个 item 是一个字典，键为行标签（如 "储存物品的火灾危险性类别：丙，厂房的耐火等级：一级"）
            for row_label, row_data in item.items():
                parts.append(f"行标签：{row_label}")
                for col_header, value in row_data.items():
                    parts.append(f"  {col_header}：{value}")

    if table_dict.get("注释"):
        parts.append(f"注释：{table_dict['注释']}")

    return "\n".join(parts)


def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证 JSON 建筑法规文件，整合正文和表格内容"""
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
                    # entry 结构：{"建筑设计防火规范 3.2.1": {"正文": "...", "表格": {...}}}
                    key = list(entry.keys())[0]
                    value = entry[key]

                    # 提取条文编号
                    match = re.match(r'^(.*?)\s+(\d+\.\d+(?:\.\d+)?)$', key)
                    if match:
                        spec_name = match.group(1)
                        clause_id = match.group(2)
                    else:
                        spec_name = "未知规范"
                        clause_id = key  # 整个键作为编号

                    # 构建文本：正文 + 表格内容
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

                    # 获取章节标题（若有，否则用规范名称）
                    full_title = value.get("表格", {}).get("标题", spec_name)
                    if not full_title:
                        full_title = spec_name

                    # 去重（基于来源文件和条文编号）
                    key_unique = f"{json_file.name}::{clause_id}"
                    if key_unique in clause_id_records:
                        print(f"⚠️  跳过重复条文：{key_unique}")
                        continue
                    clause_id_records[key_unique] = True

                    all_data.append({
                        "content": {
                            "clause_id": clause_id,
                            "full_title": full_title,
                            "content": full_text
                        },
                        "metadata": {
                            "source": json_file.name,
                            "content_type": "building_code",
                            "spec_name": spec_name
                        }
                    })

            except Exception as e:
                raise RuntimeError(f"❌ 加载文件 {json_file} 失败: {str(e)}")

    print(f"✅ 成功加载 {len(all_data)} 个建筑法规条文条目（已整合正文和表格）")
    return all_data


def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """创建建筑法规节点（绝对唯一ID）"""
    nodes = []
    id_counter = {}

    for entry in raw_data:
        code_dict = entry["content"]
        source_file = entry["metadata"]["source"]

        # 基础ID + 序号，确保唯一
        base_node_id = f"{source_file}::{code_dict['clause_id']}"
        if base_node_id in id_counter:
            id_counter[base_node_id] += 1
            unique_node_id = f"{base_node_id}::{id_counter[base_node_id]}"
        else:
            id_counter[base_node_id] = 0
            unique_node_id = base_node_id

        # 创建节点
        node = TextNode(
            text=code_dict["content"],
            id_=unique_node_id,
            metadata={
                "clause_id": code_dict["clause_id"],
                "full_title": code_dict["full_title"],
                "source_file": source_file,
                "content_type": "building_code",
                "base_id": base_node_id
            }
        )
        nodes.append(node)

    if nodes:
        print(f"✅ 生成 {len(nodes)} 个建筑法规节点（ID示例：{nodes[0].id_}）")
        duplicate_ids = [k for k, v in id_counter.items() if v > 0]
        if duplicate_ids:
            print(f"⚠️  自动处理重复ID：{duplicate_ids}")
    else:
        print("❌ 警告：未生成任何节点！")
    return nodes


# ================== 向量存储（兼容残缺文件） ==================
def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    """初始化向量存储（自动修复残缺文件）"""
    # 初始化Chroma
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 检查持久化文件是否完整
    persist_dir = Path(Config.PERSIST_DIR)
    docstore_file = persist_dir / "docstore.json"
    need_rebuild = False

    # 判断是否需要重新构建索引
    if chroma_collection.count() == 0 and nodes and len(nodes) > 0:
        need_rebuild = True
    elif persist_dir.exists() and not docstore_file.exists():
        print(f"⚠️  检测到持久化文件缺失，删除残缺目录后重新构建...")
        shutil.rmtree(Config.VECTOR_DB_DIR)
        shutil.rmtree(Config.PERSIST_DIR)
        need_rebuild = True

    # 构建/加载索引
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    if need_rebuild:
        print(f"🚀 创建新索引（{len(nodes)}个节点）...")
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # 持久化
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        print("📂 加载已有索引...")
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=Config.PERSIST_DIR,
                vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
            )
        except FileNotFoundError:
            print("⚠️  持久化文件缺失，使用空上下文...")

        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context
        )

    # 验证存储
    print("\n📊 存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"文档节点数：{doc_count}")

    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        sample_node = storage_context.docstore.get_document(sample_key)
        print(f"示例节点：{sample_node.metadata['clause_id']} - {sample_node.text[:50]}...")
    else:
        print("❌ 警告：文档存储为空！")

    return index


# ================== 主程序（完整交互） ==================
def main():
    # 定义日志配置
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # 初始化模型
    print("🔧 初始化模型...")
    embed_model, llm = init_models()

    # 使用LlamaDebugHandler构建事件回溯器，以追踪LlamaIndex执行过程中发生的事件
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager

    # 读取文档
    vector_db_path = Path(Config.VECTOR_DB_DIR)

    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        print("✅ 向量库已存在，直接加载...")
        nodes = None  # 不读数据
    else:
        print("🔍 向量库不存在，开始构建...")
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        nodes = create_nodes(raw_data)

    # 初始化向量存储
    print("\n📚 初始化向量存储...")
    start_time = time.time()
    index = init_vector_store(nodes)
    print(f"✅ 索引加载完成，耗时：{time.time() - start_time:.2f}s")

    # 构建自定义查询引擎（最适合法规场景）
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=Config.TOP_K,
        vector_store_query_mode="hybrid",
        alpha=0.7,  # 稠密权重
    )

    # 响应合成器
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=response_template,  # 你的法规提示词
    )

    # 查询引擎（带相似度过滤）
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.6)
        ]
    )

    # 交互查询
    print("\n===== 建筑法规智能查询助手 🎯 =====")
    print("💡 示例问题：单层甲级厂房一级防火分区面积是多少？")
    print("💡 输入 q 退出\n")

    while True:
        question = input("请输入你的问题: ")
        if question.lower() == 'q':
            print("👋 退出助手，再见！")
            break

        # 执行查询
        try:
            print("\n🤔 正在检索法规条文...")
            response = query_engine.query(question)

            # 显示回答
            print(f"\n✅ 智能回答：\n{response.response}")

            # 显示依据
            print("\n📖 参考依据：")
            for idx, node in enumerate(response.source_nodes, 1):
                meta = node.metadata
                print(f"\n[{idx}] 条文编号：{meta['clause_id']}")
                print(f"   章节标题：{meta['full_title']}")
                print(f"   来源文件：{meta['source_file']}")
                print(f"   相关度：{node.score:.4f}")
                print(f"   条文内容：{node.text[:200]}..." if len(node.text) > 200 else f"   条文内容：{node.text}")

        except Exception as e:
            print(f"\n❌ 查询出错：{str(e)}")
            print("💡 建议检查问题描述或重新运行程序")
    # 过程追踪
    # get_llm_inputs_outputs 返回每个LLM调用的开始/结束事件
    event_pairs = llama_debug.get_llm_inputs_outputs()

    # print(event_pairs[0][1].payload.keys()) # 输出事件结束时所有相关的属性

    # 输出 Promt 构建过程
    print(event_pairs[0][1].payload["formatted_prompt"])

if __name__ == "__main__":
    main()