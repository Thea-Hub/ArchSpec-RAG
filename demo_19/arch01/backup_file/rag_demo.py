# # ================== 建筑防火规范专家系统 —— 思维链终极版——有langgragh的agent ==================
# import logging
# import sys
# import os
# import json
# import re
# import shutil
# import torch
# import chromadb
# import streamlit as st
# from pathlib import Path
# from typing import List, Dict, Literal
# from llama_index.core.schema import TextNode
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import (
#     PromptTemplate, Settings,
#     VectorStoreIndex, StorageContext, load_index_from_storage
# )
# from transformers import BitsAndBytesConfig
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
# from llama_index.core.postprocessor import SentenceTransformerRerank
#
# # ================== LangGraph 可视化（仅后台保存） ==================
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from typing import TypedDict
#
# # ================== Streamlit 页面配置 ==================
# st.set_page_config(page_title="建筑防火规范专家", page_icon="🏗️", layout="wide")
# st.title("🏗️ 建筑设计防火规范智能查询系统（思维链推理版）")
# st.markdown("---")
#
# # ================== 【核心：思维链强制提示词】 ==================
# QA_TEMPLATE = (
#     "<|im_start|>system\n"
#     "你是专业建筑防火规范专家，只使用中文，禁止出现任何英文、符号。\n"
#     "【严格按以下思维链步骤执行，不许跳过】：\n"
#     "1. 先判断建筑类型：厂房/仓库/公共建筑/住宅/特殊场所\n"
#     "2. 提取关键条件：耐火等级、走道位置（两个安全出口之间/袋形走道尽端）、层数\n"
#     "3. 仅使用提供的法规条文，不编造、不扩展、不解释、不推导\n"
#     "4. 严格区分厂房、仓库、公共建筑，不混用\n"
#     "5. 按原文数值回答，不脑补\n"
#     "6. 严禁自行推导、严禁自创计算公式、严禁按比例估算，没有原文定值就直接说明无对应数值，不许编造规则。"
#     "7. 只回答袋形走道尽端/安全出口之间的固定疏散距离米数，不扩展疏散宽度、防火分区借用等无关内容。"
#     "相关法规条文：\n{context_str}\n<|im_end|>\n"
#     "<|im_start|>user\n{query_str}<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
# response_template = PromptTemplate(QA_TEMPLATE)
#
#
# # ================== 配置 ==================
# class Config:
#     EMBED_MODEL_PATH = r"G:\project2026\project3_architect_chat\demo_21\model\EMBED_MODEL"
#     LLM_MODEL_PATH = r"G:\project2026\项目1_中医临床智能诊疗系统\model\Qwen1.5-7B-Chat"
#     RERANK_MODEL_PATH = r"E:\PythonNotebook\juke ai\work project\l2\day20-RAG+微调实现智能专家系统（部署测试）\day20-RAG+微调实现智能专家系统（部署测试）\demo_20\model\RERANK_MODEL"
#
#     DATA_DIR = r"E:\PythonNotebook\juke ai\work project\l2\day19-RAG+微调实现智能专家系统（方案数据篇）\day19-RAG+微调实现智能专家系统（方案数据篇）\demo_19\data03"
#     VECTOR_DB_DIR = "./chroma_db_2"
#     PERSIST_DIR = "./storage_2"
#
#     COLLECTION_NAME = "building_code"
#     TOP_K = 10
#     RERANK_TOP_K = 3
#     SIMILARITY_CUTOFF = 0.6
#     MAX_REWRITE = 2
#
#
# # ================== 状态定义 ==================
# class AgentState(TypedDict):
#     query: str
#     rewritten_query: str
#     context: list
#     info_sufficient: bool
#     rewrite_count: int
#
#
# # ================== 模型加载 ==================
# @st.cache_resource(show_spinner="加载模型中...")
# def init_models():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_PATH, device=device)
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True, bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     llm = HuggingFaceLLM(
#         model_name=Config.LLM_MODEL_PATH,
#         tokenizer_name=Config.LLM_MODEL_PATH,
#         model_kwargs={"trust_remote_code": True, "quantization_config": quantization_config},
#         tokenizer_kwargs={"trust_remote_code": True},
#         generate_kwargs={"temperature": 0.1},
#         device_map="auto" if device == "cuda" else None
#     )
#     reranker = SentenceTransformerRerank(
#         model=Config.RERANK_MODEL_PATH,
#         top_n=Config.RERANK_TOP_K
#     )
#
#     Settings.embed_model = embed_model
#     Settings.llm = llm
#
#     return embed_model, llm,reranker
#
#
# # ================== 文档解析 ==================
# def convert_table_to_text(table_dict: dict) -> str:
#     parts = []
#     if table_dict.get("标题"):
#         parts.append(f"标题：{table_dict['标题']}")
#     if table_dict.get("续表标题"):
#         parts.append(f"续表标题：{table_dict['续表标题']}")
#     content = table_dict.get("内容", [])
#     for item in content:
#         for row_label, row_data in item.items():
#             parts.append(f"行标签：{row_label}")
#             for col_header, value in row_data.items():
#                 parts.append(f"  {col_header}：{value}")
#     if table_dict.get("注释"):
#         parts.append(f"注释：{table_dict['注释']}")
#     return "\n".join(parts)
#
#
# def load_and_validate_json_files(data_dir: str) -> List[Dict]:
#     json_files = list(Path(data_dir).glob("*.json"))
#     all_data = []
#     clause_id_records = {}
#     for json_file in json_files:
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             for entry in data:
#                 key = list(entry.keys())[0]
#                 value = entry[key]
#                 match = re.match(r'^(.*?)\s+(\d+\.\d+(?:\.\d+)?)$', key)
#                 spec_name = match.group(1) if match else "未知规范"
#                 clause_id = match.group(2) if match else key
#                 text_parts = []
#                 if "正文" in value:
#                     text_parts.append(value["正文"])
#                 if "表格" in value and value["表格"]:
#                     text_parts.append("\n" + convert_table_to_text(value["表格"]))
#                 full_text = "\n".join(text_parts).strip()
#                 if not full_text:
#                     continue
#                 full_title = value.get("表格", {}).get("标题", spec_name)
#                 key_unique = f"{json_file.name}::{clause_id}"
#                 if key_unique in clause_id_records:
#                     continue
#                 clause_id_records[key_unique] = True
#                 all_data.append({
#                     "content": {"clause_id": clause_id, "full_title": full_title, "content": full_text},
#                     "metadata": {"source": json_file.name}
#                 })
#     return all_data
#
#
# # ================== ✅ 修复：加入文本切块（trunk） ==================
# def create_nodes(raw_data: List[Dict], chunk_size=512, chunk_overlap=64) -> List[TextNode]:
#     from llama_index.core.node_parser import SentenceSplitter
#     splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
#     nodes = []
#     for entry in raw_data:
#         c = entry["content"]
#         full_text = c["content"]
#         chunks = splitter.split_text(full_text)
#         for idx, chunk in enumerate(chunks):
#             if len(chunk.strip()) < 10:
#                 continue
#             node = TextNode(
#                 text=chunk.strip(),
#                 metadata={
#                     "clause_id": c["clause_id"],
#                     "full_title": c["full_title"],
#                     "source": entry["metadata"]["source"],
#                     "chunk_index": idx,
#                     "chunk_total": len(chunks)
#                 }
#             )
#             nodes.append(node)
#     return nodes
#
#
# # ================== 向量库 ==================
# def init_vector_store(nodes=None):
#     chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
#     chroma_collection = chroma_client.get_or_create_collection(name=Config.COLLECTION_NAME)
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     if Path(Config.PERSIST_DIR).exists() and any(Path(Config.PERSIST_DIR).iterdir()):
#         storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR, vector_store=vector_store)
#         index = load_index_from_storage(storage_context)
#     else:
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)
#         index = VectorStoreIndex(nodes, storage_context=storage_context)
#         index.storage_context.persist(Config.PERSIST_DIR)
#     return index
#
#
# # ================== 【思维链核心：建筑类型推理】 ==================
# def infer_building_type(query: str) -> str:
#     """【通用】自动推断建筑大类，无硬编码"""
#     q = query.lower()
#
#     # 工业类
#     if any(w in q for w in ["厂房", "车间"]):
#         return "厂房"
#     if any(w in q for w in ["仓库", "库房", "储存"]):
#         return "仓库"
#
#     # 居住类
#     if any(w in q for w in ["住宅", "居住", "公寓", "宿舍"]):
#         return "住宅"
#
#     # 特殊公共类
#     if any(w in q for w in ["医院", "疗养", "养老"]):
#         return "医疗建筑"
#     if any(w in q for w in ["学校", "教学", "教室"]):
#         return "教育建筑"
#
#     # 普通公共类（兜底）
#     return "公共建筑"
#
#
# # ================== 【思维链核心：生成精准检索词】 ==================
# def enhance_query_with_reasoning(query: str) -> str:
#     """【通用】思维链增强查询：自动适配建筑类型，无硬编码"""
#     build_type = infer_building_type(query)
#     return f"建筑设计防火规范 {build_type} {query}".strip()
#
#
# # ================== 匿名化 + 预处理 + 规划 ==================
# def anonymize_query(query: str) -> str:
#     query = re.sub(r'\d{11}', '[电话]', query)
#     query = re.sub(r'[A-Za-z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[邮箱]', query)
#     query = re.sub(r'[一二三四五六七八九十A-Za-z0-9]+[项目广场苑区楼大厦]', '[项目]', query)
#     query = re.sub(r'[^\u4e00-\u9fa5\s]', '', query)
#     return query.strip()
#
#
# def preprocess_query(query: str) -> str:
#     synonym_map = {
#         "甲类": "甲级", "乙类": "乙级", "丙类": "丙级",
#         "耐火": "耐火等级", "防火": "防火分区",
#         "尽端走廊": "袋形走道尽端", "走到尽头": "袋形走道尽端",
#     }
#     q = query
#     for k, v in synonym_map.items():
#         q = q.replace(k, v)
#     for w in ["请问", "我想知道", "帮我查", "你好"]:
#         q = q.replace(w, "")
#     return q.strip()
#
#
# # ================== 可控评分 + 重写 + 路由 ==================
# def is_information_sufficient(query: str, context_nodes: list) -> bool:
#     """【通用】判断信息是否充足：不硬编码，自动匹配建筑类型"""
#     if not context_nodes:
#         return False
#
#     # 1. 通用：自动判断当前问题属于什么建筑
#     build_type = infer_building_type(query)
#
#     # 2. 通用：定义每种建筑必须在条文中出现的关键词
#     type_rules = {
#         "厂房": ["厂房"],
#         "仓库": ["仓库"],
#         "住宅": ["住宅", "居住", "宿舍"],
#         "医疗建筑": ["医院", "医疗", "疗养"],
#         "教育建筑": ["学校", "教学", "教室"],
#         "公共建筑": ["公共建筑", "办公", "商店", "旅馆", "餐饮", "展厅", "边防", "检查"]
#     }
#
#     must_have = type_rules[build_type]
#
#     # 3. 通用：只保留【真正属于该建筑类型的法规条文】
#     valid_nodes = []
#     for node in context_nodes:
#         text = node.text.lower()
#         if any(key in text for key in must_have):
#             valid_nodes.append(node)
#
#     # 4. 没有真正匹配的条文 → 信息不足
#     if not valid_nodes:
#         return False
#
#     # 5. 通用：核心疏散关键词校验（所有建筑通用）
#     core_terms = {"耐火等级", "袋形走道", "尽端", "疏散距离", "安全出口", "疏散"}
#     q_words = set(re.findall(r'[\u4e00-\u9fa5]+', query))
#     q_keys = {w for w in q_words if w in core_terms}
#
#     if not q_keys:
#         return len("\n".join(n.text for n in valid_nodes)) > 200
#
#     # 6. 通用：命中比例 + 相似度
#     hit = 0
#     for n in valid_nodes:
#         hit += sum(1 for k in q_keys if k in n.text)
#
#     ratio = hit / len(q_keys)
#     sim_ok = all(n.score >= Config.SIMILARITY_CUTOFF for n in valid_nodes)
#
#     return ratio >= 0.5 and sim_ok
#
#
# def rewrite_query(query: str) -> str:
#     """通用查询优化：去重、清理、补全关键词，无硬编码"""
#     new_q = query.strip()
#
#     # 通用：清理重复前缀（不会无限叠加）
#     new_q = re.sub(r'(建筑设计防火规范\s*){2,}', '建筑设计防火规范 ', new_q)
#     new_q = re.sub(r'(公共建筑\s*){2,}', '公共建筑 ', new_q)
#     new_q = re.sub(r'(厂房\s*){2,}', '厂房 ', new_q)
#     new_q = re.sub(r'(仓库\s*){2,}', '仓库 ', new_q)
#     new_q = re.sub(r'(住宅\s*){2,}', '住宅 ', new_q)
#
#     # 通用：清理乱码
#     new_q = re.sub(r'[A-Za-z0-9]{3,}', '', new_q)
#
#     # 通用：关键词标准化（所有建筑都适用）
#     if "耐火" in new_q and "等级" not in new_q:
#         new_q = new_q.replace("耐火", "耐火等级")
#     if "尽端" in new_q and "袋形走道" not in new_q:
#         new_q = new_q.replace("尽端", "袋形走道尽端")
#     if "走廊" in new_q:
#         new_q = new_q.replace("走廊", "走道")
#
#     return new_q.strip()
#
#
# def task_router(query: str) -> Literal["answer", "retrieve"]:
#     chat_words = ["你好", "谢谢", "再见", "在吗", "哈哈"]
#     if any(w in query for w in chat_words) or len(query) < 8:
#         return "answer"
#     build_words = ["厂房", "仓库", "耐火", "防火", "消防", "疏散", "规范", "面积", "边防", "检查"]
#     if any(w in query for w in build_words):
#         return "retrieve"
#     return "answer"
#
#
# # ================== 表格显示 ==================
# def text_to_streamlit_table(text: str):
#     lines = [l.strip() for l in text.splitlines() if l.strip()]
#     rows = []
#     current = None
#     for line in lines:
#         if line.startswith("行标签："):
#             if current:
#                 rows.append(current)
#             current = {}
#             parts = line[len("行标签："):].split("，")
#             for p in parts:
#                 if "：" in p:
#                     k, v = p.split("：", 1)
#                     current[k.strip()] = v.strip()
#         elif current and "：" in line:
#             k, v = line.split("：", 1)
#             current[k.strip()] = v.strip()
#     if current:
#         rows.append(current)
#     if not rows:
#         return text
#     cols = sorted({k for r in rows for k in r.keys()})
#     return {c: [r.get(c, "") for r in rows] for c in cols}
#
#
# # ================== 流程图（仅保存不显示） ==================
# def build_workflow():
#     workflow = StateGraph(AgentState)
#
#     def dummy_node(state):
#         return state
#
#     workflow.add_node("task_distribute", dummy_node)
#     workflow.add_node("retrieve", dummy_node)
#     workflow.add_node("check_info", dummy_node)
#     workflow.add_node("rewrite", dummy_node)
#     workflow.add_node("answer", dummy_node)
#     workflow.set_entry_point("task_distribute")
#     workflow.add_conditional_edges("task_distribute", lambda x: task_router(x["query"]),
#                                    {"answer": "answer", "retrieve": "retrieve"})
#     workflow.add_edge("retrieve", "check_info")
#     workflow.add_conditional_edges("check_info",
#                                    lambda x: "answer" if x["info_sufficient"] or x["rewrite_count"] >= 2 else "rewrite",
#                                    {"answer": "answer", "rewrite": "rewrite"})
#     workflow.add_edge("rewrite", "task_distribute")
#     workflow.add_edge("answer", END)
#     app = workflow.compile(checkpointer=MemorySaver())
#     try:
#         app.get_graph().draw_mermaid_png(output_file_path="../workflow.png")
#     except:
#         pass
#     return app
#
#
# # =============== ✅ 修复：过程追踪（防止None崩溃） =======================
# def trace_clause_status(query: str, nodes: list, step_name: str):
#     print(f"\n🔍 【追踪】{step_name}")
#     print(f"   当前查询：{query}")
#
#     # ✅ 修复：判断 None
#     if nodes is None:
#         print(f"   返回条文总数：0（节点为空）")
#         print("-" * 80)
#         return
#
#     print(f"   返回条文总数：{len(nodes)}")
#     found = False
#     for i, node in enumerate(nodes):
#         cid = node.node.metadata.get("clause_id", "")
#         score = node.score if hasattr(node, 'score') else "无"
#         text = node.node.text[:100].replace("\n", "")
#         if "5.5.21" in cid:
#             print(f"   ✅ 找到目标条文：{cid} | 相似度：{score}")
#             found = True
#         else:
#             print(f"   - 其他条文：{cid} | 相似度：{score}")
#     if not found:
#         print(f"   ❌ 5.5.21 未出现在当前节点列表中")
#     print("-" * 80)
#
#
# # ================== 终端日志 ==================
# def log_step(title, content=""):
#     print(f"\n==================================================")
#     print(f"📌 {title}")
#     if content:
#         print(f"▶ {content}")
#     print(f"==================================================")
#
#
# # ================== 主程序 ==================
# def main():
#     embed_model, llm, reranker = init_models()
#     if not Path(Config.VECTOR_DB_DIR).exists():
#         raw = load_and_validate_json_files(Config.DATA_DIR)
#         nodes = create_nodes(raw)
#     else:
#         nodes = None
#
#     with st.spinner("加载索引..."):
#         index = init_vector_store(nodes)
#
#     # 1. 基础检索器
#     retriever = VectorIndexRetriever(index=index, similarity_top_k=Config.TOP_K)
#
#     # 2. 后处理器：rerank + 相似度过滤（你漏掉的核心！）
#     post_processors = [
#         reranker,  # 你加的重排模型
#         SimilarityPostprocessor(similarity_cutoff=Config.SIMILARITY_CUTOFF)
#     ]
#
#     # 3. 响应合成
#     synth = get_response_synthesizer(
#         response_mode=ResponseMode.SIMPLE_SUMMARIZE,
#         text_qa_template=response_template
#     )
#
#     # 4. 最终查询引擎（把 rerank 绑进去）
#     query_engine = RetrieverQueryEngine(
#         retriever=retriever,
#         response_synthesizer=synth,
#         node_postprocessors=post_processors,  # ✅ 核心：你之前漏掉的！
#     )
#
#     try:
#         build_workflow()
#     except:
#         pass
#
#     query = st.text_input("请输入问题：", placeholder="例：边防检查站一级耐火尽端走廊疏散距离？")
#     if st.button("查询") and query:
#         log_step("用户输入", query)
#
#         query = anonymize_query(query)
#         log_step("1. 匿名化完成", query)
#
#         query = preprocess_query(query)
#         log_step("2. 预处理完成", query)
#
#         build_type = infer_building_type(query)
#         log_step("3. 推理建筑类型", build_type)
#
#         query = enhance_query_with_reasoning(query)
#         log_step("4. 思维链增强查询", query)
#
#         current_q = query
#         rewrite_count = 0
#         final_nodes = None
#         final_resp = None
#
#         while rewrite_count <= Config.MAX_REWRITE:
#             log_step(f"第 {rewrite_count + 1} 轮执行", current_q)
#             task = task_router(current_q)
#             log_step("任务路由", task)
#
#             if task == "answer":
#                 resp = llm.complete(f"直接回答：{current_q}")
#                 final_resp = resp.text
#                 break
#
#             resp = query_engine.query(current_q)
#             nodes = resp.source_nodes
#             log_step("检索返回条数", str(len(nodes)))
#             trace_clause_status(current_q, nodes, "刚检索完成（原始召回）")
#
#             sufficient = is_information_sufficient(current_q, nodes)
#             log_step("信息是否充足", str(sufficient))
#             trace_clause_status(current_q, nodes, "信息充足判断前")
#
#             if sufficient:
#                 final_resp = resp.response
#                 final_nodes = nodes
#                 trace_clause_status(current_q, final_nodes, "最终送入模型")  # ✅ 移到正确位置
#                 break
#
#             rewrite_count += 1
#             old_q = current_q
#             current_q = rewrite_query(current_q)
#             log_step("信息不足 → 重写", f"{old_q} → {current_q}")
#             st.info(f"第 {rewrite_count} 次优化查询：{current_q}")
#
#         if final_resp is None:
#             final_resp = "未找到足够相关条文，已按最优结果回答。"
#             if 'resp' in locals():
#                 final_nodes = resp.source_nodes
#
#         log_step("最终回答", final_resp)
#
#         st.subheader("✅ 回答（已按思维链推理）")
#         st.success(final_resp)
#         st.markdown("---")
#         st.subheader("📖 参考依据")
#
#         if final_nodes:
#             for i, n in enumerate(final_nodes, 1):
#                 with st.expander(f"条文 {i} | {n.node.metadata['clause_id']} | 相关度 {n.score:.3f}"):
#                     st.write(f"**标题**：{n.node.metadata['full_title']}")
#                     st.write(f"**来源**：{n.node.metadata.get('source', '未知')}")
#                     tbl = text_to_streamlit_table(n.node.text)
#                     if isinstance(tbl, dict):
#                         st.dataframe(tbl, width='stretch')
#                     else:
#                         st.write(tbl)
#
#
# if __name__ == "__main__":
#     main()

# ========================================极简稳定版：没有langgragh===========================
# ================== 建筑防火规范专家系统 —— 对齐劳动法同款终极版 ==================
import logging
import sys
import os
import json
import re
import shutil
import time
import torch
import chromadb
import streamlit as st
from pathlib import Path
from typing import List, Dict, Literal
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    PromptTemplate, Settings,
    VectorStoreIndex, StorageContext, load_index_from_storage
)
from transformers import BitsAndBytesConfig
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.postprocessor import SentenceTransformerRerank

# ================== Streamlit 页面配置 ==================
st.set_page_config(page_title="建筑防火规范专家", page_icon="🏗️", layout="wide")
st.title("🏗️ 建筑设计防火规范智能查询系统（对齐劳动法同款终极版）")
st.markdown("---")

def disable_streamlit_watcher():
    """禁用Streamlit热重载"""
    def _on_script_changed(_):
        return
    from streamlit import runtime
    runtime.get_instance()._on_script_changed = _on_script_changed

# ================== 【核心：思维链强制提示词】 ==================
QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是专业建筑防火规范专家，只使用中文，禁止出现任何英文、符号。\n"
    "【严格按以下思维链步骤执行，不许跳过】：\n"
    "1. 先判断建筑类型：厂房/仓库/公共建筑/住宅/特殊场所\n"
    "2. 提取关键条件：耐火等级、走道位置（两个安全出口之间/袋形走道尽端）、层数\n"
    "3. 仅使用提供的法规条文，不编造、不扩展、不解释、不推导\n"
    "4. 严格区分厂房、仓库、公共建筑，不混用\n"
    "5. 按原文数值回答，不脑补\n"
    "6. 严禁自行推导、严禁自创计算公式、严禁按比例估算，没有原文定值就直接说明无对应数值，不许编造规则。"
    "7. 只回答袋形走道尽端/安全出口之间的固定疏散距离米数，不扩展疏散宽度、防火分区借用等无关内容。"
    "相关法规条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
response_template = PromptTemplate(QA_TEMPLATE)

# ================== 配置 ==================
class Config:
    EMBED_MODEL_PATH = r"G:\project2026\project3_architect_chat\demo_21\model\EMBED_MODEL"
    LLM_MODEL_PATH = r"G:\project2026\项目1_中医临床智能诊疗系统\model\Qwen1.5-7B-Chat"
    RERANK_MODEL_PATH = r"E:\PythonNotebook\juke ai\work project\l2\day20-RAG+微调实现智能专家系统（部署测试）\day20-RAG+微调实现智能专家系统（部署测试）\demo_20\model\RERANK_MODEL"

    DATA_DIR = r"/data03"
    VECTOR_DB_DIR = "../chroma_db_final"
    PERSIST_DIR = "../storage_final"

    COLLECTION_NAME = "building_code"
    TOP_K = 10
    RERANK_TOP_K = 3
    SIMILARITY_CUTOFF = 0.6
    MIN_RERANK_SCORE = 0.4   # 对齐劳动法：Rerank最低分数阈值
    HYBRID_ALPHA = 0.5       # 对齐劳动法：混合检索权重

# ================== 模型加载 ==================
@st.cache_resource(show_spinner="加载模型中...")
def init_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_PATH, device=device)
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
    reranker = SentenceTransformerRerank(
        model=Config.RERANK_MODEL_PATH,
        top_n=Config.RERANK_TOP_K
    )
    Settings.embed_model = embed_model
    Settings.llm = llm
    return embed_model, llm, reranker

# ================== 文档解析 ==================
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

# ================== 极简预处理（只删废话，保留数字） ==================
def preprocess_query(query: str) -> str:
    stop_words = ["请问", "我想知道", "帮我查", "你好", "谢谢"]
    q = query.strip()
    for w in stop_words:
        q = q.replace(w, "")
    q = re.sub(r'[^\u4e00-\u9fa50-9\s]', '', q)
    return q.strip()

# ================== 极简查询增强 ==================
def enhance_query_with_reasoning(query: str) -> str:
    return f"建筑设计防火规范 {query}".strip()

# ================== 信息充足判断（建筑专属过滤） ==================
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

# ================== 聊天界面初始化（对齐劳动法） ==================
def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("content")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and "reference_nodes" in msg:
                show_reference_details(msg["reference_nodes"])

def show_reference_details(nodes):
    with st.expander("📖 查看参考规范依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] 条文编号：{meta['clause_id']}**")
            st.caption(f"标题：{meta['full_title']} | 相关度：{node.score:.4f}")
            st.info(f"{node.node.text}")

# ================== 主程序（对齐劳动法：多轮会话+混合检索+Rerank阈值） ==================
def main():
    disable_streamlit_watcher()
    embed_model, llm, reranker = init_models()

    if not Path(Config.VECTOR_DB_DIR).exists():
        with st.spinner("构建知识库索引中..."):
            raw = load_and_validate_json_files(Config.DATA_DIR)
            nodes = create_nodes(raw)
    else:
        nodes = None

    index = init_vector_store(nodes)

    # ========== 对齐劳动法：启用 Hybrid 混合检索 ==========
    retriever = index.as_retriever(
        similarity_top_k=Config.TOP_K,
        vector_store_query_mode="hybrid",
        alpha=Config.HYBRID_ALPHA
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=response_template
    )

    # 初始化聊天历史
    init_chat_interface()

    if prompt := st.chat_input("请输入建筑防火规范问题，例如：22层住宅一级耐火尽端走廊疏散距离？"):
        # 存入会话
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("正在检索规范并推理..."):
            start_time = time.time()

            # 预处理
            clean_q = preprocess_query(prompt)
            build_type = infer_building_type(clean_q)
            final_q = enhance_query_with_reasoning(clean_q)

            # 1. 粗召回
            initial_nodes = retriever.retrieve(final_q)
            # 2. 建筑类型强制过滤
            type_filtered_nodes = filter_by_building_type(clean_q, initial_nodes)
            # 3. Rerank精排
            reranked_nodes = reranker.postprocess_nodes(type_filtered_nodes, query_str=final_q)
            # 4. 对齐劳动法：最低分数阈值过滤
            filtered_nodes = [n for n in reranked_nodes if n.score > Config.MIN_RERANK_SCORE]

            if not filtered_nodes:
                resp_text = "⚠️ 未找到匹配的建筑规范条文，请调整问题表述。"
            else:
                response = response_synthesizer.synthesize(final_q, nodes=filtered_nodes)
                resp_text = response.response

        # 回复展示
        with st.chat_message("assistant"):
            st.markdown(resp_text)
            show_reference_details(filtered_nodes[:3])

        # 保存会话历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp_text,
            "reference_nodes": filtered_nodes[:3]
        })

if __name__ == "__main__":
    main()