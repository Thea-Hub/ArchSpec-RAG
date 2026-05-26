import json
import os
import torch
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ===================== 【关键】导入你的 RAG 系统 =====================
from core import (
    init_models,
    get_rag_engine,
    preprocess_query,
    enhance_query_with_reasoning,
    filter_by_building_type,
    convert_table_to_text,  # 你的表格解析（完全对齐）
)

# ===================== RAGAS 评估配置（裁判模型） =====================
api_key = os.getenv("DASHSCOPE_API_KEY")
eval_llm = ChatOpenAI(
    model="qwen-turbo",
    temperature=0.1,
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 嵌入模型（可与入库不同，不影响）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# ===================== 加载你的 RAG 引擎（只加载一次！） =====================
print("🔄 正在加载你的 RAG 系统...")
retriever, reranker, response_synthesizer = get_rag_engine()
print("✅ RAG 加载完成！")

# ===================== 读取 test_questions.json =====================
with open("test_questions.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# ===================== 🔥 全自动：真实检索 + 真实生成 =====================
questions = []
references = []
contexts = []
answers = []

for idx, item in enumerate(test_data):
    query = item["question"]
    ref = item["ground_truth"]

    print(f"\n📝 正在评测第 {idx + 1} 题：{query}")

    # --------------------------
    # 1. 你的 RAG 真实检索
    # --------------------------
    processed_q = preprocess_query(query)
    enhanced_q = enhance_query_with_reasoning(processed_q)
    nodes = retriever.retrieve(enhanced_q)
    filtered_nodes = filter_by_building_type(query, nodes)
    reranked_nodes = reranker.postprocess_nodes(filtered_nodes, query_str=query)

    # 拼接上下文（完全和你RAG入库格式一致）
    context_texts = [node.text for node in reranked_nodes]
    rag_context = [text.strip() for text in context_texts if text.strip()]

    # --------------------------
    # 2. 你的 RAG 真实 7B 生成回答
    # --------------------------
    response = response_synthesizer.get_response(query, rag_context)

    # 存入数据集
    questions.append(query)
    references.append(ref)
    contexts.append(rag_context)
    answers.append(response.strip())

    print(f"✅ 生成回答：{response[:50]}...")

# ===================== RAGAS 官方格式 =====================
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "reference": references,
})

# ===================== 执行评估 =====================
print("\n🚀 开始 RAGAS 评估...")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=eval_llm,
    embeddings=embeddings,
)

# ===================== 保存结果 =====================
df = result.to_pandas()
df.to_excel("RAG真实评测结果_local01.xlsx", index=False)

print("\n🎉 全部完成！")
print(result)
print("\n✅ 结果已保存到：RAG真实评测结果_local01.xlsx")