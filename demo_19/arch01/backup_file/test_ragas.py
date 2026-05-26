import json
import pandas as pd
from tqdm import tqdm
from core import (
    get_rag_engine,
    preprocess_query,
    enhance_query_with_reasoning,
    filter_by_building_type,
    Config
)
from llama_index.core import Settings

# ==================== 自己实现 RAG 四个评分 ====================
def calculate_scores(question, answer, contexts, ground_truth):
    try:
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._context_precision import ContextPrecision

        llm = Settings.llm
        embed = Settings.embed_model

        f = Faithfulness(llm=llm).compute(question, answer, contexts)
        a = AnswerRelevancy(llm=llm, embedding=embed).compute(question, answer)
        cr = ContextRecall(llm=llm).compute(question, contexts, ground_truth)
        cp = ContextPrecision(llm=llm).compute(question, contexts, ground_truth)

        return {
            "faithfulness": f,
            "answer_relevancy": a,
            "context_recall": cr,
            "context_precision": cp
        }
    except Exception as e:
        print("评分报错:", e)
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0
        }

def format_table_to_markdown(table_data: dict) -> str:
    if not table_data:
        return ""

    markdown_lines = []
    title = table_data.get("标题", "")
    content = table_data.get("内容", [])

    if title:
        markdown_lines.append(f"### {title}")

    for row in content:
        for class_name, details in row.items():
            markdown_lines.append(f"\n**{class_name}**")
            markdown_lines.append("| 项目 | 内容 |")
            markdown_lines.append("|------|------|")

            for key, value in details.items():
                value = value.replace(";", "；").strip()
                markdown_lines.append(f"| {key} | {value} |")

    return "\n".join(markdown_lines)

def build_context(chunk_data):
    context_parts = []

    if "正文" in chunk_data:
        context_parts.append(chunk_data["正文"].strip())

    if "表格" in chunk_data:
        md_table = format_table_to_markdown(chunk_data["表格"])
        context_parts.append(md_table)

    return "\n\n".join(context_parts)

# ====================== 主流程 ======================
def main():
    retriever, reranker, response_synthesizer = get_rag_engine()

    with open("../test_questions.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    rows = []
    print("开始批量测试 + 本地评分...")

    for item in tqdm(test_set):
        q = item["question"]
        gt = item["ground_truth"]

        clean_q = preprocess_query(q)
        final_q = enhance_query_with_reasoning(clean_q)
        nodes = retriever.retrieve(final_q)
        nodes = filter_by_building_type(clean_q, nodes)
        nodes = reranker.postprocess_nodes(nodes, query_str=final_q)
        nodes = [n for n in nodes if n.score > Config.MIN_RERANK_SCORE]

        if nodes:
            resp = response_synthesizer.synthesize(final_q, nodes=nodes)
            ans = resp.response

            # ====================== ✅ 【修复点】格式化上下文 ======================
            ctx = []
            for n in nodes:
                try:
                    # 解析 node 里的 JSON 结构
                    chunk_json = json.loads(n.node.text)
                    # 调用 build_context 转干净表格！
                    clean_context = build_context(chunk_json)
                    ctx.append(clean_context)
                except:
                    ctx.append(n.node.text)
            # ======================================================================

        else:
            ans = "未找到相关规范条文"
            ctx = []

        scores = calculate_scores(q, ans, ctx, gt)

        row = {
            "question": q,
            "answer": ans,
            "contexts": ctx,
            "ground_truth": gt,
            **scores
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel("RAG本地评分结果_最终版.xlsx", index=False)

    print("\n===== 最终平均分 =====")
    print(df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean())
    print("\n✅ 已保存：RAG本地评分结果_最终版.xlsx")

if __name__ == "__main__":
    main()