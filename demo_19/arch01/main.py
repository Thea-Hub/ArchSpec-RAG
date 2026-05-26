import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OPENTELEMETRY_ENABLED"] = "False"
import streamlit as st
print("st.secrets keys:", list(st.secrets.keys()))
from core import (
    Config,
    get_rag_engine,
    preprocess_query,
    enhance_query_with_reasoning,
    filter_by_building_type,
    text_to_streamlit_table
)

st.set_page_config(page_title="建筑防火规范专家", page_icon="🏗️", layout="wide")
st.title("🏗️ 建筑设计防火规范智能查询系统")
st.markdown("---")


# 缓存 RAG 引擎，避免重复加载（最重要修复）
@st.cache_resource
def load_rag_engine():
    return get_rag_engine()


def disable_streamlit_watcher():
    try:
        def _on_script_changed(_):
            return

        from streamlit import runtime
        if runtime.exists():
            runtime.get_instance()._on_script_changed = _on_script_changed
    except:
        pass


def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "reference_nodes" in msg:
                show_ref(msg["reference_nodes"])


def show_ref(nodes):
    with st.expander("📖 查看参考规范依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] 条文编号：{meta['clause_id']}**")
            st.caption(f"标题：{meta['full_title']} | 相关度：{node.score:.4f}")
            table_data = text_to_streamlit_table(node.node.text)
            if isinstance(table_data, dict):
                st.dataframe(table_data, use_container_width=True)
            else:
                st.info(node.node.text)


def main():
    disable_streamlit_watcher()

    # 从缓存加载引擎（修复）
    retriever, reranker, response_synthesizer = load_rag_engine()

    init_chat_interface()

    if prompt := st.chat_input("请输入建筑防火规范问题，例如：22层住宅一级耐火尽端走廊疏散距离？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("正在检索规范并推理..."):
            clean_q = preprocess_query(prompt)
            final_q = enhance_query_with_reasoning(clean_q)
            initial_nodes = retriever.retrieve(final_q)
            filtered_nodes = filter_by_building_type(clean_q, initial_nodes)
            reranked_nodes = reranker.postprocess_nodes(filtered_nodes, query_str=final_q)
            reranked_nodes = [n for n in reranked_nodes if n.score > Config.MIN_RERANK_SCORE]

            if not reranked_nodes:
                resp_text = "⚠️ 未找到匹配的建筑规范条文，请调整问题表述。"
            else:
                response = response_synthesizer.synthesize(final_q, nodes=reranked_nodes)
                resp_text = response.response

        with st.chat_message("assistant"):
            st.markdown(resp_text)
            show_ref(reranked_nodes[:3])

        st.session_state.messages.append({
            "role": "assistant",
            "content": resp_text,
            "reference_nodes": reranked_nodes[:3]
        })


if __name__ == "__main__":
    main()
