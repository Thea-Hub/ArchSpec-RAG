# ======================
# 🔥 终极环境修复（必须放第一行）
# ======================
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OPENTELEMETRY_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ======================
# 🔥 真正服务器级全局每日限流（20次/天，无法绕过）
# ======================
import time
import json
from datetime import datetime
import streamlit as st

# ---------- 配置 ----------
DAILY_GLOBAL_LIMIT = 20  # 全局每日20次
USER_SESSION_LIMIT = 3    # 单会话3次
LIMIT_FILE = "rate_limit.json"  # 服务器本地文件存计数

# ---------- 全局文件计数（服务器级） ----------
def load_global_limit():
    if not os.path.exists(LIMIT_FILE):
        return {"date": str(datetime.now().date()), "count": 0}
    try:
        with open(LIMIT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"date": str(datetime.now().date()), "count": 0}

def save_global_limit(data):
    with open(LIMIT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_global_daily_limit():
    """检查全局每日20次，跨天自动重置"""
    data = load_global_limit()
    today = str(datetime.now().date())

    # 跨天重置
    if data["date"] != today:
        data = {"date": today, "count": 0}
        save_global_limit(data)

    # 超过全局上限
    if data["count"] >= DAILY_GLOBAL_LIMIT:
        st.error(f"❌ 今日全局额度已用完（每日{DAILY_GLOBAL_LIMIT}次），请明天再来！")
        return False

    # 计数+1并保存
    data["count"] += 1
    save_global_limit(data)
    return True

# ---------- 单会话限流（浏览器级） ----------
if "user_requests" not in st.session_state:
    st.session_state["user_requests"] = 0

def check_user_session_limit():
    if st.session_state["user_requests"] >= USER_SESSION_LIMIT:
        st.warning(f"⚠️ 你已达到本次会话上限（{USER_SESSION_LIMIT}次），请刷新页面重新体验。")
        return False
    st.session_state["user_requests"] += 1
    return True

# ======================
# 正常导入
# ======================
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

# 缓存 RAG 引擎
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
    retriever, reranker, response_synthesizer = load_rag_engine()
    init_chat_interface()

    # 显示全局剩余次数
    global_data = load_global_limit()
    remaining_global = max(0, DAILY_GLOBAL_LIMIT - global_data["count"])
    st.caption(f"📊 今日全局剩余次数：{remaining_global} / {DAILY_GLOBAL_LIMIT}")

    if prompt := st.chat_input("请输入建筑防火规范问题..."):
        # 1. 先检查全局每日20次（服务器级，不可绕过）
        if not check_global_daily_limit():
            st.stop()
        # 2. 再检查单会话3次（浏览器级）
        if not check_user_session_limit():
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("正在检索规范并推理..."):
            clean_q = preprocess_query(prompt)
            final_q = enhance_query_with_reasoning(clean_q)
            initial_nodes = retriever.retrieve(final_q)
            filtered_nodes = filter_by_building_type(clean_q, initial_nodes)
            
            # 恢复 rerank 精排（效果回归）
            reranked_nodes = reranker.postprocess_nodes(filtered_nodes, query_str=final_q)

            # 分数过滤
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
