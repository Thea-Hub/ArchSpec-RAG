# business1_client.py
from langgraph.graph import StateGraph, END
from mcp.client import Client
from typing import TypedDict

# 连接到你的 MCP Server
mcp_client = Client("stdio://rag_mcp_server.py")

class State(TypedDict):
    query: str
    in_scope: bool
    response: str

# 节点：判断是否在业务范围
def check_scope(state: State):
    # 你的业务判断逻辑
    return {"in_scope": True}

# 节点：调用 MCP 工具
def call_rag_tool(state: State):
    result = mcp_client.call("query_building_code", query=state["query"])
    return {"response": result}

workflow = StateGraph(State)
workflow.add_node("check_scope", check_scope)
workflow.add_node("call_tool", call_rag_tool)
workflow.set_entry_point("check_scope")
workflow.add_conditional_edges(
    "check_scope",
    lambda x: "call_tool" if x["in_scope"] else END,
    {"call_tool": "call_tool", END: END}
)
workflow.add_edge("call_tool", END)

# 导出编译好的图，供主框架调用
business1_graph = workflow.compile()