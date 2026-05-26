# main_framework.py
from business1_client import business1_graph
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    query: str
    business_type: str
    response: str

# 节点：判断业务类型
def route_business(state: State):
    if "建筑" in state["query"] or "规范" in state["query"]:
        return {"business_type": "business1"}
    return {"business_type": "unknown"}

workflow = StateGraph(State)
workflow.add_node("route_business", route_business)
workflow.add_node("business1", business1_graph) # 直接导入子图

workflow.set_entry_point("route_business")
workflow.add_conditional_edges(
    "route_business",
    lambda x: "business1" if x["business_type"] == "business1" else END,
    {"business1": "business1", END: END}
)
workflow.add_edge("business1", END)

main_graph = workflow.compile()