# ========================fastapi 修复版（最终可用）=================================
import logging
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from core import BuildingCodeRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_mcp_server")

mcp = FastMCP("rag-agent-server")
rag_agent = BuildingCodeRAG()  # 全局只加载一次，正确！

@mcp.tool()
async def query_building_code(query: str) -> list[TextContent]:
    """查询建筑防火规范条文"""
    # 修复：你的类方法是 query，不是 invoke！
    result = rag_agent.query(query)
    return [TextContent(text=str(result))]

if __name__ == "__main__":
    # 推荐：streamable-http 兼容性最好
    mcp.run(transport='streamable-http')