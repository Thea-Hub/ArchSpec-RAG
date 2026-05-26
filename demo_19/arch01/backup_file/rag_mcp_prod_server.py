import uvicorn
import os
import contextlib
import logging
from collections.abc import AsyncIterator
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

# 导入你自己的RAG类
from core import BuildingCodeRAG

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# 1. 初始化FastMCP实例
mcp = FastMCP("building-code-rag-server")

# 2. 全局只初始化一次RAG（关键：只加载一次模型/向量库）
rag_agent = BuildingCodeRAG()

# 3. 注册MCP工具
@mcp.tool()
def query_building_code(query: str) -> str:
    """查询建筑防火规范条文"""
    return rag_agent.query(query)

# 4. 创建无状态流式HTTP会话管理器
session_manager = StreamableHTTPSessionManager(
    app=mcp,
    event_store=None,
    json_response=None,
    stateless=True,   # 生产级无状态，支持多客户端、多进程
)

# 5. 请求处理入口
async def handle_streamable_http(
    scope: Scope,
    receive: Receive,
    send: Send
) -> None:
    await session_manager.handle_request(scope, receive, send)

# 6. 生命周期：启动预热、关闭收尾
@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        logger.info("✅ MCP StreamableHTTP 服务已启动")
        logger.info(f"✅ 局域网访问地址示例：http://你的本机IP:{PORT}/mcp")
        try:
            yield
        finally:
            logger.info("🛑 MCP 服务正在关闭...")

# 7. 搭建Starlette ASGI应用
starlette_app = Starlette(
    debug=False,   # 生产环境关掉debug
    routes=[
        Mount("/mcp", app=handle_streamable_http),
    ],
    lifespan=lifespan,
)

# 8. 运行服务
def run_server():
    uvicorn.run(
        starlette_app,
        host=HOST,
        port=PORT,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()