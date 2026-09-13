"""NanoServe：基于 nano-vLLM 的 OpenAI 兼容 HTTP 服务包（Day11–12）。

包结构：
- config.py   ServerConfig（env/CLI 解析，不替代 engine.Config）
- schemas.py  Pydantic 请求/响应与错误模型
- service.py  InternalRequest、RequestManager、错误类型与结果 DTO
- worker.py   EngineWorker（唯一直接调用 engine.step() 的线程）
- api.py      /health、/v1/models 与两个非流式 POST 路由
- app.py      FastAPI app factory、lifespan 与统一异常处理
- server.py   python -m nanoserve.server 启动入口
- testing.py  CPU 测试桩（fake tokenizer/engine factory），不加载 GPU/权重

Day11–12 边界：接受并校验 stream 字段，但只交付 stream=false；
stream=true 返回 501，SSE 与断连检测属于 Day13。
"""

__version__ = "0.1.0"
