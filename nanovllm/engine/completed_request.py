"""完成记录通道（Day11–12 §4.4）：Engine 面向服务层的只读终态记录。

背景：LLMEngine.step() 的公开返回值是 ``[(seq_id, completion_token_ids), ...]``，
而 postprocess 完成请求后会立刻把 Sequence 从 Scheduler 活动索引清理。服务层
（Day11+ 的 HTTP worker）无法再凭 request_id 反查完成原因、prompt token 数或
请求身份。本模块定义两类不可变记录，由 Scheduler 在终态收尾（_finalize）时
捕获，LLMEngine.pop_completed()/pop_aborted() 排出给服务 worker 消费：

- CompletedRequest：正常完成（EOS/达到 max_tokens）。记录 completion token IDs
  （字段语义全项目统一：只存 completion，不含 prompt；服务层用同一 tokenizer
  解码后作为响应文本），usage 与 finish_reason 与终态迁移同源。
- AbortedRequest：取消/超时/异常收尾。不携带任何 token 明文，只用于把等待中
  的服务层 Future 收口为明确失败，绝不伪装成正常 completion。

记录只属于 rank 0 控制面，不进入 TP 的模型 payload；pop 返回后从队列移除，
重复调用不会重复返回（幂等 drain）。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    """一次正常完成的只读记录（FINISHED 迁移与 _finalize 清理之间捕获）。"""

    seq_id: int
    request_id: str
    # 明确只存 completion token IDs（不含 prompt），避免服务层误解字段语义；
    # prompt token 数单独记录在 prompt_tokens，usage 可完整还原
    completion_token_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]
    finished_at: float


@dataclass(frozen=True, slots=True)
class AbortedRequest:
    """一次非正常终止的只读记录（CANCELLED/TIMEOUT 终态迁移后捕获）。

    finish_reason 保留底层原值（cancelled/timeout/engine_error/engine_exit 等），
    服务层据此把等待中的请求映射为对应的错误响应，而不是成功 choice。
    """

    seq_id: int
    request_id: str
    finish_reason: str
    finished_at: float
