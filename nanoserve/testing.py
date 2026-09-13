"""CPU 测试桩：fake tokenizer / fake engine / fake factory（Day11–12 §8.1）。

测试与验收脚本共用本模块，不加载模型权重、不初始化 CUDA/NCCL、不访问
HuggingFace 网络。FakeEngine 模拟 Engine 对服务层的全部契约：

- add_request(prompt_token_ids, sampling_params, request_id, deadline) 返回 request ID；
- step() 按预设脚本产生完成记录、无进展或抛出异常；
- pop_completed()/pop_aborted() 返回并清空只读记录（幂等 drain）；
- cancel_request() 记录 signal 调用并立即以中止记录收尾（对应 Day10 的
  "Engine 空闲时安全点立即收尾"语义）；
- exit() 可重复调用并记录次数。

脚本条目（step_script，逐轮消费）：
- None                    → 本轮完成当前全部活动请求（默认行为，验证批量完成）
- ("pending",)            → 本轮有工作量但未完成：返回 ([], 1)，请求保持活动
                            （模拟正常 decode 轮，worker 继续驱动）
- ("noop",)               → 本轮无进展：返回 ([], 0) 且请求保持活动
                            （触发 worker 的 Day10 无进展错误语义）
- Exception 实例          → 本轮抛出异常
- [(request_id, tokens)]  → 本轮只完成指定请求
"""

import threading
import time
from time import perf_counter

from nanovllm.engine.completed_request import AbortedRequest, CompletedRequest
try:
    from nanovllm.engine.completed_request import TokenEvent
except ImportError:  # bottom layer may land the DTO after this test stub
    from nanoserve.service import TokenEvent

from nanoserve.app import EngineBundle


class FakeTokenizer:
    """确定性 tokenizer 桩：字符 → ord 映射，记录 chat template 调用参数。"""

    def __init__(self, *, template_error: Exception | None = None,
                 template_style: str = "list", token_text: dict[int, str] | None = None):
        """template_style 模拟不同 transformers 版本的模板返回类型：
        - "list": list[int]（tokenize=True 的经典返回）
        - "dict": 含 input_ids 的 BatchEncoding 形态（transformers 5.x）
        - "str":  只能返回模板字符串（需服务层用同一 tokenizer 再编码）
        """
        self.calls: list[dict] = []
        self.encode_calls: list[str] = []
        self.decode_calls: list[list[int]] = []
        self.template_error = template_error
        self.template_style = template_style
        self.token_text = dict(token_text or {})

    def encode(self, text: str) -> list[int]:
        self.encode_calls.append(text)
        return [ord(ch) % 999 + 1 for ch in text]

    def decode(self, token_ids) -> str:
        ids = list(token_ids)
        self.decode_calls.append(ids)
        if self.token_text:
            return "".join(self.token_text.get(token, f"<{token}>") for token in ids)
        return f"text<{len(ids)}>"

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=False, **kwargs):
        # 记录原始调用参数，供测试断言"收到原始有序 messages /
        # tokenize=True / add_generation_prompt=True"
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        })
        if self.template_error is not None:
            raise self.template_error
        # 确定性 token 序列：模板头 + 每条消息一个标记 + generation prompt
        tokens = [11, 12]
        tokens.extend(20 + i for i in range(len(messages)))
        if add_generation_prompt:
            tokens.append(99)
        if not tokenize:
            return "<template-string>"
        if self.template_style == "str":
            # 兼容路径：某些 tokenizer 只能返回模板字符串，
            # 服务层必须用同一 tokenizer 再编码恰好一次
            return "<template-string>"
        if self.template_style == "dict":
            # transformers 5.x 形态：tokenize=True 返回 BatchEncoding/dict
            return {"input_ids": [tokens], "attention_mask": [1] * len(tokens)}
        return tokens


class FakeEngine:
    """Engine 契约桩：线程安全级别与真实 Engine 相同（控制面由 worker 单线程串行）。"""

    def __init__(self, *, tokenizer: FakeTokenizer | None = None,
                 completion_tokens: tuple[int, ...] = (21, 22, 23),
                 step_script: list | None = None,
                 token_event_script: list | None = None,
                 token_events_script: list | None = None,
                 fail_add_request: bool = False,
                 step_delay: float = 0.0):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.completion_tokens = completion_tokens
        self.step_script = list(step_script or [])
        # 每个 step 取一个脚本轮次；轮次可为 {request_id: token_ids}，或
        # [(request_id, token_ids)]，用于模拟真实 Engine 的增量事件通道。
        self.token_event_script = list(
            token_event_script if token_event_script is not None
            else (token_events_script or []))
        self.fail_add_request = fail_add_request
        # 每轮 step 前的固定延迟：模拟真实 forward 耗时，让"运行中"窗口可观测
        self.step_delay = step_delay
        # 状态
        self.active: dict[str, dict] = {}
        self.completed_records: list[CompletedRequest] = []
        self.aborted_records: list[AbortedRequest] = []
        self.token_events: list[TokenEvent] = []
        self._round_id = 0
        self.resource = {"running": 0, "waiting": 0, "paused": 0,
                         "active": 0, "used_blocks": 0, "total_blocks": 0}
        # 观测记录
        self.step_calls = 0
        self.step_thread_ids: set[int] = set()
        self.added_requests: list[tuple] = []
        self.cancel_calls: list[tuple[str, str]] = []
        self.exit_calls = 0

    # ---------- Engine 契约 ----------

    def add_request(self, prompt_token_ids, sampling_params,
                    request_id=None, deadline=None) -> str:
        if self.fail_add_request:
            raise ValueError("stub add_request failure")
        rid = request_id or f"req-{len(self.added_requests)}"
        if rid in self.active:
            raise ValueError(f"duplicate active request id {rid!r}")
        self.active[rid] = {
            "seq_id": len(self.added_requests) + 1,
            "prompt_tokens": len(prompt_token_ids),
            "max_tokens": sampling_params.max_tokens,
            "deadline": deadline,
        }
        self.added_requests.append(
            (tuple(prompt_token_ids), sampling_params, rid, deadline))
        self.resource["active"] = len(self.active)
        self.resource["waiting"] = len(self.active)
        return rid

    def has_active_requests(self) -> bool:
        return bool(self.active)

    def get_request(self, request_id):
        info = self.active.get(request_id)
        if info is None:
            return None
        return type("FakeSequence", (), {"seq_id": info["seq_id"]})()

    def step(self):
        self.step_calls += 1
        self.step_thread_ids.add(threading.get_ident())
        if self.step_delay:
            time.sleep(self.step_delay)
        self._round_id += 1
        if self.token_event_script:
            scripted = self.token_event_script.pop(0)
            if isinstance(scripted, BaseException):
                raise scripted
            if isinstance(scripted, dict):
                scripted = list(scripted.items())
            for rid, tokens in scripted or []:
                info = self.active.get(rid)
                if info is None:
                    continue
                if isinstance(tokens, int):
                    tokens = (tokens,)
                token_ids = tuple(tokens)
                index = info.setdefault("completion_index", 0)
                self.token_events.append(TokenEvent(
                    seq_id=info["seq_id"], request_id=rid,
                    round_id=self._round_id, token_ids=token_ids,
                    completion_index=index, emitted_at=perf_counter(),
                        is_first_token=(index == 0), phase="decode"))

                info["completion_index"] = index + len(token_ids)
        action = self.step_script.pop(0) if self.step_script else None
        if isinstance(action, BaseException):
            raise action
        if action == ("noop",):
            # 无进展轮：保留全部活动请求（§8.4 无进展错误语义的触发入口）
            return [], 0
        if action == ("pending",):
            # 正常 decode 轮：有工作量（每活动请求 1 个 query token）但未完成
            return [], len(self.active)
        targets = action if isinstance(action, list) \
            else [(rid, None) for rid in list(self.active)]
        outputs = []
        for rid, override_tokens in targets:
            info = self.active.pop(rid, None)
            if info is None:
                continue
            self.resource["active"] = len(self.active)
            self.resource["waiting"] = len(self.active)
            tokens = tuple(override_tokens
                           if override_tokens is not None
                           else self.completion_tokens[:info["max_tokens"]])
            # 默认完成脚本也模拟真实 Engine 的逐 token 事件；按请求记录
            # 是否已有显式事件，不能用全局队列是否为空判断（多请求合批时，
            # 前一个请求的事件会让后一个请求错误地跳过增量输出）。
            scripted_ids = set()
            for event in self.token_events:
                if event.request_id == rid and event.round_id == self._round_id:
                    scripted_ids.add(event.request_id)
            if rid not in scripted_ids:
                index = 0
                for token_id in tokens:
                    self.token_events.append(TokenEvent(
                        seq_id=info["seq_id"], request_id=rid,
                        round_id=self._round_id, token_ids=(token_id,),
                        completion_index=index, emitted_at=perf_counter(),
                        is_first_token=index == 0, phase="decode"))
                    index += 1
            finished_at = perf_counter()
            self.completed_records.append(CompletedRequest(
                seq_id=info["seq_id"], request_id=rid,
                completion_token_ids=tokens,
                prompt_tokens=info["prompt_tokens"],
                completion_tokens=len(tokens),
                # 达到 max_tokens 记 length，其余记 stop（与底层公开语义一致）
                finish_reason="length"
                if len(tokens) >= info["max_tokens"] else "stop",
                finished_at=finished_at))
            outputs.append((info["seq_id"], list(tokens)))
        return outputs, len(outputs)

    def pop_token_events(self) -> list[TokenEvent]:
        records = list(self.token_events)
        self.token_events.clear()
        return records

    def pop_completed(self) -> list[CompletedRequest]:
        records = list(self.completed_records)
        self.completed_records.clear()
        return records

    def pop_aborted(self) -> list[AbortedRequest]:
        records = list(self.aborted_records)
        self.aborted_records.clear()
        return records

    def cancel_request(self, request_id, reason="client_cancelled") -> bool:
        self.cancel_calls.append((request_id, reason))
        info = self.active.pop(request_id, None)
        if info is None:
            return False
        self.resource["active"] = len(self.active)
        self.resource["waiting"] = len(self.active)
        self.aborted_records.append(AbortedRequest(
            seq_id=info["seq_id"], request_id=request_id,
            # 保留 Day10 控制面的首次取消原因，服务层据此映射错误。
            finish_reason=reason, finished_at=perf_counter()))
        return True

    def resource_snapshot(self) -> dict:
        """返回独立快照，测试可验证服务层不会持有 Engine 可写对象。"""
        snapshot = dict(self.resource)
        snapshot["active"] = len(self.active)
        snapshot["running"] = len(self.active)
        snapshot["waiting"] = 0
        return snapshot

    snapshot = resource_snapshot

    def exit(self) -> None:
        # 幂等：与真实 LLMEngine.exit 一致——退出后的重复调用是安全空操作
        if getattr(self, "_exited", False):
            return
        self._exited = True
        self.exit_calls += 1


def make_fake_factory(engine: FakeEngine | None = None,
                      tokenizer: FakeTokenizer | None = None,
                      max_model_len: int = 4096):
    """构造可注入 create_app 的 fake factory（返回闭包便于测试持有桩引用）。"""
    engine = engine or FakeEngine(tokenizer=tokenizer)
    tokenizer = engine.tokenizer

    def factory(server_config) -> EngineBundle:
        return EngineBundle(engine=engine, tokenizer=tokenizer,
                            max_model_len=max_model_len)

    factory.engine = engine
    factory.tokenizer = tokenizer
    return factory
