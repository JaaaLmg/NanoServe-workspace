import os
from dataclasses import dataclass
from transformers import AutoConfig


def validate_positive_int(value, field_name: str) -> int:
    """显式校验"正整数"配置项（Day7 统一口径的唯一入口）。

    - 不使用 assert：`python -O` 会移除断言，校验必须始终生效；
    - 用 type(value) is int 精确匹配类型：bool 是 int 的子类，
      isinstance 会放过 True/False，这里必须显式拒绝；
    - 失败时抛出带字段名和实际值的 ValueError，便于定位配置错误。
    """
    if type(value) is not int:
        raise ValueError(
            f"配置项 {field_name} 必须为正整数，实际为 {value!r}（类型 {type(value).__name__}）"
        )
    if value <= 0:
        raise ValueError(f"配置项 {field_name} 必须为正整数，实际为 {value!r}")
    return value


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Day8：单请求单轮 prefill query 上限（chunked prefill 的计算量控制）。
    # 与 max_num_batched_tokens（全轮预算 B）、kvcache_block_size（物理块粒度）
    # 是三个独立量，互不对齐、互不替代；初始化后不可热修改。
    chunk_size: int = 1024

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # Day7：每轮 token 预算与批序列数上限在配置入口做显式校验。
        # 两者独立生效；允许 B < max_num_seqs（decode 分批的正常配置），
        # 无须要求 B >= max_num_seqs。校验失败直接 ValueError，
        # 不进入任何资源分配（资源账本在 Scheduler 中创建，晚于本校验）。
        validate_positive_int(self.max_num_batched_tokens, "max_num_batched_tokens")
        validate_positive_int(self.max_num_seqs, "max_num_seqs")
        # Day8：chunk_size 复用同一显式校验入口（拒绝 0/负/float/str/bool，
        # python -O 下仍生效）；无须要求 chunk_size <= B 或与块大小对齐。
        validate_positive_int(self.chunk_size, "chunk_size")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
