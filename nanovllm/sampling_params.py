from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 1e-10
