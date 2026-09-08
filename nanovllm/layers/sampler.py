import torch
from torch import nn


class Sampler(nn.Module):
    """Sample one token per row with greedy, temperature and nucleus modes."""

    @staticmethod
    def sample(logits: torch.Tensor, temperatures: torch.Tensor,
               top_ps: torch.Tensor | None = None,
               generators: list[torch.Generator | None] | None = None) -> torch.Tensor:
        logits = logits.float()
        if top_ps is None:
            top_ps = torch.ones(logits.size(0), device=logits.device)
        result = []
        for row, temperature, top_p in zip(logits, temperatures, top_ps):
            if float(temperature) <= 1e-10:
                result.append(row.argmax())
                continue
            scaled = row / temperature
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative - probs >= float(top_p)
            remove[0] = False
            probs = probs.masked_fill(remove, 0)
            probs = probs / probs.sum()
            generator = generators[len(result)] if generators else None
            choice = torch.multinomial(probs, 1, generator=generator)
            result.append(sorted_indices[choice].squeeze(0))
        return torch.stack(result)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_ps: torch.Tensor | None = None,
                generators: list[torch.Generator | None] | None = None):
        return self.sample(logits, temperatures, top_ps, generators)
