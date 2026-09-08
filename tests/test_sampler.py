import torch
import pytest
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def test_greedy_and_seeded_sampling():
    logits = torch.tensor([[0.1, 3.0, 1.0], [2.0, 1.0, 0.0]])
    assert Sampler.sample(logits, torch.zeros(2)).tolist() == [1, 0]
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = Sampler.sample(logits, torch.ones(2), generators=[g1, g1])
    b = Sampler.sample(logits, torch.ones(2), generators=[g2, g2])
    assert torch.equal(a, b)


def test_top_p_keeps_only_nucleus():
    logits = torch.tensor([[4.0, 3.0, -10.0]])
    out = Sampler.sample(logits, torch.ones(1), torch.tensor([0.5]), [torch.Generator().manual_seed(1)])
    assert int(out.item()) in (0, 1)


def test_sampling_params_validation():
    assert SamplingParams(temperature=0).is_greedy
    with pytest.raises(ValueError):
        SamplingParams(top_p=0)
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)
