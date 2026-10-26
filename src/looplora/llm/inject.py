"""Pinned offline model loading and scoped adapter contexts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Iterator

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from looplora.infra.seeds import derive_seed

if TYPE_CHECKING:
    from looplora.core.config import AdapterConfig
    from looplora.core.wrapper import AdaptedLinear

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
CACHE_DIR = "/work/xg24i002/x10041/hf_home/hub"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE_DIR, local_files_only=True,
    )


def load_model(*, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE_DIR, local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.requires_grad_(False)
    return model.eval()


class AdapterBank:
    """Contexts belong to one forward call; snapshots belong to the caller.

    The bank is deliberately not an nn.Module: the wrapped projections already
    register all adapter parameters exactly once in the model's module tree.
    """

    def __init__(self, modules: dict[int, AdaptedLinear]):
        self.modules = modules
        self._last_states: dict[int, Tensor] = {}

    def zero_states(self, batch: int) -> dict[int, Tensor]:
        return {
            layer: torch.zeros(
                batch, module.adapter.cfg.r_mem, module.adapter.cfg.r_mem,
                dtype=torch.float32, device=next(module.adapter.parameters()).device,
            )
            for layer, module in self.modules.items()
        }

    def states(self) -> dict[int, Tensor]:
        return {
            layer: module.context.state
            for layer, module in self.modules.items()
        } if all(m.context is not None for m in self.modules.values()) else dict(self._last_states)

    @contextmanager
    def use(
        self, states: dict[int, Tensor] | None = None, *, loop_steps: int = 1,
        valid: Tensor | None = None, write: Tensor | None = None,
        reset: Tensor | None = None, mode: str = "read_only",
    ) -> Iterator[AdapterBank]:
        from looplora.core.wrapper import MemoryContext

        if states is None:
            if valid is None:
                raise ValueError("valid mask is required when allocating initial states")
            states = self.zero_states(valid.shape[0])
        if set(states) != set(self.modules):
            raise ValueError("snapshot layers do not match installed adapters")
        previous = {layer: module.context for layer, module in self.modules.items()}
        try:
            for layer, module in self.modules.items():
                state = states[layer]
                if state.dtype != torch.float32:
                    raise ValueError("LLM adapter memory must be fp32")
                module.context = MemoryContext(
                    state=state.clone(), loop_steps=loop_steps, valid=valid,
                    write=write, reset=reset, mode=mode,
                )
            yield self
            self._last_states = self.states()
        finally:
            for layer, module in self.modules.items():
                module.context = previous[layer]


def attach_adapters(
    model: nn.Module, config: AdapterConfig, *, layers: tuple[int, ...] = (14,), seed: int = 0,
) -> AdapterBank:
    from looplora.core.adapters import make_adapter
    from looplora.core.wrapper import AdaptedLinear

    if hasattr(model, "looplora_adapters"):
        raise ValueError("adapters are already installed")
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be nonempty and unique")
    projections = {}
    for layer in layers:
        if not 0 <= layer < len(model.model.layers):
            raise ValueError(f"invalid adapter layer {layer}")
        projection = model.model.layers[layer].self_attn.o_proj
        if not isinstance(projection, nn.Linear):
            raise TypeError("adapter insertion requires an nn.Linear o_proj")
        projections[layer] = projection
    model.requires_grad_(False)
    modules = {}
    for layer, projection in projections.items():
        cfg = replace(config, d_in=projection.in_features, d_out=projection.out_features)
        # Initialize on CPU with a local seed, preserving the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(seed, "llm-adapter", layer))
            adapter = make_adapter(cfg).to(device=projection.weight.device, dtype=torch.float32)
        wrapper = AdaptedLinear(projection, adapter)
        model.model.layers[layer].self_attn.o_proj = wrapper
        modules[layer] = wrapper
    bank = AdapterBank(modules)
    model.looplora_adapters = bank
    return bank


def detach_adapters(model: nn.Module) -> AdapterBank:
    bank = model.looplora_adapters
    for layer, module in bank.modules.items():
        model.model.layers[layer].self_attn.o_proj = module.base
    del model.looplora_adapters
    return bank
