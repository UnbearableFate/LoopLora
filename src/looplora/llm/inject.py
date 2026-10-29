"""Pinned offline model loading and scoped adapter contexts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from typing import TYPE_CHECKING, Iterator, Mapping

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from looplora.core.memory_backend import MemoryRead
from looplora.infra.seeds import derive_seed

if TYPE_CHECKING:
    from looplora.core.config import AdapterConfig
    from looplora.core.wrapper import AdaptedLinear
    from .residual import ResidualInputLinear

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
CACHE_DIR = "/work/xg24i002/x10041/hf_home/hub"


def configure_numerics() -> None:
    """Make exact Gate 0 comparisons reproducible across worker processes.

    Configure before the first model operation: default GPU kernel selection
    produced process-dependent BF16 support states in the full smoke suite.
    Deterministic algorithms fail explicitly if an unsupported operation is used.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def backbone_pin(backbone: Mapping[str, str] | None = None) -> tuple[str, str]:
    """(model id, revision) of the backbone. ``None`` is the pinned 0.6B default; a configuration may name another
    cached backbone of the same family with ``{"id": ..., "revision": ...}`` (`config.backbone`)."""
    if backbone is None:
        return MODEL_ID, MODEL_REVISION
    return backbone["id"], backbone["revision"]


def load_tokenizer(backbone: Mapping[str, str] | None = None):
    model_id, revision = backbone_pin(backbone)
    return AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=CACHE_DIR, local_files_only=True,
    )


def load_model(*, dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
               backbone: Mapping[str, str] | None = None):
    configure_numerics()
    model_id, revision = backbone_pin(backbone)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, cache_dir=CACHE_DIR, local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.requires_grad_(False)
    return model.eval()


class AdapterBank:
    """Contexts belong to one forward call; snapshots belong to the caller.

    The bank is deliberately not an nn.Module: the wrapped projections already
    register all adapter parameters exactly once in the model's module tree.
    """

    def __init__(self, modules: dict[int, AdaptedLinear | ResidualInputLinear]):
        self.modules = modules
        self._last_states: dict[int, Tensor] = {}
        self.input_hooks = []

    def zero_states(self, batch: int) -> dict[int, Tensor]:
        return {
            layer: torch.zeros(
                batch, module.adapter.cfg.r_mem, module.adapter.cfg.key_dim,
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
                if isinstance(state, MemoryRead):
                    state.check_fp32()
                elif state.dtype != torch.float32:
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
    input_site: str = "o_proj",
) -> AdapterBank:
    from looplora.core.adapters import make_adapter
    from looplora.core.wrapper import AdaptedLinear
    from .residual import ResidualInputLinear

    if hasattr(model, "looplora_adapters"):
        raise ValueError("adapters are already installed")
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be nonempty and unique")
    if input_site not in {"o_proj", "residual"}:
        raise ValueError("input_site must be o_proj or residual")
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
        dimensions = model.config.hidden_size if input_site == "residual" else projection.in_features
        cfg = replace(config, d_in=dimensions, d_out=projection.out_features)
        # Initialize on CPU with a local seed, preserving the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(seed, "llm-adapter", layer))
            adapter = make_adapter(cfg).to(device=projection.weight.device, dtype=torch.float32)
        wrapper = (ResidualInputLinear(projection, adapter) if input_site == "residual"
                   else AdaptedLinear(projection, adapter))
        model.model.layers[layer].self_attn.o_proj = wrapper
        modules[layer] = wrapper
    bank = AdapterBank(modules)
    if input_site == "residual":
        for layer, module in modules.items():
            bank.input_hooks.append(model.model.layers[layer].register_forward_pre_hook(
                lambda _, args, wrapper=module: wrapper.capture(args[0])))
    model.looplora_adapters = bank
    return bank


def detach_adapters(model: nn.Module) -> AdapterBank:
    bank = model.looplora_adapters
    for handle in bank.input_hooks:
        handle.remove()
    bank.input_hooks.clear()
    for layer, module in bank.modules.items():
        model.model.layers[layer].self_attn.o_proj = module.base
    del model.looplora_adapters
    return bank
