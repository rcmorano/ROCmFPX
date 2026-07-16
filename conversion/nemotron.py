from __future__ import annotations

import json
import math

from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("NemotronHForCausalLM")
class NemotronHModel(GraniteHybridModel):
    model_arch = gguf.MODEL_ARCH.NEMOTRON_H
    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # Nemotron-specific parameters
        self.gguf_writer.add_layer_norm_type("rms")

    def modify_tensors(self):
        # Handle Nemotron-specific tensor transformations
        pass


@ModelBase.register("NemotronHPuzzleForCausalLM")
class NemotronHPuzzleModel(NemotronHModel):
    model_arch = gguf.MODEL_ARCH.NEMOTRON_H_MOE
    is_moe = True

    def __init__(self, dir_model: Path, *args, **kwargs):
        # Load hparams if not provided
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, self.is_mistral_format)

        # Check for block_configs in top-level or text_config (nested)
        block_configs = hparams.get("block_configs", [])
        mtp_block_configs = hparams.get("mtp_block_configs", [])
        if not block_configs and "text_config" in hparams:
            text_cfg = hparams["text_config"]
            block_configs = text_cfg.get("block_configs", [])
            mtp_block_configs = text_cfg.get("mtp_block_configs", [])

        # Derive n_layer_all for parent initialization
        if "n_layer_all" not in hparams:
            hparams["n_layer_all"] = len(block_configs) + len(mtp_block_configs)

        # Set n_layers from n_layer_all to ensure parent class can find block count
        # This is needed because TextModel.find_hparam checks for n_layers first
        hparams["n_layers"] = hparams["n_layer_all"]

        # Build combined layers_block_type from both lists BEFORE calling super().__init__()
        # This ensures the pattern length matches block_count.
        layers_block_type = []
        for cfg in block_configs:
            layers_block_type.append(cfg.get("block_type", "default"))
        for cfg in mtp_block_configs:
            layers_block_type.append(cfg.get("block_type", "mtp"))

        # Set the pattern in top-level hparams
        hparams["layers_block_type"] = layers_block_type

        # If text_config exists, also set it there to survive the merge in TextModel.__init__
        if "text_config" in hparams:
            hparams["text_config"]["layers_block_type"] = layers_block_type
            # Also set n_layers in text_config to survive merge
            hparams["text_config"]["n_layers"] = hparams["n_layers"]

        # Pass modified hparams to parent initialization
        kwargs["hparams"] = hparams

        # Call super().__init__ which will go through the MRO chain
        super().__init__(dir_model, *args, **kwargs)

        # Set head_dim and d_inner - these should be set after super().__init__ has run
        self.head_dim = self.hparams.get("head_dim", 64)
        self.d_inner = self.hparams.get("d_inner", 4096)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # Build combined all_block_configs list
        all_block_configs = []
        if "block_configs" in self.hparams:
            all_block_configs.extend(self.hparams["block_configs"])
        if "mtp_block_configs" in self.hparams:
            all_block_configs.extend(self.hparams["mtp_block_configs"])

        # Extract ffn_lengths and experts_used from the combined configs
        ffn_lengths = [cfg.get("ffn_length", 0) for cfg in all_block_configs]
        experts_used = [cfg.get("experts_used", 0) for cfg in all_block_configs]

        # Add expert parameters
        self.gguf_writer.add_feed_forward_length(ffn_lengths)
        self.gguf_writer.add_expert_feed_forward_length(ffn_lengths)
        self.gguf_writer.add_expert_used_count(experts_used)

        # Add other expert parameters as needed
        if self._experts is not None:
            for i, expert in enumerate(self._experts):
                if "w1" in expert:
                    self.gguf_writer.add_expert_weight(i, 0, "w1", expert["w1"])
                if "w2" in expert:
                    self.gguf_writer.add_expert_weight(i, 0, "w2", expert["w2"])
                if "w3" in expert:
                    self.gguf_writer.add_expert_weight(i, 0, "w3", expert["w3"])

        # Add nextn predict layers
        self.gguf_writer.add_nextn_predict_layers(len(self.mtp_block_configs))

    def modify_tensors(self, data_torch, name, bid):
        # Handle tensors starting with "mtp." specially
        if name.startswith("mtp."):
            # Map MTP tensors to their correct names
            if name == "mtp.eh_proj":
                new_name = "eh_proj"
            elif name == "mtp.enorm":
                new_name = "enorm"
            elif name == "mtp.hnorm":
                new_name = "hnorm"
            elif name == "mtp.final_layernorm":
                new_name = "final_layernorm"
            else:
                # For other MTP tensors, rewrite names to use backbone.layers.{mtp_bid} pattern
                parts = name.split(".")[1:]  # Skip "mtp"
                if len(parts) >= 2 and parts[0].isdigit():
                    mtp_bid = parts[0]
                    remaining = ".".join(parts[1:])
                    new_name = f"backbone.layers.{mtp_bid}.{remaining}"
                else:
                    new_name = "backbone." + name

            yield from super().modify_tensors(data_torch, new_name, bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)
