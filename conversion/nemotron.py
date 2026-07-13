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
class NemotronHModel(TextModel):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Extract block_configs and mtp_block_configs from hparams
        block_configs = self.hparams.get("block_configs", [])
        mtp_block_configs = self.hparams.get("mtp_block_configs", [])

        # Build combined layers_block_type from both lists
        layers_block_type = []
        for cfg in block_configs:
            layers_block_type.append(cfg.get("block_type", "default"))
        for cfg in mtp_block_configs:
            layers_block_type.append(cfg.get("block_type", "mtp"))

        # Call GraniteHybridModel.__init__ instead of NemotronHModel.__init__
        from .granite import GraniteHybridModel
        GraniteHybridModel.__init__(self, *args, **kwargs)

        # Set head_dim and d_inner
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

    def modify_tensors(self):
        # Handle tensors starting with "mtp." specially
        mtp_tensor_map = {}

        for tensor_name, tensor in list(self.tensors.items()):
            if tensor_name.startswith("mtp."):
                # Map MTP tensors to their correct names
                if tensor_name == "mtp.eh_proj":
                    mtp_tensor_map["eh_proj"] = tensor
                elif tensor_name == "mtp.enorm":
                    mtp_tensor_map["enorm"] = tensor
                elif tensor_name == "mtp.hnorm":
                    mtp_tensor_map["hnorm"] = tensor
                elif tensor_name == "mtp.final_layernorm":
                    mtp_tensor_map["final_layernorm"] = tensor
                else:
                    # For other MTP tensors, rewrite names to use backbone.layers.{mtp_bid} pattern
                    # Extract the block index and remaining name
                    parts = tensor_name.split(".")[1:]  # Skip "mtp"
                    if len(parts) >= 2 and parts[0].isdigit():
                        mtp_bid = parts[0]
                        remaining = ".".join(parts[1:])
                        new_name = f"backbone.layers.{mtp_bid}.{remaining}"
                        self.tensors[new_name] = tensor
                    else:
                        # Keep original name if pattern doesn't match
                        new_name = "backbone." + tensor_name
                        self.tensors[new_name] = tensor

                # Remove the original mtp. tensor
                del self.tensors[tensor_name]

        # Call super().modify_tensors() for the remaining tensors
        super().modify_tensors()
