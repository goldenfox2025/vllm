# SPDX-License-Identifier: Apache-2.0

# pylint: disable=unused-argument
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.adapter_commons.layers import AdapterMapping
from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.utils import divide
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import (
    LinearScalingRotaryEmbedding, RotaryEmbedding)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase

import os


def _get_lora_device(base_layer: nn.Module) -> torch.device:
    # code borrowed from https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/vllm/lora/layers.py#L34
    """Returns the device for where to place the LoRA tensors."""
    # unquantizedLinear
    if hasattr(base_layer, "weight"):
        return base_layer.weight.device
    # Compressed Tensor
    elif hasattr(base_layer, "weight_packed"):
        return base_layer.weight_packed.device
    # GPTQ/AWQ
    elif hasattr(base_layer, "qweight"):
        return base_layer.qweight.device
    # marlin
    elif hasattr(base_layer, "B"):
        return base_layer.B.device
    # HQQ marlin
    elif hasattr(base_layer, "W_q"):
        return base_layer.W_q.device
    else:
        raise ValueError(f"Unsupported base layer: {base_layer}")


def _not_fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of not using fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        decorate = kwargs.pop("decorate") if "decorate" in kwargs else True
        condition = (not kwargs["lora_config"].fully_sharded_loras
                     if decorate else True)
        return can_replace(*args, **kwargs) and condition

    return dec


@dataclass
class LoRAMapping(AdapterMapping):
    is_prefill: bool = False


class BaseLayerWithLoRA(nn.Module):

    def slice_lora_a(
        self, lora_a: Union[torch.Tensor, list[Union[torch.Tensor, None]]]
    ) -> Union[torch.Tensor, list[Union[torch.Tensor, None]]]:
        """Slice lora a if splitting for tensor parallelism."""
        ...

    def slice_lora_b(
        self, lora_b: Union[torch.Tensor, list[Union[torch.Tensor, None]]]
    ) -> Union[torch.Tensor, list[Union[torch.Tensor, None]]]:
        """Slice lora b if splitting with tensor parallelism."""
        ...

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.embeddings_slice: Optional[tuple[int, int]]
        self.embeddings_weights: Optional[torch.Tensor]

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:

        if self.base_layer.num_added_embeddings_per_partition > 0:
            # We can start adding lora weights
            self.embeddings_weights = self.base_layer.weight.data[
                self.base_layer.num_org_embeddings_per_partition:self.
                base_layer.num_org_embeddings_per_partition +
                self.base_layer.num_added_embeddings_per_partition]
            self.embeddings_slice = (
                self.base_layer.shard_indices.added_vocab_start_index -
                self.base_layer.org_vocab_size,
                self.base_layer.shard_indices.added_vocab_end_index -
                self.base_layer.org_vocab_size)
            self.base_layer.weight.data[
                self.base_layer.num_org_embeddings_per_partition:].fill_(0)
        else:
            self.embeddings_slice = None
            self.embeddings_weights = None

        self.embeddings_tensors = torch.zeros(
            (
                max_loras,
                lora_config.lora_extra_vocab_size,
                self.base_layer.embedding_dim,
            ),
            dtype=self.base_layer.weight.dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.org_vocab_size +
                lora_config.lora_extra_vocab_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                self.base_layer.embedding_dim,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked_2d = self.lora_a_stacked.view(
            self.lora_a_stacked.shape[0] * self.lora_a_stacked.shape[1],
            self.lora_a_stacked.shape[2],
        )

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        self.reset_lora(index)
        self.lora_a_stacked[index, :lora_a.shape[0], :lora_a.shape[1]].copy_(
            lora_a, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensors[
                index,
                :embeddings_tensor.shape[0],
                :embeddings_tensor.shape[1],
            ].copy_(embeddings_tensor, non_blocking=True)
            if self.embeddings_slice is not None:
                # TODO(yard1): Optimize this copy, we don't need to copy
                # everything, just the modified part
                embeddings = self.embeddings_tensors.view(
                    self.embeddings_tensors.shape[0] *
                    self.embeddings_tensors.shape[1],
                    self.embeddings_tensors.shape[2],
                )[self.embeddings_slice[0]:self.embeddings_slice[1]]
                assert self.embeddings_weights is not None
                self.embeddings_weights[:embeddings.shape[0]].copy_(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_tokens_mask = torch.where(x > self.base_layer.org_vocab_size - 1,
                                        1, 0)
        embeddings_indices = torch.narrow(
            self.punica_wrapper._embeddings_indices, 1, 0, x.size(0))

        indices = embeddings_indices[1]
        full_lora_a_embeddings = F.embedding(
            x + indices,
            self.lora_a_stacked_2d,
        )
        indices = embeddings_indices[0]
        full_output = self.base_layer.forward(x +
                                              (indices * added_tokens_mask))

        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(
                full_output.shape[0] * full_output.shape[1], -1)
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(
                full_lora_a_embeddings.shape[0] *
                full_lora_a_embeddings.shape[1],
                -1,
            )

        lora_output: Optional[
            torch.Tensor] = self.punica_wrapper.add_lora_embedding(
                full_output,
                full_lora_a_embeddings,
                self.lora_b_stacked,
                add_input=True)

        if not current_platform.can_update_inplace():
            full_output = lora_output

        return full_output.view_as(full_output_org)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is VocabParallelEmbedding

    @property
    def weight(self):
        return self.base_layer.weight


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.device = _get_lora_device(self.base_layer)
        self.lora_bias_stacked: Optional[tuple[torch.Tensor, ...]] = None

        self.output_slices: tuple[int, ...]
        self.tp_size: int
        self.output_size: int
        self.n_slices: int

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config
        #
        if isinstance(self.base_layer, ReplicatedLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, ColumnParallelLinear):
            lora_a_out_size = (lora_config.max_lora_rank if
                               not lora_config.fully_sharded_loras else divide(
                                   lora_config.max_lora_rank, self.tp_size))
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, RowParallelLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = (self.output_size if
                               not lora_config.fully_sharded_loras else divide(
                                   self.output_size, self.tp_size))
        else:
            raise NotImplementedError

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_out_size,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        if lora_config.bias_enabled:
            lora_bias_out_size = lora_b_out_size
            self.lora_bias_stacked = tuple(
                torch.zeros(
                    max_loras,
                    1,
                    lora_bias_out_size,
                    dtype=lora_config.lora_dtype,
                    device=self.device,
                ) for _ in range(self.n_slices))
        self.output_slices = (self.lora_b_stacked[0].shape[2], )

    def reset_lora(self, index: int):
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0
            if self.lora_config.bias_enabled:
                # Make mypy happy
                self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                              self.lora_bias_stacked)
                self.lora_bias_stacked[s_index][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert (len(self.lora_a_stacked) == len(self.lora_b_stacked) ==
                self.n_slices == 1)

        self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
            if lora_bias is not None:
                lora_bias = self.slice_bias(lora_bias)

        self.lora_a_stacked[0][index,
                               0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                   lora_a.T, non_blocking=True)
        self.lora_b_stacked[0][index,
                               0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                   lora_b.T, non_blocking=True)
        if lora_bias is not None:

            self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                          self.lora_bias_stacked)
            assert len(self.lora_bias_stacked)
            self.lora_bias_stacked[0][index, 0, :lora_bias.shape[0]].copy_(
                lora_bias.T, non_blocking=True)

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: Optional[
            torch.Tensor] = self.punica_wrapper.add_lora_linear(
                output, x, self.lora_a_stacked, self.lora_b_stacked,
                self.lora_bias_stacked, 1.0, self.output_slices)
        if not current_platform.can_update_inplace():
            output = lora_output

        return output

    @property
    def weight(self) -> torch.Tensor:

        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        # HQQ marlin
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None


class ReplicatedLinearWithLoRA(BaseLinearLayerWithLoRA):

    def __init__(self, base_layer: ReplicatedLinear) -> None:
        super().__init__(base_layer, )
        # To ensure interface compatibility, set to 1 always.
        self.tp_size = 1
        self.output_size = self.base_layer.output_size
        self.n_slices = 1

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of ReplicatedLinearWithLoRA

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output = self.apply(input_, bias)

        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)

        if not self.base_layer.return_bias:
            return output

        return output, output_bias

    # ReplicatedLinear should always be replaced, regardless of the fully
    # sharded LoRAs setting, because it is, by definition, copied per GPU.
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ReplicatedLinear


class ColumnParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    """
    LoRA on top of ColumnParallelLinear layer.
    LoRA B is sliced for tensor parallelism.
    There are two types for the `base_layer`:
    1. ColumnParallelLinear, e.g.`dense_h_to_4h` in `FalconForCausalLM`.
    2. MergedColumnParallelLinear, e.g.`gate_up_proj` in `Phi3ForCausalLM`.
    """

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__(base_layer)
        # The base_layer type is ColumnParallelLinear or
        # MergedColumnParallelLinear, their weight sharding logic is
        # inconsistent when TP is greater than 1.
        self.is_merged_col_linear = type(
            base_layer) is MergedColumnParallelLinear
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_size = self.base_layer.output_size_per_partition
        # There is only one LoRA layer
        self.n_slices = 1

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        # Applicable to cases where the base_layer is
        # MergedColumnParallelLinear.
        if self.is_merged_col_linear:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = self.output_size // 2
            offset = lora_b.shape[-1] // 2

            left_weight = lora_b[:, tp_rank * shard_size:(tp_rank + 1) *
                                 shard_size]
            right_weight = lora_b[:, offset + tp_rank * shard_size:offset +
                                  (tp_rank + 1) * shard_size]
            lora_b = torch.cat([left_weight, right_weight], dim=1)
        # Applicable to cases where the base_layer is
        # ColumnParallelLinear.
        else:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.output_size
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            lora_b = lora_b[:, start_idx:end_idx]
        return lora_b

    def slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        # TODO: Fix the slicing logic of bias.
        if bias is None:
            return bias
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        shard_size = self.output_size
        start_idx = tensor_model_parallel_rank * shard_size
        end_idx = (tensor_model_parallel_rank + 1) * shard_size
        bias = bias[start_idx:end_idx]
        return bias

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output_parallel = self.apply(input_, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.base_layer.return_bias:
            return output

        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1)


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(
        self, base_layer: Union[MergedColumnParallelLinear,
                                QKVParallelLinear]) -> None:
        super().__init__(base_layer)
        # There are two LoRA layers
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        # the output_sizes in MergedColumnParallelLinear is not sharded by tp
        # we need to divide it by the tp_size to get correct slices size
        output_sizes = self.base_layer.output_sizes
        self.output_slices = tuple(
            divide(output_size, self.tp_size) for output_size in output_sizes)
        self.n_slices = len(self.output_slices)
        self.output_ids = (self.tp_rank, ) * self.n_slices

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """
        The main reason for overriding this function is to enhance  code 
        maintainability.
        """
        self.lora_config = lora_config

        lora_a_output_size_per_partition = (
            lora_config.max_lora_rank if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size))

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                output_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for output_size in self.output_slices)
        if lora_config.bias_enabled:
            self.lora_bias_stacked = tuple(
                torch.zeros(
                    max_loras,
                    1,
                    output_size,
                    dtype=lora_config.lora_dtype,
                    device=self.device,
                ) for output_size in self.output_slices)

    def slice_lora_a(
        self, lora_a: list[Union[torch.Tensor, None]]
    ) -> list[Union[torch.Tensor, None]]:
        return lora_a

    def slice_lora_b(
        self, lora_b: list[Union[torch.Tensor, None]]
    ) -> list[Union[torch.Tensor, None]]:
        for i, (shard_id, shard_size) in enumerate(
                zip(self.output_ids, self.output_slices)):
            if (lora_b_i := lora_b[i]) is not None:
                lora_b[i] = lora_b_i[:, shard_size * shard_id:shard_size *
                                     (shard_id + 1)]
        return lora_b

    def slice_bias(
        self, bias: list[Union[torch.Tensor,
                               None]]) -> list[Union[torch.Tensor, None]]:
        for i, (shard_id, shard_size) in enumerate(
                zip(self.output_ids, self.output_slices)):
            if (bias_i := bias[i]) is not None:
                bias[i] = bias_i[shard_size * shard_id:shard_size *
                                 (shard_id + 1)]
        return bias

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
            if lora_bias is not None:
                lora_bias = self.slice_bias(lora_bias)

        for i in range(self.n_slices):
            if (lora_a_i := lora_a[i]) is not None:
                self.lora_a_stacked[i][
                    index, 0, :lora_a_i.shape[1], :lora_a_i.shape[0]].copy_(
                        lora_a_i.T, non_blocking=True)
            if (lora_b_i := lora_b[i]) is not None:
                self.lora_b_stacked[i][
                    index, 0, :lora_b_i.shape[1], :lora_b_i.shape[0]].copy_(
                        lora_b_i.T, non_blocking=True)

        if lora_bias is not None:
            self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                          self.lora_bias_stacked)
            for i in range(self.n_slices):
                if (lora_bias_i := lora_bias[i]) is not None:
                    self.lora_bias_stacked[i][index,
                                              0, :lora_bias_i.shape[0]].copy_(
                                                  lora_bias_i.T,
                                                  non_blocking=True)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (type(source_layer) is MergedColumnParallelLinear
                and len(packed_modules_list) == 2)


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    """
    ColumnParallelLinear layer that is specifically designed for
    qkv_proj. Certain models, such as chatglm3 and baichuan-7b,
    only contains a single LoRA within their qkv_proj layer.

    During inference with Tensor Parallel, the weights of lora_b
    must be accurately partitioned according to the respective ranks.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.q_proj_total_size = (self.base_layer.total_num_heads *
                                  self.base_layer.head_size)
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.kv_proj_total_size = (self.base_layer.total_num_kv_heads *
                                   self.base_layer.head_size)
        # There is only one LoRA layer
        self.n_slices = 1

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        self.q_shard_id = tp_rank
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        lora_b_q = lora_b[:, self.q_proj_shard_size *
                          self.q_shard_id:self.q_proj_shard_size *
                          (self.q_shard_id + 1)]
        k_offset = self.q_proj_total_size
        lora_b_k = lora_b[:, k_offset +
                          self.kv_proj_shard_size * self.kv_shard_id:k_offset +
                          self.kv_proj_shard_size * (self.kv_shard_id + 1)]
        v_offset = k_offset + self.kv_proj_total_size
        lora_b_v = lora_b[:, v_offset +
                          self.kv_proj_shard_size * self.kv_shard_id:v_offset +
                          self.kv_proj_shard_size * (self.kv_shard_id + 1)]
        lora_b = torch.cat([lora_b_q, lora_b_k, lora_b_v], dim=1)
        return lora_b

    def slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        bias_q = bias[self.q_proj_shard_size *
                      self.q_shard_id:self.q_proj_shard_size *
                      (self.q_shard_id + 1)]
        k_offset = self.q_proj_total_size
        bias_k = bias[k_offset +
                      self.kv_proj_shard_size * self.kv_shard_id:k_offset +
                      self.kv_proj_shard_size * (self.kv_shard_id + 1)]
        v_offset = k_offset + self.kv_proj_total_size
        bias_v = bias[v_offset +
                      self.kv_proj_shard_size * self.kv_shard_id:v_offset +
                      self.kv_proj_shard_size * (self.kv_shard_id + 1)]
        bias = torch.cat([bias_q, bias_k, bias_v], dim=1)
        return bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: list,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is QKVParallelLinear and len(
            packed_modules_list) == 1


class MergedQKVParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
    """MergedColumnParallelLinear layer that is composed of 3 sublayers (slices)
    packed together in qkv proj fashion
    (q_proj + k_proj + v_proj -> qkv_proj).

    This means we have 3 LoRAs, each applied to one slice of the layer.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(
        self, base_layer: Union[MergedColumnParallelLinear,
                                QKVParallelLinear]) -> None:
        super().__init__(base_layer)
        # There are three LoRA layer.
        self.n_slices = len(self.base_layer.output_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas

        self.output_slices = (
            self.q_proj_shard_size,
            self.kv_proj_shard_size,
            self.kv_proj_shard_size,
        )
        self.output_ids = (
            self.q_shard_id,
            self.kv_shard_id,
            self.kv_shard_id,
        )

        # 用于缓存懒加载的融合权重
        self.fused_qkv_lora_a_weight: Optional[torch.Tensor] = None
        # LoRA A权重的布局信息，用于fused_expand
        self.qkv_output_size = self.q_proj_shard_size + self.kv_proj_shard_size * 2
        
        # 计时相关变量
        # 使用环境变量控制：VLLM_ENABLE_TIMING=0 禁用计时，VLLM_ENABLE_TIMING=1 启用计时（默认）
        self.timing_enabled = os.environ.get("VLLM_ENABLE_TIMING", "1") == "1"
        self.total_calls = 0
        self.total_gemm_time = 0.0
        self.total_split_time = 0.0
        self.total_bias_time = 0.0
        self.total_expand_time = 0.0
        self.total_traditional_time = 0.0
        self.total_overall_time = 0.0
        # 传统方法的详细计时统计
        self.total_traditional_qkv_time = 0.0
        self.total_traditional_lora_time = 0.0
        self.traditional_calls = 0

    def _build_fused_qkv_lora_a_weight(self):
        # Base QKV 部分
        W_T = self.base_layer.weight.t().contiguous()
        in_features, out_features = W_T.shape

        all_weights = [W_T]
        
        # LoRA A 部分
        # lora_a_stacked 是一个元组，每个元素对应一个slice (Q, K, V)
        # 每个slice的形状是 [max_loras, rank, hidden_size]
        # 我们需要按 lora_id -> slice_id 的顺序拼接所有LoRA A的权重

        max_loras = self.lora_a_stacked[0].shape[0]
        all_lora_a_weights = []
        lora_slice_ranks_list = []

        # 遍历每个LoRA适配器
        for lora_id in range(max_loras):
            # 遍历每个slice (Q, K, V)
            for s in range(self.n_slices):
                # 获取当前 (lora_id, slice_id) 的LoRA A权重
                # 它的形状是 [rank, in_features]
                lora_a_slice = self.lora_a_stacked[s][lora_id][0]
                
                # 记录该slice的rank
                lora_slice_ranks_list.append(lora_a_slice.shape[0])

                if lora_a_slice.shape[0] > 0:
                    # 转置以匹配拼接维度 -> [in_features, rank]
                    all_lora_a_weights.append(lora_a_slice.t().contiguous())

        # 将所有有效的LoRA A权重拼接起来
        if all_lora_a_weights:
            fused_lora_a_weights = torch.cat(all_lora_a_weights, dim=1).contiguous()
            all_weights.append(fused_lora_a_weights)

        # 最终拼接成一个大的权重矩阵
        self.fused_qkv_lora_a_weight = torch.cat(all_weights, dim=1).contiguous()

        # --- 元数据计算 ---
        # 计算并保存 lora_slice_ranks 和 lora_a_slice_starts
        # 这两个张量描述了在大的融合LoRA A权重中，每个(lora_id, slice_id)对应的权重块的位置和大小(rank)
        
        # lora_slice_ranks: [max_loras * num_slices]
        self.lora_slice_ranks = torch.tensor(lora_slice_ranks_list, dtype=torch.int32, device=self.device)
        
        # lora_a_slice_starts: [max_loras * num_slices]
        # 使用cumsum高效计算起始位置（exclusive scan）
        # 这是每个LoRA A slice在融合A矩阵中的起始列索引
        lora_a_slice_starts_temp = torch.cumsum(self.lora_slice_ranks, dim=0, dtype=torch.int32)
        self.lora_a_slice_starts = torch.cat([
            torch.tensor([0], device=self.device, dtype=torch.int32), 
            lora_a_slice_starts_temp[:-1]
        ])

    def _compute_traditional(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """传统的QKV+LoRA计算方法，用于验证"""
        print(f"\n📋 传统方法设备检查:")
        print(f"   输入 x: shape={x.shape}, device={x.device}")
        if bias is not None:
            print(f"   bias: shape={bias.shape}, device={bias.device}")
        print(f"   base_layer.weight: shape={self.base_layer.weight.shape}, device={self.base_layer.weight.device}")
        
        # 传统方法的分阶段计时
        if self.timing_enabled:
            start_trad_qkv = torch.cuda.Event(enable_timing=True)
            end_trad_qkv = torch.cuda.Event(enable_timing=True)
            start_trad_lora = torch.cuda.Event(enable_timing=True)
            end_trad_lora = torch.cuda.Event(enable_timing=True)
            
            # Stage 1: 基础QKV线性变换
            start_trad_qkv.record()
        
        qkv_output = F.linear(x, self.base_layer.weight, bias)
        print(f"   QKV输出: shape={qkv_output.shape}, device={qkv_output.device}")
        
        if self.timing_enabled:
            end_trad_qkv.record()
            start_trad_lora.record()
        
        if self.lora_a_stacked is not None and len(self.lora_a_stacked) > 0:
            x_flat = x.flatten(0, 1) if x.ndim == 3 else x
            qkv_output_flat = qkv_output.flatten(0, 1) if qkv_output.ndim == 3 else qkv_output
            print(f"   LoRA输入扁平化: x_flat.device={x_flat.device}, qkv_output_flat.device={qkv_output_flat.device}")
            self.punica_wrapper.add_lora_linear(
                qkv_output_flat,
                    x_flat,
                    self.lora_a_stacked,
                    self.lora_b_stacked,
                    self.lora_bias_stacked,
                1.0, # scale
                    self.output_slices,
            )
        
        if self.timing_enabled:
            end_trad_lora.record()
            torch.cuda.synchronize()
            
            # 计算传统方法各阶段耗时
            trad_qkv_time = start_trad_qkv.elapsed_time(end_trad_qkv)
            trad_lora_time = start_trad_lora.elapsed_time(end_trad_lora)
            
            # 更新累计统计
            self.traditional_calls += 1
            self.total_traditional_qkv_time += trad_qkv_time
            self.total_traditional_lora_time += trad_lora_time
            
            # 打印传统方法的详细计时
            print(f"\n📋 Traditional Method Detailed Timing:")
            print(f"  Base QKV Linear:    {trad_qkv_time:.3f} ms")
            print(f"  LoRA Computation:   {trad_lora_time:.3f} ms")
            print(f"  Traditional Total:  {trad_qkv_time + trad_lora_time:.3f} ms")
            
            # 打印传统方法的累计平均耗时
            print(f"\n📊 Traditional Method Cumulative Average (over {self.traditional_calls} calls):")
            print(f"  Avg Base QKV Linear:  {self.total_traditional_qkv_time / self.traditional_calls:.3f} ms")
            print(f"  Avg LoRA Computation: {self.total_traditional_lora_time / self.traditional_calls:.3f} ms")
            print(f"  Avg Traditional Total: {(self.total_traditional_qkv_time + self.total_traditional_lora_time) / self.traditional_calls:.3f} ms")
            
        print(f"📋 传统方法最终输出: shape={qkv_output.shape}, device={qkv_output.device}")
        return qkv_output

    def _compare_and_validate_outputs(self, 
                                        fused_output: torch.Tensor, 
                                        traditional_output: torch.Tensor) -> None:
        """
        比较 fused 方法和传统方法的输出结果，如果不一致则报错退出
        """
        # 如果是 [batch, seq, dim]，先展开为 [batch*seq, dim]
        traditional_flat = traditional_output.flatten(0, 1) if traditional_output.ndim == 3 else traditional_output
        fused_flat = fused_output.flatten(0, 1) if fused_output.ndim == 3 else fused_output

        # 计算差异
        diff = torch.abs(fused_flat - traditional_flat)
        max_diff = torch.max(diff).item()
        is_close = torch.allclose(fused_flat, traditional_flat, rtol=1e-1, atol=1e-1)

        # 只在验证失败时输出详细信息
        if not is_close:
            print("🔍 QKV+LoRA 融合结果验证")
            print(f"Shape: {traditional_flat.shape}, Max diff: {max_diff:.2e}, 一致性: {'✅' if is_close else '❌'}")
            # 展示前10个
            print(f"前10个传统输出: {traditional_flat[:10]}")
            print(f"前10个融合输出: {fused_flat[:10]}")
            # 只展示最大的 N 个差异位置
            diff_flat = diff.view(-1)
            N = min(5, diff_flat.numel())
            topk = torch.topk(diff_flat, k=N)

            print("\n⚠️ 发现显著差异，展示 Top-5 最大差异: ")
            for i, (d_val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
                t_val = traditional_flat.view(-1)[idx].item()
                f_val = fused_flat.view(-1)[idx].item()
                print(f"  [{i+1}] idx={idx}: trad={t_val:.6f}, fused={f_val:.6f}, diff={d_val:.2e}")

            raise RuntimeError(f"❌ 验证失败：最大差异 {max_diff:.2e} 超出容差")

        # 验证通过时仅简单记录
        # print("✅ 验证通过：Traditional 与 Fused 输出一致")


    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        重写apply方法，使用"一次GEMM"的融合优化方案。
        核心修复：每次调用都重新构建融合权重，以解决权重陈旧问题。
        """
        # 下面的代码意思是：如果环境变量VLLM_ENABLE_QKV_LORA_FUSION为0，则不使用融合优化，使用传统方法
        if os.environ.get("VLLM_ENABLE_QKV_LORA_FUSION", "0") == "0":
            return self._compute_traditional(x, bias)
        
        # 检查是否启用终极融合内核
        if os.environ.get("VLLM_ENABLE_ULTIMATE_FUSION", "0") == "1":
            return self._compute_ultimate_fusion(x, bias)
            
        # CUDA Events for timing
        if self.timing_enabled:
            start_overall = torch.cuda.Event(enable_timing=True)
            end_overall = torch.cuda.Event(enable_timing=True)
            start_gemm = torch.cuda.Event(enable_timing=True)
            end_gemm = torch.cuda.Event(enable_timing=True)
            start_split = torch.cuda.Event(enable_timing=True)
            end_split = torch.cuda.Event(enable_timing=True)
            start_bias = torch.cuda.Event(enable_timing=True)
            end_bias = torch.cuda.Event(enable_timing=True)
            start_expand = torch.cuda.Event(enable_timing=True)
            end_expand = torch.cuda.Event(enable_timing=True)
            start_traditional = torch.cuda.Event(enable_timing=True)
            end_traditional = torch.cuda.Event(enable_timing=True)
            
            start_overall.record()
        
        # 关键修复：确保每次前向传播都重新构建最新的融合权重（不计时）
        self._build_fused_qkv_lora_a_weight()

        # 1. 准备输入
        x_flat = x.flatten(0, 1) if x.ndim == 3 else x

        # Stage 1: 执行一次大的GEMM
        if self.timing_enabled:
            start_gemm.record()
        
        fused_output_matmul = torch.matmul(x_flat, self.fused_qkv_lora_a_weight)
        
        if self.timing_enabled:
            end_gemm.record()

        # Stage 2: 拆分GEMM的输出
        if self.timing_enabled:
            start_split.record()
            
        qkv_output_fused = fused_output_matmul[:, :self.qkv_output_size].contiguous()
        lora_a_output = fused_output_matmul[:, self.qkv_output_size:].contiguous()
        
        if self.timing_enabled:
            end_split.record()

        # Stage 3: 添加bias
        if self.timing_enabled:
            start_bias.record()
            
        if bias is not None:
            qkv_output_fused = qkv_output_fused + bias
            
        if self.timing_enabled:
            end_bias.record()

        # 4. 准备调用自定义的融合expand CUDA内核所需的元数据
        from vllm.lora.punica_wrapper.cuda_punica.fused_expand_ctypes_wrapper import cuda_fused_qkv_expand_interface
        
        num_tokens = x_flat.shape[0]
        meta_args = self.punica_wrapper.token_mapping_meta.meta_args(num_tokens)
        (
            _,
            token_indices_sorted,         
            num_tokens_per_lora,          
            lora_token_start_loc,         
            lora_ids,                     
            no_lora_flag,                 
        ) = meta_args

        # Stage 4: 调用自定义的融合expand内核
        if self.timing_enabled:
            start_expand.record()
            
        success = cuda_fused_qkv_expand_interface(
            fused_matmul_output=lora_a_output,
            output_tensor=qkv_output_fused,
            lora_b_stacked=self.lora_b_stacked,
            lora_bias_stacked=self.lora_bias_stacked,
            output_slices=self.output_slices,
            lora_a_slice_starts=self.lora_a_slice_starts,
            lora_slice_ranks=self.lora_slice_ranks,
            token_indices_sorted=token_indices_sorted,
            num_tokens_per_lora=num_tokens_per_lora,
            lora_token_start_loc=lora_token_start_loc,
            lora_ids=lora_ids,
            qkv_output_size=self.qkv_output_size,
            no_lora_flag=no_lora_flag,
        )
        
        
        if self.timing_enabled:
            end_expand.record()
        
        if not success:
            raise RuntimeError("❌ Fused expand kernel failed")

        # Stage 5: (可选) 验证正确性
        verify_enabled = os.environ.get("VLLM_VERIFY_FUSED_LORA", "0") == "1"
        if verify_enabled:
            if self.timing_enabled:
                start_traditional.record()
                
            qkv_output_traditional = self._compute_traditional(x, bias)
            
            if self.timing_enabled:
                end_traditional.record()
                
            self._compare_and_validate_outputs(qkv_output_fused, qkv_output_traditional)
        
        final_output = qkv_output_fused.view_as(x) if x.ndim == 3 else qkv_output_fused
        
        if self.timing_enabled:
            end_overall.record()
            
            # 等待所有CUDA操作完成
            torch.cuda.synchronize()
            
            # 计算各阶段耗时
            gemm_time = start_gemm.elapsed_time(end_gemm)
            split_time = start_split.elapsed_time(end_split)
            bias_time = start_bias.elapsed_time(end_bias)
            expand_time = start_expand.elapsed_time(end_expand)
            overall_time = start_overall.elapsed_time(end_overall)
            traditional_time = 0.0
            if verify_enabled:
                traditional_time = start_traditional.elapsed_time(end_traditional)
            
            # 累计统计
            self.total_calls += 1
            self.total_gemm_time += gemm_time
            self.total_split_time += split_time
            self.total_bias_time += bias_time
            self.total_expand_time += expand_time
            self.total_traditional_time += traditional_time
            self.total_overall_time += overall_time
            
            # 打印当前轮次的详细计时
            print(f"\n=== 🚀 Fused QKV LoRA Timing Report (Call #{self.total_calls}) ===")
            print(f"Stage 1 - Large GEMM:     {gemm_time:.3f} ms")
            print(f"Stage 2 - Output Split:   {split_time:.3f} ms") 
            print(f"Stage 3 - Add Bias:       {bias_time:.3f} ms")
            print(f"Stage 4 - Fused Expand:   {expand_time:.3f} ms")
            if verify_enabled:
                print(f"Stage 5 - Traditional:    {traditional_time:.3f} ms")
            print(f"Overall Time:             {overall_time:.3f} ms")
            
            # 打印累计平均耗时
            print(f"\n📊 Cumulative Average Timing (over {self.total_calls} calls):")
            print(f"Avg Large GEMM:     {self.total_gemm_time / self.total_calls:.3f} ms")
            print(f"Avg Output Split:   {self.total_split_time / self.total_calls:.3f} ms")
            print(f"Avg Add Bias:       {self.total_bias_time / self.total_calls:.3f} ms") 
            print(f"Avg Fused Expand:   {self.total_expand_time / self.total_calls:.3f} ms")
            if self.total_traditional_time > 0:
                print(f"Avg Traditional:    {self.total_traditional_time / self.total_calls:.3f} ms")
            print(f"Avg Overall:        {self.total_overall_time / self.total_calls:.3f} ms")
            
            # 如果有传统方法的详细统计，显示对比分析
            if self.traditional_calls > 0:
                avg_fused = (self.total_gemm_time + self.total_split_time + self.total_bias_time + self.total_expand_time) / self.total_calls
                avg_traditional_total = (self.total_traditional_qkv_time + self.total_traditional_lora_time) / self.traditional_calls
                speedup = avg_traditional_total / avg_fused if avg_fused > 0 else 0.0
                
                print(f"\n🔥 Method Comparison Analysis:")
                print(f"Fused Method Core (GEMM+Split+Bias+Expand): {avg_fused:.3f} ms")
                print(f"Traditional Method (QKV+LoRA):              {avg_traditional_total:.3f} ms")
                print(f"Speedup Ratio:                               {speedup:.2f}x")
                if speedup > 1.0:
                    print(f"✅ Fused method is {speedup:.2f}x FASTER than traditional")
                elif speedup < 1.0:
                    print(f"⚠️  Traditional method is {1/speedup:.2f}x FASTER than fused")
                else:
                    print(f"⚖️  Both methods have similar performance")
            
            print(f"=== ✅ End Timing Report ===\n")
        
        return final_output

    def _compute_ultimate_fusion(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """使用终极融合内核的计算方法"""
        try:
            from vllm.lora.punica_wrapper.cuda_punica.ultimate_fusion_ctypes_wrapper import cuda_ultimate_fusion_interface
            
            print("\n🚀 使用终极融合内核...")
            print(f"🔍 输入维度调试: x.shape={x.shape}, x.device={x.device}")
            if bias is not None:
                print(f"🔍 Bias维度调试: bias.shape={bias.shape}, bias.device={bias.device}")
            
            # 准备输入
            x_flat = x.flatten(0, 1) if x.ndim == 3 else x
            num_tokens = x_flat.shape[0]
            
            print(f"🔍 处理后输入: x_flat.shape={x_flat.shape}, x_flat.device={x_flat.device}, num_tokens={num_tokens}")
            
            # 获取Punica元数据
            meta_args = self.punica_wrapper.token_mapping_meta.meta_args(num_tokens)
            (
                _,
                token_indices_sorted,         
                num_tokens_per_lora,          
                lora_token_start_loc,         
                lora_ids,                     
                no_lora_flag,                 
            ) = meta_args
            
            print(f"🔍 Punica元数据: lora_ids={lora_ids}, lora_ids.device={lora_ids.device}")
            print(f"🔍 Token映射: token_indices_sorted.shape={token_indices_sorted.shape}, device={token_indices_sorted.device}")
            print(f"🔍 其他元数据设备:")
            print(f"   num_tokens_per_lora.device={num_tokens_per_lora.device}")
            print(f"   lora_token_start_loc.device={lora_token_start_loc.device}")
            
            # 准备QKV权重（使用原始权重，不需要转置）
            qkv_weights = self.base_layer.weight # [qkv_output_size, hidden_size]
            print(f"🔍 QKV权重: qkv_weights.shape={qkv_weights.shape}, device={qkv_weights.device}")
            print(f"🔍 输出切片: output_slices={self.output_slices}")
            
            # 检查LoRA权重维度和设备
            print(f"🔍 LoRA A维度和设备:")
            for i, lora_a in enumerate(self.lora_a_stacked):
                print(f"   slice {i}: shape={lora_a.shape}, device={lora_a.device}")
            print(f"🔍 LoRA B维度和设备:")
            for i, lora_b in enumerate(self.lora_b_stacked):
                print(f"   slice {i}: shape={lora_b.shape}, device={lora_b.device}")
            
            # 调用终极融合内核
            print("🔧 准备调用终极融合内核...")
            output = cuda_ultimate_fusion_interface(
                inputs=x_flat,
                qkv_weights=qkv_weights,
                lora_a_stacked=self.lora_a_stacked,
                lora_b_stacked=self.lora_b_stacked,
                output_slices=self.output_slices,
                token_indices_sorted=token_indices_sorted,
                num_tokens_per_lora=num_tokens_per_lora,
                lora_token_start_loc=lora_token_start_loc,
                lora_ids=lora_ids,
            )
            
            print(f"✅ 终极融合内核输出: output.shape={output.shape}, device={output.device}")
            
            # 添加bias（现在内核稳定了，可以安全处理bias）
            if bias is not None:
                print(f"🔧 添加bias: bias.shape={bias.shape}, output.shape={output.shape}")
                # bias应该和output的最后一个维度匹配
                if bias.shape[0] == output.shape[1]:
                    output = output + bias
                    print(f"✅ Bias添加成功: 最终output.shape={output.shape}")
                else:
                    print(f"⚠️ Bias维度不匹配: bias.shape[0]={bias.shape[0]}, output.shape[1]={output.shape[1]}")
                    print(f"🔄 跳过bias添加以避免错误")
            else:
                print("📋 没有bias需要添加")
            
            # 恢复原始形状
            final_output = output.view_as(x) if x.ndim == 3 else output
            print(f"✅ 最终输出: final_output.shape={final_output.shape}, device={final_output.device}")
            
            print("✅ 终极融合内核计算完成!")
            return final_output
            
        except Exception as e:
            print(f"❌ 终极融合内核失败: {e}")
            print("🔄 回退到传统方法...")
            # 回退到传统方法
            return self._compute_traditional(x, bias)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (type(source_layer) is QKVParallelLinear
                and len(packed_modules_list) == 3)


class LinearScalingRotaryEmbeddingWithLoRA(BaseLayerWithLoRA):
    """Implements RoPE-scaled embeddings with linear scaling for
    multiple LoRA adapters with a specialized kernel.

    Replace LinearScalingRotaryEmbedding with MultiLinearScalingRotaryEmbedding
    which can handle multi lora adapters in a specialied kernel.
    """

    def __init__(self, base_layer: RotaryEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer

    @property
    def scaling_factors(self):
        return self.base_layer.scaling_factors

    @property
    def rotary_dim(self):
        return self.base_layer.rotary_dim

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        scaling_factors = (list(lora_config.long_lora_scaling_factors)
                           if lora_config.long_lora_scaling_factors else [])
        base_scaling_factor = (self.base_layer.scaling_factor if isinstance(
            self.base_layer, LinearScalingRotaryEmbedding) else 1.0)
        scaling_factors = sorted(
            list(set([base_scaling_factor] + scaling_factors)))
        self.base_layer = LinearScalingRotaryEmbedding(
            self.base_layer.head_size,
            self.base_layer.rotary_dim,
            self.base_layer.max_position_embeddings,
            self.base_layer.base,
            self.base_layer.is_neox_style,
            scaling_factors,
            self.base_layer.dtype,
        )

    def reset_lora(self, index: int):
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        ...

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base_layer(
            positions,
            query,
            key,
            offsets=self.punica_wrapper.long_lora_indices,
        )

    @property
    def scaling_factor_to_offset(self) -> dict[float, int]:
        return self.base_layer.scaling_factor_to_offset

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        return (type(source_layer) is LinearScalingRotaryEmbedding
                or type(source_layer) is RotaryEmbedding)

    def extra_repr(self) -> str:
        return self.base_layer.extra_repr()


#TODO: Implement this
class QKVCrossParallelLinearWithLoRA(BaseLayerWithLoRA):
    pass


class RowParallelLinearWithLoRA(BaseLinearLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__(base_layer)

        self.tp_size = get_tensor_model_parallel_world_size()
        # reset input_size
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size

        self.tp_rank = get_tensor_model_parallel_rank()
        # There is only one LoRA layer.
        self.n_slices = 1

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:

        shard_size = self.input_size
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_a = lora_a[start_idx:end_idx, :]
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        return lora_b

    def slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        return bias

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            # TODO: simplify code below
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias

        if not self.base_layer.return_bias:
            return output

        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear


class LogitsProcessorWithLoRA(BaseLayerWithLoRA):
    """
    LoRA wrapper for LogitsProcessor, with extra logic to handle the
    application of the LoRA adapter and added LoRA vocabulary.

    Args:
        base_layer: LogitsProcessor layer
        hidden_size: hidden size of the model
        dtype: data type of the model
        device: device of the model
        sharded_to_full_mapping: index mapping from sharded vocab to full vocab
            received from base_layer.get_sharded_to_full_mapping(). If None,
            no reindexing will be done.
    """

    def __init__(self, base_layer: LogitsProcessor, hidden_size: int,
                 dtype: torch.dtype, device: torch.device,
                 sharded_to_full_mapping: Optional[list[int]]) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sharded_to_full_mapping = sharded_to_full_mapping

    @property
    def logits_as_input(self):
        return self.base_layer.logits_as_input

    @property
    def vocab_size(self):
        return self.base_layer.vocab_size

    @property
    def scale(self):
        return self.base_layer.scale

    @property
    def soft_cap(self):
        return self.base_layer.soft_cap

    @property
    def use_all_gather(self):
        return self.base_layer.use_all_gather

    @property
    def org_vocab_size(self):
        return self.base_layer.org_vocab_size

    @property
    def include_gpu_probs_tensor(self):
        return self.base_layer.include_gpu_probs_tensor

    @property
    def should_modify_greedy_probs_inplace(self):
        return self.base_layer.should_modify_greedy_probs_inplace

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        # TODO: Verify if this condition can be further relaxed
        if 32000 < self.base_layer.vocab_size > 257024:
            raise ValueError("When using LoRA, vocab size must be "
                             "32000 >= vocab_size <= 257024")
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                # Pad for kernel compatibility
                math.ceil(self.base_layer.vocab_size /
                          lora_config.lora_vocab_padding_size) *
                lora_config.lora_vocab_padding_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.embeddings_tensors = torch.full(
            (max_loras, lora_config.lora_extra_vocab_size, self.hidden_size),
            fill_value=float("-inf"),
            dtype=self.dtype,
            device=self.device,
        )
        if self.sharded_to_full_mapping is not None:
            self.sharded_to_full_mapping_gpu = torch.tensor(
                self.sharded_to_full_mapping,
                device=self.device,
                dtype=torch.long)
        else:
            self.sharded_to_full_mapping_gpu = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = float("-inf")

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        self.reset_lora(index)
        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensors[
                index,
                :embeddings_tensor.shape[0],
                :embeddings_tensor.shape[1],
            ] = embeddings_tensor

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states)
        if embedding_bias is not None:
            logits += embedding_bias

        # Gather logits for TP
        logits = self.base_layer._gather_logits(logits)

        if logits is None:
            return None

        if self.sharded_to_full_mapping_gpu is not None:
            # Reindex full logits tensor to ensure 1:1 mapping between
            # index and token_id
            # Example for:
            #   org_vocab_size = 4
            #   added_vocab_size = 2
            #   pad_to_size = 8
            #   tp_size = 2

            # indices:  [0, 1, 2,  3, 4, 5, 6,  7]
            # token_id: [0, 1, 4, -1, 2, 3, 5, -1]

            # Therefore, the mapping is expected to be:
            # [0, 1, 4, 6, 2, 3, 5, 7] so that when we reindex,
            # we get:
            # indices:  [0, 1, 2, 3, 4, 5,  6,  7]
            # token_id: [0, 1, 2, 3, 4, 5, -1, -1]
            logits = logits[:, self.sharded_to_full_mapping_gpu]

        lora_logits = torch.empty(
            self.embeddings_tensors.shape[0] + 1,
            self.embeddings_tensors.shape[1],
            hidden_states.shape[0],
            dtype=self.embeddings_tensors.dtype,
            device=self.embeddings_tensors.device,
        )
        torch.matmul(self.embeddings_tensors,
                     hidden_states.T,
                     out=lora_logits[:-1])

        neg_inf, pos_inf = current_platform.get_infinity_values(
            lora_logits.dtype)

        lora_logits[-1] = neg_inf
        lora_logits = lora_logits.mT
        indices_padded = self.punica_wrapper.sampler_indices_padded

        if current_platform.is_tpu():
            indices_padded = indices_padded[:logits.size(0)]

        lora_logits = (lora_logits.reshape(
            lora_logits.shape[0] * lora_logits.shape[1],
            lora_logits.shape[2],
        ).index_select(0, indices_padded).nan_to_num_(nan=neg_inf,
                                                      posinf=pos_inf,
                                                      neginf=neg_inf))

        # HPU needs special handling to prune out dummy samples.
        if current_platform.is_hpu():
            lora_logits = lora_logits[:logits.shape[0], :]

        logits[:,
               self.base_layer.org_vocab_size:self.base_layer.org_vocab_size +
               lora_logits.shape[1]] = lora_logits

        lora_output: Optional[
            torch.Tensor] = self.punica_wrapper.add_lora_logits(
                logits, hidden_states, self.lora_a_stacked,
                self.lora_b_stacked, 1.0)

        if not current_platform.can_update_inplace():
            logits = lora_output

        # Remove paddings in vocab (if any).
        logits = logits[:, :self.base_layer.vocab_size]
        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # Special handling for the LogitsProcessor.
        return False
