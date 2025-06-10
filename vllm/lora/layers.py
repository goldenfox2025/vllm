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

    def __init__(self, base_layer: QKVParallelLinear) -> None:
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

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """
        The main reason for overloading this function is to handle inconsistent 
        weight dimensions in qkv lora.
        """
        super().create_lora_weights(max_loras, lora_config, model_config)
    
    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """重写apply方法以支持QKV+LoRA融合"""
        print(f"🎯 [QKV+LoRA Fusion] apply方法被调用 - 输入形状: {x.shape}")
        print(f"🎯 [QKV+LoRA Fusion] 当前类: {self.__class__.__name__}")
        
        # 检查环境变量
        import os
        fusion_enabled = os.environ.get("VLLM_ENABLE_QKV_LORA_FUSION", "0")
        enable_timing = os.environ.get("VLLM_ENABLE_LORA_TIMING", "0") == "1"
        print(f"🎯 [QKV+LoRA Fusion] 环境变量 VLLM_ENABLE_QKV_LORA_FUSION = {fusion_enabled}")
        print(f"🎯 [QKV+LoRA Fusion] 性能测量 VLLM_ENABLE_LORA_TIMING = {enable_timing}")
        
        # 检查LoRA权重状态（仅用于调试，不影响融合决策）
        print(f"🎯 [QKV+LoRA Fusion] n_slices = {self.n_slices}")
        for i in range(self.n_slices):
            lora_sum = self.lora_a_stacked[i].abs().sum().item()
            print(f"🎯 [QKV+LoRA Fusion] LoRA A[{i}] 权重总和: {lora_sum}")
        
        # 如果启用融合，始终尝试融合计算（不管LoRA权重是否为0）
        if fusion_enabled == "1":
            try:
                print("🚀 [QKV+LoRA Fusion] 开始融合计算（不管LoRA权重值）")
                
                if enable_timing:
                    # 带性能测量的计算（允许回退）
                    return self._compute_with_timing(x, bias)
                else:
                    # 正常计算模式：正确性优先，验证失败则报错
                    print("⚡ [QKV+LoRA Fusion] 正确性优先模式：验证失败将报错退出")
                    
                    # 计算传统方法的结果用于验证
                    traditional_output = self._compute_traditional_method(x, bias)
                    
                    # 计算融合方法的结果
                    fused_output = self._fused_computation(x, bias)
                    
                    # 验证结果一致性
                    if self._verify_outputs(traditional_output, fused_output, rtol=1e-2, atol=2.0):
                        print("✅ [QKV+LoRA Fusion] 融合计算结果验证通过，使用融合结果")
                        return fused_output
                    else:
                        # 正确性优先：验证失败直接报错，不回退
                        error_msg = (
                            f"❌ [QKV+LoRA Fusion] 融合计算结果验证失败！\n"
                            f"传统方法输出统计: min={traditional_output.min():.6f}, "
                            f"max={traditional_output.max():.6f}, mean={traditional_output.mean():.6f}\n"
                            f"融合方法输出统计: min={fused_output.min():.6f}, "
                            f"max={fused_output.max():.6f}, mean={fused_output.mean():.6f}\n"
                            f"最大绝对差异: {torch.max(torch.abs(traditional_output - fused_output)).item():.6f}\n"
                            f"这表明融合实现存在错误，需要修复后再使用。"
                        )
                        print(error_msg)
                        raise RuntimeError(error_msg)
                    
            except Exception as e:
                if enable_timing:
                    # 性能测量模式：允许回退
                    print(f"⚠️  [QKV+LoRA Fusion] 融合计算出错: {e}，回退到传统方法")
                    return self._compute_traditional_method(x, bias)
                else:
                    # 正确性优先模式：直接抛出异常
                    error_msg = f"❌ [QKV+LoRA Fusion] 融合计算发生异常: {e}"
                    print(error_msg)
                    raise RuntimeError(error_msg) from e
        
        # 默认使用传统方法
        return self._compute_traditional_method(x, bias)
    
    def _compute_with_timing(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """带详细性能测量的计算"""
        print(f"\n⏱️  [性能测量] 开始详细计时分析")
        
        # 创建CUDA事件用于精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        print("🔥 [性能测量] Warmup阶段...")
        for _ in range(3):
            _ = self._compute_traditional_method(x, bias)
            _ = self._fused_computation(x, bias)
        torch.cuda.synchronize()
        
        # 测量传统方法
        print("📊 [性能测量] 测量传统方法...")
        traditional_times = self._measure_traditional_method(x, bias, num_iterations=10)
        
        # 测量融合方法
        print("📊 [性能测量] 测量融合方法...")
        fused_times, fused_output = self._measure_fused_method(x, bias, num_iterations=10)
        
        # 输出详细的性能报告
        self._print_performance_report(traditional_times, fused_times)
        
        return fused_output
    
    def _measure_traditional_method(self, x: torch.Tensor, bias: Optional[torch.Tensor], num_iterations: int = 10) -> dict:
        """测量传统方法的各个阶段耗时"""
        import os
        
        # 暂时禁用CUDA LoRA kernel以确保使用Triton（传统方法+Triton LoRA是绝对正确的基准）
        original_cuda_flag = os.environ.get("VLLM_FORCE_TRITON_LORA", "0")
        os.environ["VLLM_FORCE_TRITON_LORA"] = "1"
        
        try:
            qkv_times = []
            shrink_times = []
            expand_times = []
            total_times = []
            
            for i in range(num_iterations):
                start_total = torch.cuda.Event(enable_timing=True)
                end_total = torch.cuda.Event(enable_timing=True)
                
                start_qkv = torch.cuda.Event(enable_timing=True)
                end_qkv = torch.cuda.Event(enable_timing=True)
                
                start_shrink = torch.cuda.Event(enable_timing=True)
                end_shrink = torch.cuda.Event(enable_timing=True)
                
                start_expand = torch.cuda.Event(enable_timing=True)
                end_expand = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.synchronize()
                start_total.record()
                
                # 处理批次维度
                x_flat = x.flatten(0, 1) if x.ndim == 3 else x
                
                # 1. QKV计算
              
                start_qkv.record()
                # qkv_output = self.base_layer.quant_method.apply(self.base_layer, x_flat, bias)
                qkv_output = torch.nn.functional.linear(x_flat, self.base_layer.weight, bias)
                end_qkv.record()
                
                # 2. LoRA shrink - 使用Triton kernel（绝对正确的基准）
                start_shrink.record()
                # 创建buffer用于shrink结果
                r = self.lora_b_stacked[0].size(-1)
                buffer = torch.zeros(
                    (len(self.output_slices), x_flat.size(0), r),
                    dtype=torch.float32,
                    device=x_flat.device,
                )
                self.punica_wrapper.add_shrink(
                    buffer,
                    x_flat,
                    self.lora_a_stacked,
                    1.0
                )
                end_shrink.record()
                
                # 3. LoRA expand
                start_expand.record()
                lora_output = self.punica_wrapper.add_expand(
                    qkv_output,
                    buffer,
                    self.lora_b_stacked,
                    self.lora_bias_stacked,
                    self.output_slices,
                    add_inputs=True
                )
                end_expand.record()
                
                end_total.record()
                torch.cuda.synchronize()
                
                qkv_time = start_qkv.elapsed_time(end_qkv)
                shrink_time = start_shrink.elapsed_time(end_shrink)
                expand_time = start_expand.elapsed_time(end_expand)
                total_time = start_total.elapsed_time(end_total)
                
                qkv_times.append(qkv_time)
                shrink_times.append(shrink_time)
                expand_times.append(expand_time)
                total_times.append(total_time)
            
            return {
                'qkv_times': qkv_times,
                'shrink_times': shrink_times,
                'expand_times': expand_times,
                'total_times': total_times,
                'method': 'traditional'
            }
        finally:
            # 恢复原始设置
            os.environ["VLLM_FORCE_TRITON_LORA"] = original_cuda_flag
    
    def _measure_fused_method(self, x: torch.Tensor, bias: Optional[torch.Tensor], num_iterations: int = 10) -> tuple[dict, torch.Tensor]:
        """测量融合方法的各个阶段耗时"""
        fused_matmul_times = []
        expand_times = []
        total_times = []
        build_weight_times = []
        split_bias_times = []
        final_output = None
        
        for i in range(num_iterations):
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            
            start_build = torch.cuda.Event(enable_timing=True)
            end_build = torch.cuda.Event(enable_timing=True)
            
            start_fused = torch.cuda.Event(enable_timing=True)
            end_fused = torch.cuda.Event(enable_timing=True)
            
            start_split = torch.cuda.Event(enable_timing=True)
            end_split = torch.cuda.Event(enable_timing=True)
            
            start_expand = torch.cuda.Event(enable_timing=True)
            end_expand = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_total.record()
            
            # 处理批次维度
            x_flat = x.flatten(0, 1) if x.ndim == 3 else x
            
            # 1. 构建融合权重（现在计时！）
            start_build.record()
            slice_has_lora = [True] * self.n_slices
            fused_weight, lora_rank_info = self._build_qkv_lora_fused_weight(x_flat.device, x_flat.dtype, slice_has_lora)
            end_build.record()
            
            # 2. 融合的matmul计算（纯计算部分）
            start_fused.record()
            fused_output = torch.matmul(x_flat, fused_weight)
            end_fused.record()
            
            # 3. 分拆和bias处理（现在计时！）
            start_split.record()
            qkv_part, lora_shrink_parts = self._split_qkv_lora_output(fused_output, lora_rank_info)
            if bias is not None:
                qkv_part = qkv_part + bias
            end_split.record()
            
            # 4. LoRA expand
            start_expand.record()
            if lora_shrink_parts is not None and len(lora_rank_info) > 0:
                # 调用新的fused expand方法，直接处理融合shrink结果
                self.punica_wrapper.add_fused_expand(
                    qkv_part,                    # y: QKV输出，会被就地修改
                    lora_shrink_parts,           # fused_shrink_input: 融合计算的shrink结果
                    self.lora_b_stacked,         # lora_b权重
                    self.lora_bias_stacked,      # lora_bias权重  
                    self.output_slices,          # 输出分片
                    lora_rank_info,              # slice rank信息
                    offset_start=0,
                    add_inputs=True              # 累加到QKV结果上
                )
            end_expand.record()
            
            end_total.record()
            torch.cuda.synchronize()
            
            build_time = start_build.elapsed_time(end_build)
            fused_time = start_fused.elapsed_time(end_fused)
            split_time = start_split.elapsed_time(end_split)
            expand_time = start_expand.elapsed_time(end_expand)
            total_time = start_total.elapsed_time(end_total)
            
            build_weight_times.append(build_time)
            fused_matmul_times.append(fused_time)
            split_bias_times.append(split_time)
            expand_times.append(expand_time)
            total_times.append(total_time)
            
            if i == num_iterations - 1:
                final_output = qkv_part
        
        times_dict = {
            'build_weight_times': build_weight_times,
            'fused_matmul_times': fused_matmul_times,
            'split_bias_times': split_bias_times,
            'expand_times': expand_times,
            'total_times': total_times,
            'method': 'fused'
        }
        
        return times_dict, final_output
    
    def _print_performance_report(self, traditional_times: dict, fused_times: dict):
        """打印详细的性能报告"""
        import numpy as np
        
        print(f"\n📈 [性能报告] QKV+LoRA计算性能对比")
        print(f"=" * 80)
        
        # 传统方法统计
        trad_qkv_avg = np.mean(traditional_times['qkv_times'])
        trad_shrink_avg = np.mean(traditional_times['shrink_times'])
        trad_expand_avg = np.mean(traditional_times['expand_times'])
        trad_total_avg = np.mean(traditional_times['total_times'])
        
        # 融合方法统计
        fused_build_avg = np.mean(fused_times['build_weight_times'])
        fused_matmul_avg = np.mean(fused_times['fused_matmul_times'])
        fused_split_avg = np.mean(fused_times['split_bias_times'])
        fused_expand_avg = np.mean(fused_times['expand_times'])
        fused_total_avg = np.mean(fused_times['total_times'])
        
        print(f"🔵 传统方法 (QKV + LoRA Shrink + LoRA Expand):")
        print(f"   QKV计算:      {trad_qkv_avg:.3f} ms")
        print(f"   LoRA Shrink:  {trad_shrink_avg:.3f} ms")
        print(f"   LoRA Expand:  {trad_expand_avg:.3f} ms")
        print(f"   总计:         {trad_total_avg:.3f} ms")
        print(f"   验证总和:     {trad_qkv_avg + trad_shrink_avg + trad_expand_avg:.3f} ms")
        print(f"")
        
        print(f"🟢 融合方法 (详细时间分解):")
        print(f"   构建融合权重: {fused_build_avg:.3f} ms")
        print(f"   融合Matmul:   {fused_matmul_avg:.3f} ms (纯计算)")
        print(f"   分拆+Bias:    {fused_split_avg:.3f} ms")
        print(f"   LoRA Expand:  {fused_expand_avg:.3f} ms")
        print(f"   总计:         {fused_total_avg:.3f} ms")
        print(f"   验证总和:     {fused_build_avg + fused_matmul_avg + fused_split_avg + fused_expand_avg:.3f} ms")
        print(f"")
        
        # 🔍 计算复杂度分析
        print(f"🧮 计算复杂度分析:")
        
        # 获取实际的矩阵维度
        qkv_output_size = sum(self.output_slices)  # QKV输出维度
        input_size = self.input_size  # 输入维度 
        total_lora_rank = self.n_slices * self.lora_a_stacked[0].shape[2]  # 总LoRA rank
        fused_output_size = qkv_output_size + total_lora_rank
        
        print(f"   输入维度: {input_size}")
        print(f"   QKV输出维度: {qkv_output_size}")
        print(f"   LoRA总rank: {total_lora_rank} (每slice: {self.lora_a_stacked[0].shape[2]}, 共{self.n_slices}个slice)")
        print(f"   融合输出维度: {fused_output_size}")
        
        # 计算理论FLOPs
        # 传统方法：QKV matmul + LoRA shrink + LoRA expand
        qkv_flops = 2 * input_size * qkv_output_size  # 2 for multiply+add
        lora_shrink_flops = 2 * input_size * total_lora_rank
        lora_expand_flops = 2 * total_lora_rank * qkv_output_size
        traditional_total_flops = qkv_flops + lora_shrink_flops + lora_expand_flops
        
        # 融合方法：大matmul + LoRA expand
        fused_matmul_flops = 2 * input_size * fused_output_size
        fused_total_flops = fused_matmul_flops + lora_expand_flops  # expand部分相同
        
        print(f"   传统方法理论FLOPs:")
        print(f"     QKV: 2×{input_size}×{qkv_output_size} = {qkv_flops:,}")
        print(f"     LoRA Shrink: 2×{input_size}×{total_lora_rank} = {lora_shrink_flops:,}")
        print(f"     LoRA Expand: 2×{total_lora_rank}×{qkv_output_size} = {lora_expand_flops:,}")
        print(f"     总计: {traditional_total_flops:,}")
        print(f"   融合方法理论FLOPs:")
        print(f"     融合Matmul: 2×{input_size}×{fused_output_size} = {fused_matmul_flops:,}")
        print(f"     LoRA Expand: {lora_expand_flops:,} (同传统)")
        print(f"     总计: {fused_total_flops:,}")
        
        # 理论vs实际性能分析
        flops_ratio = traditional_total_flops / fused_total_flops
        actual_ratio = trad_total_avg / fused_total_avg
        
        print(f"   理论FLOPs比率: {flops_ratio:.3f} (传统/融合)")
        print(f"   实际时间比率: {actual_ratio:.3f} (传统/融合)")
        
        # 🚨 异常分析
        print(f"\n🔍 性能异常分析:")
        qkv_vs_fused_ratio = trad_qkv_avg / fused_matmul_avg
        qkv_alone_flops = qkv_flops
        fused_alone_flops = fused_matmul_flops
        qkv_alone_ratio = qkv_alone_flops / fused_alone_flops
        
        print(f"   单独计算对比:")
        print(f"     传统QKV时间: {trad_qkv_avg:.3f}ms")
        print(f"     融合Matmul时间: {fused_matmul_avg:.3f}ms")
        print(f"     实际速度比: {qkv_vs_fused_ratio:.3f}x")
        print(f"     理论FLOPs比: {qkv_alone_ratio:.3f}x (QKV FLOPs / 融合 FLOPs)")
        
        if qkv_vs_fused_ratio > 1.5:
            print(f"   ✨ 融合matmul意外地比QKV计算快 {qkv_vs_fused_ratio:.1f}倍！")
            print(f"      可能原因:")
            print(f"      1. GPU内存带宽利用率：较大矩阵获得更好的带宽利用")
            print(f"      2. CUDA kernel启动开销摊销：大计算摊销启动成本")
            print(f"      3. 数据局部性：连续大矩阵访问模式更优")
            print(f"      4. GPU计算单元利用率：更大并行度更好利用SM")
            print(f"      5. 内存合并访问：更好的内存访问模式")
        elif qkv_vs_fused_ratio < 0.8:
            print(f"   ⚠️  融合matmul比QKV计算慢，这符合预期（计算量更大）")
        else:
            print(f"   ⚖️  融合matmul与QKV计算时间接近，在合理范围内")
        
        # 时间差异分析
        fused_calculated_total = fused_build_avg + fused_matmul_avg + fused_split_avg + fused_expand_avg
        time_diff = abs(fused_total_avg - fused_calculated_total)
        if time_diff > 0.01:  # 如果差异超过0.01ms
            print(f"\n⚠️  时间测量差异: {time_diff:.3f} ms (可能有未归类的开销)")
        else:
            print(f"\n✅ 时间测量一致性验证通过 (差异: {time_diff:.3f} ms)")
        
        # 计算加速比
        if trad_total_avg > 0:
            speedup = trad_total_avg / fused_total_avg
            print(f"\n⚡ 性能提升:")
            print(f"   总体加速比:   {speedup:.2f}x")
            print(f"   时间节省:     {trad_total_avg - fused_total_avg:.3f} ms ({((trad_total_avg - fused_total_avg) / trad_total_avg * 100):.1f}%)")
            
            # 更详细的分析
            print(f"\n🔍 详细分析:")
            print(f"   传统计算时间: QKV({trad_qkv_avg:.3f}) + Shrink({trad_shrink_avg:.3f}) = {trad_qkv_avg + trad_shrink_avg:.3f}ms")
            print(f"   融合计算时间: Build({fused_build_avg:.3f}) + Matmul({fused_matmul_avg:.3f}) + Split({fused_split_avg:.3f}) = {fused_build_avg + fused_matmul_avg + fused_split_avg:.3f}ms")
            
            # 核心计算对比（排除构建开销）
            trad_compute = trad_qkv_avg + trad_shrink_avg
            fused_compute = fused_matmul_avg  # 纯matmul时间
            compute_speedup = trad_compute / fused_compute if fused_compute > 0 else 0
            
            print(f"   纯计算加速比: {trad_compute:.3f}ms → {fused_compute:.3f}ms (加速 {compute_speedup:.2f}x)")
            print(f"   Expand阶段对比: {trad_expand_avg:.3f}ms → {fused_expand_avg:.3f}ms")
            
            if speedup > 1.05:
                print(f"   ✅ 融合优化有效！总体加速 {(speedup-1)*100:.1f}%")
            elif speedup > 0.95:
                print(f"   ⚖️  融合优化效果中性 (±5%范围内)")
            else:
                print(f"   ⚠️  融合优化出现性能下降 {(1-speedup)*100:.1f}%")
                print(f"      可能原因：构建权重开销({fused_build_avg:.3f}ms)过大")
        
        print(f"=" * 80)

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

    def _compute_traditional_method(
        self, 
        x: torch.Tensor, 
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算传统的非融合方法，用于对比验证"""
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        
        # 处理批次维度
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
    
    def _verify_outputs(
        self, 
        traditional_output: torch.Tensor, 
        fused_output: torch.Tensor, 
        rtol: float = 1e-2, 
        atol: float = 2.0
    ) -> bool:
        """验证融合计算和传统计算的结果一致性"""
        try:
            # 检查形状
            if traditional_output.shape != fused_output.shape:
                print(f"❌ [QKV+LoRA Fusion] 输出形状不匹配: traditional {traditional_output.shape} vs fused {fused_output.shape}")
                return False
            
            # 检查数值差异
            max_diff = torch.max(torch.abs(traditional_output - fused_output)).item()
            rel_diff = torch.max(torch.abs((traditional_output - fused_output) / (traditional_output + 1e-8))).item()
            
            print(f"🔍 [QKV+LoRA Fusion] 输出验证:")
            print(f"   Traditional统计: min={traditional_output.min():.6f}, max={traditional_output.max():.6f}, mean={traditional_output.mean():.6f}")
            print(f"   Fused统计: min={fused_output.min():.6f}, max={fused_output.max():.6f}, mean={fused_output.mean():.6f}")
            print(f"   最大绝对差异: {max_diff:.6f}")
            print(f"   最大相对差异: {rel_diff:.6f}")
            
            # 使用torch.allclose进行验证
            is_close = torch.allclose(traditional_output, fused_output, rtol=rtol, atol=atol)
            
            if is_close:
                print(f"✅ [QKV+LoRA Fusion] 输出验证通过 (rtol={rtol}, atol={atol})")
            else:
                print(f"❌ [QKV+LoRA Fusion] 输出验证失败 (rtol={rtol}, atol={atol})")
            
            return is_close
            
        except Exception as e:
            print(f"❌ [QKV+LoRA Fusion] 输出验证出错: {e}")
            return False

    def _fused_computation(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """融合的QKV+LoRA计算"""
        print(f"🚀 [QKV+LoRA Fusion] Starting fused computation for {x.shape[0]} tokens")
        
        # 处理批次维度
        if x.ndim == 3:
            x = x.flatten(0, 1)
        
        # Step 1: 检查每个slice的LoRA权重状态（仅用于调试，始终处理所有slice）
        slice_has_lora = []
        for i in range(self.n_slices):
            # 注意：即使权重为0，也认为"有LoRA"，因为这是LoRA层
            # LoRA权重为0可能是warmup阶段或其他原因，但仍需要参与计算
            has_lora = True  # 始终为True，因为这是LoRA层
            slice_has_lora.append(has_lora)
            lora_sum = self.lora_a_stacked[i].abs().sum().item()
            print(f"🔍 [QKV+LoRA Fusion] Slice {i} LoRA权重总和: {lora_sum} (强制处理)")
        
        print(f"🔧 [QKV+LoRA Fusion] 所有slice都将参与融合计算: {slice_has_lora}")
        
        # Step 2: 构建融合权重矩阵（处理所有slice）
        fused_weight, lora_rank_info = self._build_qkv_lora_fused_weight(x.device, x.dtype, slice_has_lora)
        
        if fused_weight is None:
            print("⚠️ [QKV+LoRA Fusion] Failed to build fused weight, fallback to traditional")
            return self._compute_traditional_method(x, bias)
        
        # Step 3: 执行融合的matmul计算
        fused_output = self._compute_qkv_lora_fused(x, fused_weight)
        
        # Step 4: 分拆融合输出
        qkv_part, lora_shrink_parts = self._split_qkv_lora_output(fused_output, lora_rank_info)
        
        # Step 5: 应用bias到QKV部分
        if bias is not None:
            qkv_part = qkv_part + bias
        
        # Step 6: 处理LoRA expand（所有slice都参与）
        if lora_shrink_parts is not None and len(lora_rank_info) > 0:
            print(f"🔄 [QKV+LoRA Fusion] Processing LoRA expand with shrink shape: {lora_shrink_parts.shape}")
            
            print(f"🚀 [QKV+LoRA Fusion] Calling fused expand: QKV shape {qkv_part.shape}, shrink shape {lora_shrink_parts.shape}")
            
            # 调用新的fused expand操作
            # 注意：lora_shrink_parts的格式是 [num_tokens, total_lora_rank]
            # 其中 total_lora_rank = max_loras * (slice0_rank + slice1_rank + slice2_rank) 
            # 但由于当前的融合构建只处理单个LoRA的slice，实际是 slice0_rank + slice1_rank + slice2_rank
            self.punica_wrapper.add_fused_expand(
                qkv_part,                    # y: QKV输出，会被就地修改
                lora_shrink_parts,           # fused_shrink_input: 融合计算的shrink结果 [num_tokens, total_lora_rank]
                self.lora_b_stacked,         # lora_b权重
                self.lora_bias_stacked,      # lora_bias权重  
                self.output_slices,          # 输出分片
                lora_rank_info,              # slice rank信息，kernel内部会重新计算真实偏移
                offset_start=0,
                add_inputs=True              # 累加到QKV结果上
            )
            
            print(f"✅ [QKV+LoRA Fusion] Fused expand completed, final output shape: {qkv_part.shape}")
        
        print(f"✅ [QKV+LoRA Fusion] Completed fused computation")
        return qkv_part

    def _build_qkv_lora_fused_weight(self, device: torch.device, dtype: torch.dtype, slice_has_lora: list) -> tuple[Optional[torch.Tensor], list]:
        """构建融合的QKV+LoRA权重矩阵"""
        try:
            # 获取QKV权重并转置到正确格式
            qkv_weight = self.base_layer.weight  # [output_size_per_partition, input_size_per_partition]
            qkv_weight = qkv_weight.T  # 转置为 [input_size_per_partition, output_size_per_partition]
            print(f"🔧 [QKV+LoRA Fusion] QKV weight shape after transpose: {qkv_weight.shape}")
            
            # 收集所有slice的LoRA A权重和rank信息（包括权重为0的）
            lora_a_weights = []
            lora_rank_info = []
            current_col = 0  # 正确累加列位置
            
            for i in range(self.n_slices):
                lora_a = self.lora_a_stacked[i]  # [max_loras, 1, lora_rank, input_size]
                print(f"🔧 [QKV+LoRA Fusion] LoRA A[{i}] raw shape: {lora_a.shape}")
                
                # 处理每个slice（不管权重是否为0）
                # 重塑为2D: [lora_rank, input_size]，然后转置为 [input_size, lora_rank]
                lora_a_2d = lora_a[0, 0]  # [lora_rank, input_size]
                valid_lora_a = lora_a_2d.T  # [input_size, lora_rank]
                print(f"🔧 [QKV+LoRA Fusion] LoRA A[{i}] processed shape: {valid_lora_a.shape}")
                
                lora_a_weights.append(valid_lora_a)
                lora_rank_info.append({
                    'slice_idx': i,
                    'rank': valid_lora_a.shape[1],  # lora_rank
                    'start_col': current_col
                })
                current_col += valid_lora_a.shape[1]  # 累加rank大小
            
            # 拼接所有LoRA A权重 
            all_lora_a = torch.cat(lora_a_weights, dim=1)  # [input_size, total_lora_rank]
            print(f"🔧 [QKV+LoRA Fusion] All LoRA A concatenated shape: {all_lora_a.shape}")
            
            # 打印rank信息用于调试
            for info in lora_rank_info:
                print(f"🔧 [QKV+LoRA Fusion] Slice {info['slice_idx']}: rank={info['rank']}, start_col={info['start_col']}")
            
            # 确保维度兼容性
            if qkv_weight.shape[0] != all_lora_a.shape[0]:
                print(f"❌ [QKV+LoRA Fusion] Dimension mismatch: QKV {qkv_weight.shape[0]} vs LoRA {all_lora_a.shape[0]}")
                return None, []
            
            # 构建融合权重矩阵: [input_size, qkv_output_size + total_lora_rank]
            fused_weight = torch.cat([qkv_weight, all_lora_a], dim=1)
            print(f"🔧 [QKV+LoRA Fusion] Fused weight shape: {fused_weight.shape}")
            
            return fused_weight, lora_rank_info
            
        except Exception as e:
            print(f"❌ [QKV+LoRA Fusion] Error building fused weight: {e}")
            return None, []

    def _compute_qkv_lora_fused(self, x: torch.Tensor, fused_weight: torch.Tensor) -> torch.Tensor:
        """执行融合的matmul计算"""
        # 一次大的matmul替代多个小的计算
        fused_output = torch.matmul(x, fused_weight)  # [num_tokens, qkv_output_size + total_lora_rank]
        
        print(f"🧮 [QKV+LoRA Fusion] Fused matmul: {x.shape} × {fused_weight.shape} = {fused_output.shape}")
        return fused_output

    def _split_qkv_lora_output(self, fused_output: torch.Tensor, lora_rank_info: list) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """分拆融合输出为QKV部分和LoRA shrink部分"""
        qkv_output_size = sum(self.output_slices)
        
        # 分拆
        qkv_part = fused_output[:, :qkv_output_size]
        
        if fused_output.shape[1] > qkv_output_size and lora_rank_info:
            lora_shrink_part = fused_output[:, qkv_output_size:]
            print(f"📊 [QKV+LoRA Fusion] Split output - QKV: {qkv_part.shape}, LoRA shrink: {lora_shrink_part.shape}")
            return qkv_part, lora_shrink_part
        else:
            return qkv_part, None

    def _reconstruct_shrink_for_expand(self, lora_shrink_parts: torch.Tensor, lora_rank_info: list, slice_has_lora: list) -> torch.Tensor:
        """重构shrink结果以匹配punica expand接口"""
        # punica expand期望的格式：[num_slices, num_tokens, lora_rank]
        num_tokens = lora_shrink_parts.shape[0]
        
        # 为每个slice创建对应的shrink结果
        slice_results = []
        for i in range(self.n_slices):
            # 查找这个slice对应的LoRA rank信息（现在所有slice都应该有info）
            slice_info = None
            for info in lora_rank_info:
                if info['slice_idx'] == i:
                    slice_info = info
                    break
            
            if slice_info is not None:
                # 提取对应的shrink部分
                start_col = slice_info['start_col']
                end_col = start_col + slice_info['rank']
                slice_shrink = lora_shrink_parts[:, start_col:end_col]  # [num_tokens, rank]
                slice_results.append(slice_shrink)
                print(f"🔄 [QKV+LoRA Fusion] Slice {i} shrink: {slice_shrink.shape} (from cols {start_col}:{end_col})")
            else:
                # 如果找不到info，说明代码有问题，但为了兼容性还是创建零矩阵
                print(f"⚠️ [QKV+LoRA Fusion] 警告：找不到slice {i}的rank信息，使用默认")
                if hasattr(self.lora_a_stacked[i], 'shape') and len(self.lora_a_stacked[i].shape) >= 3:
                    rank = self.lora_a_stacked[i].shape[2]  # [max_loras, 1, rank, input_size]
                else:
                    rank = 64  # 默认rank
                zero_shrink = torch.zeros(num_tokens, rank, device=lora_shrink_parts.device, dtype=lora_shrink_parts.dtype)
                slice_results.append(zero_shrink)
                print(f"🔄 [QKV+LoRA Fusion] Slice {i} 使用零矩阵: {zero_shrink.shape}")
        
        # 堆叠成期望的格式
        reconstructed = torch.stack(slice_results, dim=0)  # [num_slices, num_tokens, lora_rank]
        
        print(f"🔄 [QKV+LoRA Fusion] Reconstructed shrink tensor: {reconstructed.shape}")
        return reconstructed


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
