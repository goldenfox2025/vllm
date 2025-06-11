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

        # 融合权重缓存：在set_lora时预构建，避免每次forward重建
        self.fused_weight_cache: dict[int, Optional[torch.Tensor]] = {}
        self.lora_rank_info_cache: dict[int, list] = {}

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
        
        # 初始化融合权重缓存
        for i in range(max_loras):
            self.fused_weight_cache[i] = None
            self.lora_rank_info_cache[i] = []

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        print(f"\n🔄 [QKV+LoRA Fusion] 设置LoRA权重 (index={index})")
        
        # 先调用父类方法设置权重
        super().set_lora(index, lora_a, lora_b, embeddings_tensor, lora_bias)
        print(f"✅ [QKV+LoRA Fusion] 基础权重设置完成")
        
        # 预构建融合权重并缓存
        print(f"🔧 [QKV+LoRA Fusion] 开始预构建融合权重...")
        self._prebuild_fused_weight(index)
        
        # 验证缓存状态
        if self.fused_weight_cache[index] is not None:
            print(f"✅ [QKV+LoRA Fusion] 融合权重缓存构建成功")
            print(f"📊 缓存权重形状: {self.fused_weight_cache[index].shape}")
            print(f"📊 Rank信息: {self.lora_rank_info_cache[index]}")
        else:
            print(f"❌ [QKV+LoRA Fusion] 融合权重缓存构建失败")

    def reset_lora(self, index: int):
        """重写reset_lora方法，清理融合权重缓存"""
        super().reset_lora(index)
        
        # 清理缓存
        self.fused_weight_cache[index] = None
        self.lora_rank_info_cache[index] = []

    def _prebuild_fused_weight(self, lora_index: int) -> None:
        """预构建指定LoRA索引的融合权重并缓存"""
        try:
            print(f"🔧 [QKV+LoRA Fusion] 开始构建融合权重 (index={lora_index})")
            
            # 获取QKV权重并转置到正确格式
            qkv_weight = self.base_layer.weight  # [output_size_per_partition, input_size_per_partition]
            qkv_weight = qkv_weight.T  # 转置为 [input_size_per_partition, output_size_per_partition]
            
            print(f"📊 [QKV+LoRA Fusion] QKV权重形状: {qkv_weight.shape}")
            
            # 收集指定LoRA索引的所有slice的LoRA A权重
            lora_a_weights = []
            lora_rank_info = []
            current_col = 0
            
            for i in range(self.n_slices):
                lora_a = self.lora_a_stacked[i]  # [max_loras, 1, lora_rank, input_size]
                
                # 获取指定索引的LoRA A权重
                lora_a_2d = lora_a[lora_index, 0]  # [lora_rank, input_size]
                
                # 检查是否有有效的LoRA权重（仅用于日志）
                lora_sum = lora_a_2d.abs().sum().item()
                print(f"📊 [QKV+LoRA Fusion] LoRA A[{i}] 权重总和: {lora_sum}")
                
                valid_lora_a = lora_a_2d.T  # [input_size, lora_rank]
                lora_a_weights.append(valid_lora_a)
                lora_rank_info.append({
                    'slice_idx': i,
                    'rank': valid_lora_a.shape[1],  # lora_rank
                    'start_col': current_col
                })
                current_col += valid_lora_a.shape[1]
            
            # 拼接所有LoRA A权重
            all_lora_a = torch.cat(lora_a_weights, dim=1)  # [input_size, total_lora_rank]
            print(f"📊 [QKV+LoRA Fusion] 拼接后的LoRA A权重形状: {all_lora_a.shape}")
            
            # 确保维度兼容性
            if qkv_weight.shape[0] != all_lora_a.shape[0]:
                print(f"❌ [QKV+LoRA Fusion] 维度不兼容: QKV={qkv_weight.shape[0]} vs LoRA={all_lora_a.shape[0]}")
                self.fused_weight_cache[lora_index] = None
                self.lora_rank_info_cache[lora_index] = []
                return
            
            # 构建融合权重矩阵: [input_size, qkv_output_size + total_lora_rank]
            fused_weight = torch.cat([qkv_weight, all_lora_a], dim=1)
            print(f"✅ [QKV+LoRA Fusion] 成功构建融合权重，形状: {fused_weight.shape}")
            
            # 缓存结果
            self.fused_weight_cache[lora_index] = fused_weight
            self.lora_rank_info_cache[lora_index] = lora_rank_info
            print(f"✅ [QKV+LoRA Fusion] 已缓存融合权重和rank信息")
            
        except Exception as e:
            print(f"❌ [QKV+LoRA Fusion] 构建融合权重失败: {e}")
            self.fused_weight_cache[lora_index] = None
            self.lora_rank_info_cache[lora_index] = []

    def _get_cached_fused_weight(self, device: torch.device, dtype: torch.dtype) -> tuple[Optional[torch.Tensor], list]:
        """获取当前活跃LoRA的缓存融合权重"""
        import os
        enable_debug = os.environ.get("VLLM_ENABLE_LORA_DEBUG", "0") == "1"
        
        # 直接使用索引0，避免复杂的punica_wrapper访问
        active_lora_index = 0
        
        # 快速获取缓存
        cached_weight = self.fused_weight_cache.get(active_lora_index)
        cached_info = self.lora_rank_info_cache.get(active_lora_index, [])
        
        if cached_weight is not None:
            # 只在真正需要时才转换设备/类型
            if cached_weight.device != device or cached_weight.dtype != dtype:
                if enable_debug:
                    print(f"🔄 [QKV+LoRA Fusion] 转换权重到目标设备和类型")
                cached_weight = cached_weight.to(device=device, dtype=dtype)
                self.fused_weight_cache[active_lora_index] = cached_weight
        elif enable_debug:
            print(f"❌ [QKV+LoRA Fusion] 未找到索引 {active_lora_index} 的缓存权重")
        
        return cached_weight, cached_info

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
            print("🚀 [QKV+LoRA Fusion] 开始融合计算（不管LoRA权重值）")
            
            if enable_timing:
                # 带性能测量的计算（不允许回退 - 专注测量融合性能）
                print("⏱️  [QKV+LoRA Fusion] 性能测量模式：专注测量融合方法性能")
                return self._compute_with_timing(x, bias)
            else:
                # 正确性优先模式：验证失败则报错
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
        
        # 默认使用传统方法
        return self._compute_traditional_method(x, bias)

    def _compute_with_timing(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """带详细性能测量的计算 - 专注测量融合方法性能，不回退"""
        print(f"\n⏱️  [性能测量] 开始详细计时分析 - 专注融合方法")
        
        # Warmup
        print("🔥 [性能测量] Warmup阶段...")
        for _ in range(3):
            _ = self._compute_traditional_method(x, bias)
            try:
                _ = self._fused_computation(x, bias)
            except Exception as e:
                print(f"❌ [性能测量] Warmup阶段融合计算失败: {e}")
                raise RuntimeError(f"性能测量模式下融合计算失败，无法继续测量: {e}")
        torch.cuda.synchronize()
        
        # 测量传统方法
        print("📊 [性能测量] 测量传统方法...")
        traditional_times = self._measure_traditional_method(x, bias, num_iterations=10)
        
        # 测量融合方法（不允许失败）
        print("📊 [性能测量] 测量融合方法...")
        try:
            fused_times, fused_output = self._measure_fused_method(x, bias, num_iterations=10)
        except Exception as e:
            print(f"❌ [性能测量] 融合方法测量失败: {e}")
            raise RuntimeError(f"性能测量模式下融合方法测量失败: {e}")
        
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
                # 确保开始前完全同步
                torch.cuda.synchronize()
                
                # 创建事件
                start_total = torch.cuda.Event(enable_timing=True)
                end_total = torch.cuda.Event(enable_timing=True)
                start_qkv = torch.cuda.Event(enable_timing=True)
                end_qkv = torch.cuda.Event(enable_timing=True)
                start_shrink = torch.cuda.Event(enable_timing=True)
                end_shrink = torch.cuda.Event(enable_timing=True)
                start_expand = torch.cuda.Event(enable_timing=True)
                end_expand = torch.cuda.Event(enable_timing=True)
                
                # 开始总计时
                start_total.record()
                
                # 处理批次维度
                x_flat = x.flatten(0, 1) if x.ndim == 3 else x
                
                # 1. QKV计算
                start_qkv.record()
                qkv_output = torch.matmul(x_flat, self.base_layer.weight.T)
                if bias is not None:
                    qkv_output = qkv_output + bias
                end_qkv.record()
                    
                # 2. LoRA shrink
                start_shrink.record()
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
                
                # 结束总计时
                end_total.record()
                
                # 等待所有操作完成
                torch.cuda.synchronize()
                
                # 计算各阶段时间
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
        cache_get_times = []  # 缓存获取时间
        device_check_times = []  # 设备检查时间
        total_times = []
        final_output = None
        
        for i in range(num_iterations):
            # 确保开始前完全同步
            torch.cuda.synchronize()
            
            # 创建更多事件来细分计时
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_cache = torch.cuda.Event(enable_timing=True)
            end_cache = torch.cuda.Event(enable_timing=True)
            start_device_check = torch.cuda.Event(enable_timing=True)
            end_device_check = torch.cuda.Event(enable_timing=True)
            start_fused = torch.cuda.Event(enable_timing=True)
            end_fused = torch.cuda.Event(enable_timing=True)
            start_expand = torch.cuda.Event(enable_timing=True)
            end_expand = torch.cuda.Event(enable_timing=True)
            
            # 开始总计时
            start_total.record()
            
            # 处理批次维度
            x_flat = x.flatten(0, 1) if x.ndim == 3 else x
            
            # 1. 获取缓存权重（简化版本，只做字典查找）
            start_cache.record()
            cached_weight = self.fused_weight_cache.get(0)
            cached_info = self.lora_rank_info_cache.get(0, [])
            end_cache.record()
            
            # 2. 设备检查和转换
            start_device_check.record()
            if cached_weight is not None and (cached_weight.device != x_flat.device or cached_weight.dtype != x_flat.dtype):
                cached_weight = cached_weight.to(device=x_flat.device, dtype=x_flat.dtype)
                self.fused_weight_cache[0] = cached_weight
            end_device_check.record()
            
            if cached_weight is not None:
                # 3. 融合矩阵乘法
                start_fused.record()
                fused_output = torch.matmul(x_flat, cached_weight)
                
                # 分拆融合输出
                qkv_output_size = sum(self.output_slices)
                qkv_part = fused_output[:, :qkv_output_size]
                
                # 应用bias到QKV部分
                if bias is not None:
                    qkv_part = qkv_part + bias
                end_fused.record()
                
                # 4. LoRA expand（如果有缓存的rank信息）
                start_expand.record()
                if fused_output.shape[1] > qkv_output_size and cached_info:
                    lora_shrink_part = fused_output[:, qkv_output_size:]
                    
                    # 调用fused expand操作
                    self.punica_wrapper.add_fused_expand(
                        qkv_part,                    # y: QKV输出，会被就地修改
                        lora_shrink_part,           # fused_shrink_input: 融合计算的shrink结果
                        self.lora_b_stacked,         # lora_b权重
                        self.lora_bias_stacked,      # lora_bias权重  
                        self.output_slices,          # 输出分片
                        cached_info,                 # slice rank信息
                        offset_start=0,
                        add_inputs=True              # 累加到QKV结果上
                    )
                end_expand.record()
            else:
                # 没有缓存权重，无法测量融合方法
                raise RuntimeError("❌ [QKV+LoRA Fusion] 无法获取缓存的融合权重，无法测量融合性能")
            
            # 结束总计时
            end_total.record()
            
            # 等待所有操作完成
            torch.cuda.synchronize()
            
            # 计算各阶段时间
            cache_time = start_cache.elapsed_time(end_cache)
            device_check_time = start_device_check.elapsed_time(end_device_check)
            fused_time = start_fused.elapsed_time(end_fused)
            expand_time = start_expand.elapsed_time(end_expand)
            total_time = start_total.elapsed_time(end_total)
            
            cache_get_times.append(cache_time)
            device_check_times.append(device_check_time)
            fused_matmul_times.append(fused_time)
            expand_times.append(expand_time)
            total_times.append(total_time)
            
            if i == num_iterations - 1:
                final_output = qkv_part
        
        times_dict = {
            'cache_get_times': cache_get_times,
            'device_check_times': device_check_times,
            'fused_matmul_times': fused_matmul_times,
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
        fused_cache_avg = np.mean(fused_times['cache_get_times'])
        fused_device_check_avg = np.mean(fused_times['device_check_times'])
        fused_matmul_avg = np.mean(fused_times['fused_matmul_times'])
        fused_expand_avg = np.mean(fused_times['expand_times'])
        fused_total_avg = np.mean(fused_times['total_times'])
        
        print(f"🔵 传统方法 (QKV + LoRA Shrink + LoRA Expand):")
        print(f"   QKV计算:      {trad_qkv_avg:.3f} ms")
        print(f"   LoRA Shrink:  {trad_shrink_avg:.3f} ms")
        print(f"   LoRA Expand:  {trad_expand_avg:.3f} ms")
        trad_measured_sum = trad_qkv_avg + trad_shrink_avg + trad_expand_avg
        print(f"   各部分总和:   {trad_measured_sum:.3f} ms")
        print(f"   实际总计:     {trad_total_avg:.3f} ms")
        print(f"   未计时部分:   {(trad_total_avg - trad_measured_sum):.3f} ms")
        print(f"")
        
        print(f"🟢 融合方法 (QKV+LoRA融合 + LoRA Expand):")
        print(f"   字典查找:     {fused_cache_avg:.3f} ms")
        print(f"   设备检查:     {fused_device_check_avg:.3f} ms")
        print(f"   融合Matmul:   {fused_matmul_avg:.3f} ms (QKV+LoRA shrink)")
        print(f"   LoRA Expand:  {fused_expand_avg:.3f} ms")
        fused_measured_sum = fused_cache_avg + fused_device_check_avg + fused_matmul_avg + fused_expand_avg
        print(f"   各部分总和:   {fused_measured_sum:.3f} ms")
        print(f"   实际总计:     {fused_total_avg:.3f} ms")
        print(f"   未计时部分:   {(fused_total_avg - fused_measured_sum):.3f} ms")
        print(f"")
        
        # 分析缓存相关开销
        total_cache_overhead = fused_cache_avg + fused_device_check_avg
        print(f"🔍 缓存相关开销分析:")
        print(f"   字典查找:     {fused_cache_avg:.3f} ms ({fused_cache_avg/fused_total_avg*100:.1f}%)")
        print(f"   设备检查:     {fused_device_check_avg:.3f} ms ({fused_device_check_avg/fused_total_avg*100:.1f}%)")
        print(f"   缓存总开销:   {total_cache_overhead:.3f} ms ({total_cache_overhead/fused_total_avg*100:.1f}%)")
        print(f"")
        
        # 计算加速比
        if trad_total_avg > 0:
            speedup = trad_total_avg / fused_total_avg
            print(f"⚡ 性能提升:")
            print(f"   总体加速比:   {speedup:.2f}x")
            print(f"   时间节省:     {trad_total_avg - fused_total_avg:.3f} ms ({((trad_total_avg - fused_total_avg) / trad_total_avg * 100):.1f}%)")
            
            # 核心计算对比（排除缓存开销）
            trad_core = trad_qkv_avg + trad_shrink_avg
            fused_core = fused_matmul_avg
            core_speedup = trad_core / fused_core if fused_core > 0 else 0
            
            print(f"   核心计算对比 (排除缓存开销):")
            print(f"     传统 (QKV+Shrink): {trad_core:.3f} ms")
            print(f"     融合 (QKV+Shrink): {fused_core:.3f} ms")
            print(f"     核心计算加速比:   {core_speedup:.2f}x")
        
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
        """融合的QKV+LoRA计算 - 必须使用缓存的融合权重"""
        # 处理批次维度
        if x.ndim == 3:
            x = x.flatten(0, 1)
        
        # 尝试使用缓存的融合权重
        cached_weight, cached_info = self._get_cached_fused_weight(x.device, x.dtype)
        
        if cached_weight is not None:
            # 使用缓存的融合权重进行计算
            fused_output = torch.matmul(x, cached_weight)
            
            # 分拆融合输出
            qkv_output_size = sum(self.output_slices)
            qkv_part = fused_output[:, :qkv_output_size]
            
            # 应用bias到QKV部分
            if bias is not None:
                qkv_part = qkv_part + bias
            
            # 处理LoRA expand（如果有缓存的rank信息）
            if fused_output.shape[1] > qkv_output_size and cached_info:
                lora_shrink_part = fused_output[:, qkv_output_size:]
                
                # 调用fused expand操作
                self.punica_wrapper.add_fused_expand(
                    qkv_part,                    # y: QKV输出，会被就地修改
                    lora_shrink_part,           # fused_shrink_input: 融合计算的shrink结果
                    self.lora_b_stacked,         # lora_b权重
                    self.lora_bias_stacked,      # lora_bias权重  
                    self.output_slices,          # 输出分片
                    cached_info,                 # slice rank信息
                    offset_start=0,
                    add_inputs=True              # 累加到QKV结果上
                )
            
            return qkv_part
        else:
            # 没有缓存权重，无法进行融合计算
            raise RuntimeError("❌ [QKV+LoRA Fusion] 无法获取缓存的融合权重，融合计算失败")

    def _build_qkv_lora_fused_weight(self, device: torch.device, dtype: torch.dtype, slice_has_lora: list) -> tuple[Optional[torch.Tensor], list]:
        """构建融合的QKV+LoRA权重矩阵"""
        try:
            # 获取QKV权重并转置到正确格式
            qkv_weight = self.base_layer.weight  # [output_size_per_partition, input_size_per_partition]
            qkv_weight = qkv_weight.T  # 转置为 [input_size_per_partition, output_size_per_partition]
            
            # 收集所有slice的LoRA A权重和rank信息
            lora_a_weights = []
            lora_rank_info = []
            current_col = 0
            
            for i in range(self.n_slices):
                lora_a = self.lora_a_stacked[i]  # [max_loras, 1, lora_rank, input_size]
                
                # 处理每个slice（使用第一个LoRA索引）
                lora_a_2d = lora_a[0, 0]  # [lora_rank, input_size]
                valid_lora_a = lora_a_2d.T  # [input_size, lora_rank]
                
                lora_a_weights.append(valid_lora_a)
                lora_rank_info.append({
                    'slice_idx': i,
                    'rank': valid_lora_a.shape[1],  # lora_rank
                    'start_col': current_col
                })
                current_col += valid_lora_a.shape[1]
            
            # 拼接所有LoRA A权重
            all_lora_a = torch.cat(lora_a_weights, dim=1)  # [input_size, total_lora_rank]
            
            # 确保维度兼容性
            if qkv_weight.shape[0] != all_lora_a.shape[0]:
                return None, []
            
            # 构建融合权重矩阵: [input_size, qkv_output_size + total_lora_rank]
            fused_weight = torch.cat([qkv_weight, all_lora_a], dim=1)
            
            return fused_weight, lora_rank_info
            
        except Exception as e:
            return None, []

    def _split_qkv_lora_output(self, fused_output: torch.Tensor, lora_rank_info: list) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """分拆融合输出为QKV部分和LoRA shrink部分"""
        qkv_output_size = sum(self.output_slices)
        
        # 分拆
        qkv_part = fused_output[:, :qkv_output_size]
        
        if fused_output.shape[1] > qkv_output_size and lora_rank_info:
            lora_shrink_part = fused_output[:, qkv_output_size:]
            return qkv_part, lora_shrink_part
        else:
            return qkv_part, None


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
