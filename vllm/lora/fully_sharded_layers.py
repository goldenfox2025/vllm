# SPDX-License-Identifier: Apache-2.0

# pylint: disable=unused-argument
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig, PretrainedConfig
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLoRA,
                              QKVParallelLinearWithLoRA,
                              RowParallelLinearWithLoRA)
from vllm.platforms import current_platform

import os
import warnings

if TYPE_CHECKING:
    pass


def _fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        return (can_replace(*args, **kwargs)
                and kwargs["lora_config"].fully_sharded_loras)

    return dec


def _mcp_apply(x, bias, layer: ColumnParallelLinearWithLoRA):
    """ 
    For `ColumnParallelLinearWithLoRA` or classes that inherit from 
    `ColumnParallelLinearWithLoRA`, they share the same `apply` logic.
    """
    assert (layer.n_slices == len(layer.lora_a_stacked) == len(
        layer.lora_b_stacked) == len(layer.output_slices))
    if layer.lora_bias_stacked is not None:
        assert layer.n_slices == len(layer.lora_bias_stacked)

    output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)

    x = x.view(-1, x.shape[-1])
    output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape

    # Since communication is needed, the buffer is directly initialized as a
    # tensor rather than a tuple of tensor.
    buffers = torch.zeros(
        (layer.n_slices, x.shape[0], layer.lora_a_stacked[0].shape[2]),
        dtype=torch.float32,
        device=x.device,
    )

    shrunk_buffers: Optional[torch.Tensor] = layer.punica_wrapper.add_shrink(
        buffers, x, layer.lora_a_stacked, 1.0)

    if not current_platform.can_update_inplace():
        buffers = shrunk_buffers

    buffers = tensor_model_parallel_all_gather(buffers)

    lora_output: Optional[torch.Tensor] = layer.punica_wrapper.add_expand(
        output,
        buffers,
        layer.lora_b_stacked,
        layer.lora_bias_stacked,
        layer.output_slices,
        offset_start=0,
        add_input=True)

    if not current_platform.can_update_inplace():
        output = lora_output

    output = output.view(*out_orig_shape)
    # now have column partitioned and packed output
    return output


def _mcp_apply_fused(x, bias, layer: ColumnParallelLinearWithLoRA):
    """
    融合版本的apply实现 - 将QKV计算和LoRA shrink融合在一起
    模仿QKV融合的方式，减少kernel启动次数和提高内存带宽利用率
    """
    # 环境变量控制是否启用融合优化
    enable_fusion = os.environ.get("VLLM_ENABLE_QKV_LORA_FUSION", "0") == "1"
    
    if not enable_fusion:
        return _mcp_apply(x, bias, layer)
    
    assert (layer.n_slices == len(layer.lora_a_stacked) == len(
        layer.lora_b_stacked) == len(layer.output_slices))
    if layer.lora_bias_stacked is not None:
        assert layer.n_slices == len(layer.lora_bias_stacked)

    # Step 1: 检查是否有有效的LoRA权重
    has_valid_lora = any(
        layer.lora_a_stacked[i].abs().sum() > 0 
        for i in range(layer.n_slices)
    )
    
    if not has_valid_lora:
        # 没有有效LoRA，直接进行基础计算
        output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)
        return output

    # Step 2: 构建融合权重矩阵 [input_size, qkv_size + lora_sizes]
    try:
        fused_weight = _build_qkv_lora_fused_weight(layer, x.device)
        if fused_weight is None:
            # 融合失败，回退到原始实现
            return _mcp_apply(x, bias, layer)
        
        # Step 3: 执行融合的matmul计算
        x_flat = x.view(-1, x.shape[-1])
        fused_output = _compute_qkv_lora_fused(x_flat, fused_weight, bias, layer)
        
        # Step 4: 分拆结果
        qkv_output, lora_shrink_output = _split_qkv_lora_output(fused_output, layer)
        
        # Step 5: 应用expand操作
        qkv_output, out_orig_shape = qkv_output.view(-1, qkv_output.shape[-1]), qkv_output.shape
        
        # 如需要，进行all_gather
        if hasattr(layer, 'tp_size') and layer.tp_size > 1:
            lora_shrink_output = tensor_model_parallel_all_gather(lora_shrink_output)
        
        lora_output: Optional[torch.Tensor] = layer.punica_wrapper.add_expand(
            qkv_output,
            lora_shrink_output,
            layer.lora_b_stacked,
            layer.lora_bias_stacked,
            layer.output_slices,
            offset_start=0,
            add_input=True)

        if not current_platform.can_update_inplace():
            qkv_output = lora_output

        return qkv_output.view(*out_orig_shape)
        
    except Exception as e:
        # 融合计算失败，回退到原始实现
        warnings.warn(f"QKV+LoRA fusion failed: {e}, falling back to original implementation")
        return _mcp_apply(x, bias, layer)


def _build_qkv_lora_fused_weight(layer: ColumnParallelLinearWithLoRA, device) -> Optional[torch.Tensor]:
    """
    构建融合的QKV+LoRA权重矩阵
    模仿QKV权重拼接的方式，将LoRA A权重也拼接进去
    
    结果形状: [input_size, qkv_output_size + total_lora_rank]
    """
    try:
        # 获取基础QKV权重
        base_weight = layer.base_layer.weight  # [qkv_output_size, input_size]
        input_size = base_weight.shape[1]
        qkv_output_size = base_weight.shape[0]
        
        # 获取LoRA参数
        lora_rank = layer.lora_a_stacked[0].shape[2]
        max_loras = layer.lora_a_stacked[0].shape[0] 
        n_slices = layer.n_slices
        
        # 计算总的LoRA输出大小
        total_lora_rank = n_slices * max_loras * lora_rank
        
        # 创建融合权重矩阵 [input_size, qkv_output_size + total_lora_rank]
        fused_weight = torch.zeros(
            input_size, qkv_output_size + total_lora_rank,
            dtype=base_weight.dtype,
            device=device
        )
        
        # 填充QKV权重部分 - 转置后放入 
        fused_weight[:, :qkv_output_size] = base_weight.T
        
        # 填充LoRA A权重部分
        lora_offset = qkv_output_size
        for slice_idx in range(n_slices):
            for lora_idx in range(max_loras):
                # 获取LoRA A权重 [lora_rank, input_size]
                lora_a = layer.lora_a_stacked[slice_idx][lora_idx, 0]  
                
                # 转置并放入融合矩阵 [input_size, lora_rank]  
                fused_weight[:, lora_offset:lora_offset + lora_rank] = lora_a.T
                lora_offset += lora_rank
        
        return fused_weight
        
    except Exception as e:
        return None


def _compute_qkv_lora_fused(x: torch.Tensor, fused_weight: torch.Tensor, 
                           bias: Optional[torch.Tensor], layer) -> torch.Tensor:
    """
    执行融合的QKV+LoRA计算
    
    Args:
        x: 输入 [num_tokens, input_size]  
        fused_weight: 融合权重 [input_size, qkv_output_size + total_lora_rank]
        bias: 偏置（如果有）
        layer: LoRA层
    
    Returns:
        fused_output: [num_tokens, qkv_output_size + total_lora_rank]
    """
    # 执行大的matmul
    fused_output = torch.mm(x, fused_weight)
    
    # 如果有bias，只应用到QKV部分
    if bias is not None:
        qkv_output_size = layer.base_layer.weight.shape[0]
        fused_output[:, :qkv_output_size] += bias
    
    return fused_output


def _split_qkv_lora_output(fused_output: torch.Tensor, layer) -> tuple[torch.Tensor, torch.Tensor]:
    """
    分拆融合输出为QKV部分和LoRA shrink部分
    
    Args:
        fused_output: [num_tokens, qkv_output_size + total_lora_rank]
        layer: LoRA层
    
    Returns:
        qkv_output: [num_tokens, qkv_output_size]
        lora_shrink_output: [n_slices, num_tokens, lora_rank] 
    """
    num_tokens = fused_output.shape[0]
    qkv_output_size = layer.base_layer.weight.shape[0]
    
    # 分拆QKV和LoRA部分
    qkv_output = fused_output[:, :qkv_output_size]
    lora_part = fused_output[:, qkv_output_size:]
    
    # 重塑LoRA部分为shrink格式
    lora_rank = layer.lora_a_stacked[0].shape[2] 
    max_loras = layer.lora_a_stacked[0].shape[0]
    n_slices = layer.n_slices
    
    # 重塑为 [num_tokens, n_slices, max_loras, lora_rank]
    lora_reshaped = lora_part.view(num_tokens, n_slices, max_loras, lora_rank)
    
    # 创建shrink输出格式 [n_slices, num_tokens, lora_rank]
    # 🔧 修复：保持数据类型一致，使用与输入相同的dtype
    lora_shrink_output = torch.zeros(
        n_slices, num_tokens, lora_rank,
        dtype=fused_output.dtype,  # 使用输入数据的dtype而不是float32
        device=fused_output.device
    )
    
    # 🔧 修复：避免不必要的数据类型转换
    for slice_idx in range(n_slices):
        lora_shrink_output[slice_idx] = lora_reshaped[:, slice_idx, 0, :]  # 不转换为float
    
    return qkv_output, lora_shrink_output


# these layers are based on the tensor parallelism strategy given in
# Y. Sheng et al., S-LoRA: Serving Thousands of Concurrent LoRA Adapters. 2023,
# https://arxiv.org/abs/2311.03285.


class ColumnParallelLinearWithShardedLoRA(ColumnParallelLinearWithLoRA):
    """
    Differs from ColumnParallelLinearWithLoRA by slicing LoRA A also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    # For all LoRA layers where the `base_layer` is `ColumnParallelLinear`,
    # their `lora_a` and `lora_b` have different sharding patterns. After
    # completing the `lora_a` GEMM , a gather operation is performed.
    # Therefore, the sharding of `lora_a` only needs to correspond with the
    # gather operation.
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.lora_a_stacked[0].shape[2]
        start_idx = tp_rank * shard_size
        lora_a = lora_a[:, start_idx:start_idx + shard_size]
        return lora_a

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class MergedColumnParallelLinearWithShardedLoRA(
        MergedColumnParallelLinearWithLoRA):
    """
    Differs from MergedColumnParallelLinearWithLoRA by slicing the
    LoRA A's also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(
        self, lora_a: list[Union[torch.Tensor, None]]
    ) -> list[Union[torch.Tensor, None]]:
        #NOTE: lora_a contains 2 subloras, and each sublora could be None.
        output_shard_size = self.lora_a_stacked[0].shape[2]
        output_start_idx = self.tp_rank * output_shard_size
        lora_a = [
            lora_a[0][:, output_start_idx:output_start_idx +
                      output_shard_size] if lora_a[0] is not None else None,
            lora_a[1][:, output_start_idx:output_start_idx +
                      output_shard_size] if lora_a[1] is not None else None,
        ]
        return lora_a

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 支持融合优化（主要用于gate_up_proj等）
        enable_fusion = os.environ.get("VLLM_ENABLE_QKV_LORA_FUSION", "0") == "1"
        
        if enable_fusion:
            if not hasattr(self, '_fusion_logged'):
                print(f"🚀 [Fusion] Enabled for {self.__class__.__name__}")
                self._fusion_logged = True
            return _mcp_apply_fused(x, bias, self)
        else:
            return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class QKVParallelLinearWithShardedLoRA(QKVParallelLinearWithLoRA):
    """
    Differs from QKVParallelLinearWithLoRA by slicing the
    LoRA A's also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.lora_a_stacked[0].shape[2]
        start_idx = tp_rank * shard_size
        lora_a = lora_a[:, start_idx:start_idx + shard_size]
        return lora_a

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: list,
                          model_config: Optional[PretrainedConfig]) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class MergedQKVParallelLinearWithShardedLoRA(MergedQKVParallelLinearWithLoRA):
    """
    Differs from MergedQKVParallelLinearWithLoRA by slicing the 
    LoRA A's also.

    Based on S-LoRA, slicing happens along the rank dim.
    
    现在支持QKV+LoRA融合优化！
    """

    def slice_lora_a(
        self, lora_a: list[Union[torch.Tensor, None]]
    ) -> list[Union[torch.Tensor, None]]:
        # NOTE: lora_a contains 3 subloras, and each sublora could be None.
        shard_size = [self.lora_a_stacked[i].shape[2] for i in range(3)]
        start_idx = [self.tp_rank * shard_size[i] for i in range(3)]
        lora_a = [
            lora_a[0][:, start_idx[0]:start_idx[0] +
                      shard_size[0]] if lora_a[0] is not None else None,
            lora_a[1][:, start_idx[1]:start_idx[1] +
                      shard_size[1]] if lora_a[1] is not None else None,
            lora_a[2][:, start_idx[2]:start_idx[2] +
                      shard_size[2]] if lora_a[2] is not None else None,
        ]
        return lora_a

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 🚀 使用融合优化版本！
        enable_fusion = os.environ.get("VLLM_ENABLE_QKV_LORA_FUSION", "0") == "1"
        
        if enable_fusion:
            # 记录融合使用情况
            if not hasattr(self, '_fusion_logged'):
                print(f"🚀 [QKV+LoRA Fusion] Enabled for {self.__class__.__name__}")
                self._fusion_logged = True
            return _mcp_apply_fused(x, bias, self)
        else:
            return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class FusedMergedQKVParallelLinearWithShardedLoRA(MergedQKVParallelLinearWithShardedLoRA):
    """
    专门为QKV+LoRA融合优化设计的版本
    
    这个类默认启用融合优化，无需环境变量控制
    """
    
    def __init__(self, base_layer):
        super().__init__(base_layer)
        self._force_fusion = True
        print(f"🚀 [Fused QKV+LoRA] Initialized {self.__class__.__name__} with forced fusion")
    
    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 强制使用融合优化，无论环境变量如何设置
        return _mcp_apply_fused(x, bias, self)
    
    def set_fusion_mode(self, enable: bool):
        """动态控制融合模式，便于性能对比"""
        self._force_fusion = enable
        if enable:
            print(f"🔧 [Fused QKV+LoRA] Enabled fusion for {self.__class__.__name__}")
        else:
            print(f"🔧 [Fused QKV+LoRA] Disabled fusion for {self.__class__.__name__}")
    
    def apply_with_mode_control(self,
                               x: torch.Tensor,
                               bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """支持动态模式切换的apply方法"""
        if self._force_fusion:
            return _mcp_apply_fused(x, bias, self)
        else:
            return _mcp_apply(x, bias, self)


class RowParallelLinearWithShardedLoRA(RowParallelLinearWithLoRA):
    """
    Differs from RowParallelLinearWithLoRA by slicing the
    LoRA B's also.

    Based on S-LoRA, slicing happens along the output dim.
    This yields a combined partial sum from the row parallel base
    layer and column partitioned output from the LoRA.
    """

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_b_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = lora_b[:, start_idx:end_idx]
        return lora_b

    def slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        if bias is None:
            return bias
        self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                      self.lora_bias_stacked)
        shard_size = self.lora_bias_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        bias = bias[start_idx:end_idx]
        return bias

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x)

        x = x.view(-1, x.shape[-1])
        output, out_orig_shape = output.view(-1,
                                             output.shape[-1]), output.shape
        buffer = torch.zeros(
            (self.n_slices, x.shape[0], self.lora_a_stacked[0].shape[2]),
            dtype=torch.float32,
            device=x.device,
        )

        shrunk_buffer: Optional[torch.Tensor] = self.punica_wrapper.add_shrink(
            buffer, x, self.lora_a_stacked, 1.0)
        if not current_platform.can_update_inplace():
            buffer = shrunk_buffer

        buffer = tensor_model_parallel_all_reduce(buffer)

        # following S-LoRA, allows the fusing of all_gather and all_reduce
        # by adding the column partitioned lora output to a slice of output
        # tensor, which is a partial sum due to row parallel. All that
        # remains is a standard all_reduce. User should be aware though that
        # the output is not the same as a normal row_parallel, it should be
        # reduced before being used
        # NOTE offset are based on the rank.
        shard_size = self.lora_b_stacked[0].shape[2]
        offset_start = self.tp_rank * shard_size
        lora_output: Optional[torch.Tensor] = self.punica_wrapper.add_expand(
            output,
            buffer,
            self.lora_b_stacked,
            self.lora_bias_stacked,
            self.output_slices,
            offset_start=offset_start,
            add_input=True,
        )

        if not current_platform.can_update_inplace():
            output = lora_output

        output = output.view(*out_orig_shape)
        return output

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )
