# SPDX-License-Identifier: Apache-2.0
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import TYPE_CHECKING, Optional, Union, final

import torch
import sys
import os

import vllm.envs as envs
from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (LoRAKernelMeta, lora_expand,
                                          lora_shrink)

# Try to import CUDA LoRA kernels
try:
    from .cuda_punica.ctypes_wrapper import cuda_lora_shrink_triton_interface, C_LIB_AVAILABLE
    from .cuda_punica.expand_ctypes_wrapper import cuda_lora_expand_triton_interface
    CUDA_LORA_AVAILABLE = C_LIB_AVAILABLE
    if CUDA_LORA_AVAILABLE:
        print("✅ CUDA LoRA kernels (shrink + expand) available")
except ImportError:
    CUDA_LORA_AVAILABLE = False
from .punica_base import PunicaWrapperBase

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.models import LongContextLoRAContext


def _compare_tensors_and_exit_if_different(triton_result: torch.Tensor,
                                          cuda_result: torch.Tensor,
                                          operation_name: str,
                                          rtol: float = 1e-1,
                                          atol: float = 1e-1) -> None:
    """
    对比 Triton 和 CUDA 的输出结果，如果不一致则退出推理

    Args:
        triton_result: Triton 实现的输出结果
        cuda_result: CUDA 实现的输出结果
        operation_name: 操作名称，用于错误信息
        rtol: 相对容差
        atol: 绝对容差
    """
    if not torch.allclose(triton_result, cuda_result, rtol=rtol, atol=atol):
        # 计算差异统计信息
        diff = torch.abs(triton_result - cuda_result)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        # 找到最大差异的位置
        max_diff_idx = torch.argmax(diff.flatten())
        max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)

        print(f"❌ {operation_name} 结果不一致!")
        print(f"   张量形状: {triton_result.shape}")
        print(f"   最大差异: {max_diff:.8f}")
        print(f"   平均差异: {mean_diff:.8f}")
        print(f"   最大差异位置: {max_diff_pos}")
        print(f"   Triton 结果范围: [{torch.min(triton_result).item():.6f}, {torch.max(triton_result).item():.6f}]")
        print(f"   CUDA 结果范围: [{torch.min(cuda_result).item():.6f}, {torch.max(cuda_result).item():.6f}]")
        print(f"   容差设置: rtol={rtol}, atol={atol}")

        # 展平张量以便查看具体数值
        triton_flat = triton_result.flatten()
        cuda_flat = cuda_result.flatten()
        diff_flat = diff.flatten()

        # 输出前10个数值
        print(f"\n📊 前10个数值对比:")
        for i in range(min(10, len(triton_flat))):
            print(f"   [{i:2d}] Triton: {triton_flat[i].item():10.6f}, CUDA: {cuda_flat[i].item():10.6f}, 差异: {diff_flat[i].item():10.6f}")

        # 找到最大差异周围的数值
        max_idx = max_diff_idx.item()
        start_idx = max(0, max_idx - 5)
        end_idx = min(len(triton_flat), max_idx + 6)

        print(f"\n🎯 最大差异位置 [{max_idx}] 前后10个数值:")
        for i in range(start_idx, end_idx):
            marker = " *** " if i == max_idx else "     "
            print(f"{marker}[{i:2d}] Triton: {triton_flat[i].item():10.6f}, CUDA: {cuda_flat[i].item():10.6f}, 差异: {diff_flat[i].item():10.6f}")

        # 检查是否有 NaN 或 Inf
        triton_nan = torch.isnan(triton_result).sum().item()
        cuda_nan = torch.isnan(cuda_result).sum().item()
        triton_inf = torch.isinf(triton_result).sum().item()
        cuda_inf = torch.isinf(cuda_result).sum().item()

        if triton_nan > 0 or cuda_nan > 0 or triton_inf > 0 or cuda_inf > 0:
            print(f"\n⚠️  异常值检测:")
            print(f"   Triton NaN: {triton_nan}, Inf: {triton_inf}")
            print(f"   CUDA NaN: {cuda_nan}, Inf: {cuda_inf}")

        print("🛑 退出推理以避免错误结果传播")

        # 退出推理
        sys.exit(1)
    else:
        print(f"✅ {operation_name} 结果一致 (Triton vs CUDA)")


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        self.max_loras = kwargs['max_loras']

        self.token_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                      max_num_batched_tokens,
                                                      device=device)

        # When cudagraph capture size is greater than max_num_seqs (max_batches,
        # here), V0 captures the graph as if max_num_seqs is set to
        # the capture size.
        # V1 doesn't have this problem and always respects max_num_seqs.
        max_num_prompts = (max_batches
                           if envs.VLLM_USE_V1 else max_num_batched_tokens)
        self.prompt_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                       max_num_prompts,
                                                       device=device)

    def update_metadata(
            self,
            mapping: LoRAMapping,
            lora_index_to_id: list[Optional[int]],
            max_loras: int,
            vocab_size: int,
            extra_vocab_size: int,
            long_lora_context: Optional["LongContextLoRAContext"] = None,
            **kwargs):

        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size,
                                   long_lora_context)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   lora_a_stacked: tuple[torch.Tensor,
                                         ...], scale: float, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])

        # # 添加详细的调试信息
        # print(f"\n🔍 LoRA Shrink 调试信息:")
        # print(f"   输入 x 形状: {x.shape}")
        # print(f"   输出 y 形状: {y.shape}")
        # print(f"   LoRA A 权重数量: {len(lora_a_stacked)}")
        # print(f"   缩放因子: {scale}")

        # for i, lora_a in enumerate(lora_a_stacked):
        #     print(f"   LoRA A[{i}] 形状: {lora_a.shape}")
        #     print(f"   LoRA A[{i}] 数据类型: {lora_a.dtype}")
        #     print(f"   LoRA A[{i}] 设备: {lora_a.device}")
        #     print(f"   LoRA A[{i}] 数值范围: [{torch.min(lora_a).item():.6f}, {torch.max(lora_a).item():.6f}]")

        # # 检查是否有 LoRA 映射信息
        # if hasattr(self, '_token_lora_indices') and self._token_lora_indices is not None:
        #     print(f"   Token LoRA 索引形状: {self._token_lora_indices.shape}")
        #     print(f"   Token LoRA 索引内容: {self._token_lora_indices[:min(10, len(self._token_lora_indices))].tolist()}")
        # else:
        #     print(f"   ⚠️  Token LoRA 索引未设置")

        # # 检查元数据
        # if hasattr(self, 'token_mapping_meta'):
        #     meta_args = self.token_mapping_meta.meta_args(x.size(0))
        #     print(f"   元数据参数数量: {len(meta_args)}")
        #     for i, arg in enumerate(meta_args):
        #         if torch.is_tensor(arg):
        #             print(f"   元数据[{i}] 形状: {arg.shape}, 内容: {arg[:min(5, len(arg))].tolist()}")
        #         else:
        #             print(f"   元数据[{i}]: {arg}")
        # else:
        #     print(f"   ⚠️  Token mapping meta 未设置")

        # 如果 CUDA 和 Triton 都可用，则先调用 Triton，再调用 CUDA，对比结果
        if CUDA_LORA_AVAILABLE and HAS_TRITON:
            import os
            force_triton = os.environ.get("VLLM_FORCE_TRITON_LORA", "0") == "1"
            
            if force_triton:
                print("🔵 强制使用 Triton LoRA shrink (VLLM_FORCE_TRITON_LORA=1)")
                lora_shrink(
                    x,
                    lora_a_stacked,
                    y,
                    *self.token_mapping_meta.meta_args(x.size(0)),
                    scale,
                )
                return
            
            # 1. 先调用 Triton 实现
            y_triton = y.clone()  # 保存 Triton 结果

            print(f"\n🔵 Triton Shrink 调试信息:")
            print(f"   输入 x 形状: {x.shape}, 数据类型: {x.dtype}")
            print(f"   输出 y_triton 形状: {y_triton.shape}, 数据类型: {y_triton.dtype}")
            print(f"   输入 x 数值范围: [{torch.min(x).item():.6f}, {torch.max(x).item():.6f}]")
            print(f"   输入 y_triton (调用前) 数值范围: [{torch.min(y_triton).item():.6f}, {torch.max(y_triton).item():.6f}]")

            # 获取元数据参数
            triton_meta_args = self.token_mapping_meta.meta_args(x.size(0))
            print(f"   Triton 元数据参数数量: {len(triton_meta_args)}")
            for i, arg in enumerate(triton_meta_args):
                if torch.is_tensor(arg):
                    print(f"   Triton 元数据[{i}] 形状: {arg.shape}, 内容: {arg[:min(5, len(arg))].tolist()}")
                else:
                    print(f"   Triton 元数据[{i}]: {arg}")

            lora_shrink(
                x,
                lora_a_stacked,
                y_triton,
                *triton_meta_args,
                scale,
            )
            print(f"   输出 y_triton (调用后) 数值范围: [{torch.min(y_triton).item():.6f}, {torch.max(y_triton).item():.6f}]")
            print("🔵 Triton LoRA shrink 完成")

            # 2. 再调用 CUDA 实现
            y_cuda = y.clone()  # 保存 CUDA 结果
            cuda_success = self._try_cuda_shrink(y_cuda, x, lora_a_stacked, scale)

            if cuda_success:
                print("🟢 CUDA LoRA shrink 完成")

                # 3. 对比结果
                print(f"\n🔍 详细对比 Triton vs CUDA 结果:")
                print(f"   Triton 输出统计: min={torch.min(y_triton).item():.6f}, max={torch.max(y_triton).item():.6f}, mean={torch.mean(y_triton).item():.6f}")
                print(f"   CUDA 输出统计: min={torch.min(y_cuda).item():.6f}, max={torch.max(y_cuda).item():.6f}, mean={torch.mean(y_cuda).item():.6f}")

                # 先用宽松的容差检查是否有基本的相似性
                loose_match = torch.allclose(y_triton, y_cuda, rtol=1e-1, atol=1e-1)
                print(f"   宽松容差匹配 (rtol=1e-1, atol=1e-1): {loose_match}")

                if not loose_match:
                    print("   ⚠️  即使宽松容差也不匹配，可能存在根本性差异")

                    # 检查是否一个全零一个非零
                    triton_nonzero = torch.count_nonzero(y_triton).item()
                    cuda_nonzero = torch.count_nonzero(y_cuda).item()
                    print(f"   Triton 非零元素: {triton_nonzero}/{y_triton.numel()}")
                    print(f"   CUDA 非零元素: {cuda_nonzero}/{y_cuda.numel()}")

                    # 检查数据类型
                    print(f"   Triton 数据类型: {y_triton.dtype}")
                    print(f"   CUDA 数据类型: {y_cuda.dtype}")

                    # 检查形状
                    print(f"   Triton 形状: {y_triton.shape}")
                    print(f"   CUDA 形状: {y_cuda.shape}")

                    # 分析零值分布
                    print(f"\n🔍 零值分布分析:")
                    for slice_idx in range(y_triton.shape[0]):
                        triton_slice_nonzero = torch.count_nonzero(y_triton[slice_idx]).item()
                        cuda_slice_nonzero = torch.count_nonzero(y_cuda[slice_idx]).item()
                        print(f"   Slice {slice_idx}: Triton 非零={triton_slice_nonzero}, CUDA 非零={cuda_slice_nonzero}")

                    # 检查输入数据是否相同
                    print(f"\n🔍 输入数据检查:")
                    print(f"   输入 x 统计: min={torch.min(x).item():.6f}, max={torch.max(x).item():.6f}, mean={torch.mean(x).item():.6f}")
                    print(f"   输入 x 非零元素: {torch.count_nonzero(x).item()}/{x.numel()}")

                    # 检查 LoRA 权重
                    print(f"\n🔍 LoRA 权重检查:")
                    for i, lora_a in enumerate(lora_a_stacked):
                        print(f"   LoRA A[{i}] 统计: min={torch.min(lora_a).item():.6f}, max={torch.max(lora_a).item():.6f}, mean={torch.mean(lora_a).item():.6f}")
                        print(f"   LoRA A[{i}] 非零元素: {torch.count_nonzero(lora_a).item()}/{lora_a.numel()}")

                    # 暂时使用更宽松的容差继续运行，以便收集更多信息
                    print(f"\n⚠️  检测到 CUDA 实现问题，但继续运行以收集更多调试信息...")


                # 严格检查，发现问题立即退出
                _compare_tensors_and_exit_if_different(
                    y_triton, y_cuda, "LoRA Shrink", rtol=1e-2, atol=1e-2
                )

                # 使用 Triton 结果（作为参考标准）
                y.copy_(y_cuda)
                return
            else:
                print("⚠️  CUDA LoRA shrink 失败，退出推理")
                sys.exit(1)
                # y.copy_(y_triton)
                # return

        # 如果只有 Triton 可用，直接使用 Triton
        elif HAS_TRITON:
            lora_shrink(
                x,
                lora_a_stacked,
                y,
                *self.token_mapping_meta.meta_args(x.size(0)),
                scale,
            )
            print("🔵 仅使用 Triton LoRA shrink")

        # 如果只有 CUDA 可用，直接使用 CUDA
        elif CUDA_LORA_AVAILABLE:
            if self._try_cuda_shrink(y, x, lora_a_stacked, scale):
                print("🟢 仅使用 CUDA LoRA shrink")
            else:
                raise RuntimeError("CUDA LoRA shrink 失败且无 Triton 备选方案")
        else:
            raise RuntimeError("无可用的 LoRA shrink 实现 (Triton 和 CUDA 均不可用)")

    def _try_cuda_shrink(self, y: torch.Tensor, x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...], scale: float) -> bool:
        """
        Try CUDA implementation of LoRA shrink using ctypes wrapper with Triton interface

        Returns:
            bool: True if successful, False if failed (will fallback to Triton)
        """
        try:
            # Use the same metadata as Triton
            num_tokens = x.size(0)

            print(f"\n🟢 CUDA Shrink 调试信息:")
            print(f"   输入 x 形状: {x.shape}, 数据类型: {x.dtype}")
            print(f"   输出 y 形状: {y.shape}, 数据类型: {y.dtype}")
            print(f"   num_tokens: {num_tokens}")
            print(f"   缩放因子: {scale}")

            # 获取元数据参数
            meta_args = self.token_mapping_meta.meta_args(num_tokens)
            print(f"   CUDA 元数据参数数量: {len(meta_args)}")
            for i, arg in enumerate(meta_args):
                if torch.is_tensor(arg):
                    print(f"   CUDA 元数据[{i}] 形状: {arg.shape}, 内容: {arg[:min(5, len(arg))].tolist()}")
                else:
                    print(f"   CUDA 元数据[{i}]: {arg}")

            # 检查输入数据的数值范围
            print(f"   输入 x 数值范围: [{torch.min(x).item():.6f}, {torch.max(x).item():.6f}]")
            print(f"   输入 y (调用前) 数值范围: [{torch.min(y).item():.6f}, {torch.max(y).item():.6f}]")

            # Call CUDA kernel with Triton interface
            success = cuda_lora_shrink_triton_interface(
                x,                                              # inputs
                lora_a_stacked,                                # lora_a_weights (直接传递tuple)
                y,                                             # output_tensor
                *meta_args,                                    # Same metadata as Triton
                scale                                          # scaling
            )

            if success:
                print(f"   输出 y (调用后) 数值范围: [{torch.min(y).item():.6f}, {torch.max(y).item():.6f}]")
            else:
                print(f"   ⚠️  CUDA kernel 返回失败")

            return success

        except Exception as e:
            # Log error but don't crash - fallback to Triton
            print(f"⚠️  CUDA LoRA shrink failed: {e}")
            # import traceback
            # traceback.print_exc()
            return False

    def _try_cuda_expand(self, y: torch.Tensor, x: torch.Tensor,
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        offset_start: int) -> bool:
        """
        尝试使用CUDA expand kernel

        Args:
            y: 输出张量 [num_tokens, hidden_size]
            x: 输入张量 [num_slices, num_tokens, lora_rank]
            lora_b_stacked: LoRA B权重元组
            offset_start: 输出偏移（暂时忽略，简化实现）

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 🚀 现在支持多slice和非零offset_start，用于GQA等场景
            # 移除之前的限制，让CUDA kernel处理更多情况

            # 提取单个slice: [num_tokens, lora_rank]
            x_2d = x[0]  # [num_tokens, lora_rank]
            num_tokens = x_2d.size(0)

            # 调用CUDA expand kernel，使用与Triton完全一致的接口
            success = cuda_lora_expand_triton_interface(
                x,  # [num_slices, num_tokens, lora_rank] 与Triton一致
                list(lora_b_stacked),  # LoRA B权重列表
                y,  # [num_tokens, hidden_size] 输出
                *self.token_mapping_meta.meta_args(num_tokens),  # Punica元数据
                offset_start=offset_start,  # 输出偏移
                add_inputs=True,  # 累加输入
            )

            if success:
                print("🚀 Using CUDA LoRA expand kernel")
                return True
            else:
                return False

        except Exception as e:
            print(f"⚠️  CUDA expand kernel failed: {e}")
            return False

    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   lora_b_stacked: tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                   output_slices: tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] +
                    lora_bias_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]):
                bias's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            self._apply_bias(token_lora_indices, y, output_slices,
                             lora_bias_stacked)

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)  # first dimension is the num slices

     
        if CUDA_LORA_AVAILABLE and HAS_TRITON :
            # 检查是否强制使用Triton
            force_triton = os.environ.get("VLLM_FORCE_TRITON_LORA", "0") == "1"
            
            if force_triton:
                print("🔵 强制使用 Triton LoRA expand (VLLM_FORCE_TRITON_LORA=1)")
                lora_expand(
                    x,
                    lora_b_stacked,
                    y,
                    *self.token_mapping_meta.meta_args(num_tokens),
                    offset_start=offset_start,
                    add_inputs=True,
                )
                y = y.view_as(y_org)
                return
            
            # 1. 先调用 Triton 实现
            y_triton = y.clone()  # 保存 Triton 结果
            lora_expand(
                x,
                lora_b_stacked,
                y_triton,
                *self.token_mapping_meta.meta_args(num_tokens),
                offset_start=offset_start,
                add_inputs=True,
            )
            print("🔵 Triton LoRA expand 完成")

            # 2. 再调用 CUDA 实现
            y_cuda = y.clone()  # 保存 CUDA 结果
            cuda_success = self._try_cuda_expand(y_cuda, x, lora_b_stacked, offset_start)

            if cuda_success:
                print("🟢 CUDA LoRA expand 完成")

                # 3. 对比结果
                print(f"\n🔍 详细对比 Triton vs CUDA Expand 结果:")
                print(f"   Triton 输出统计: min={torch.min(y_triton).item():.6f}, max={torch.max(y_triton).item():.6f}, mean={torch.mean(y_triton).item():.6f}")
                print(f"   CUDA 输出统计: min={torch.min(y_cuda).item():.6f}, max={torch.max(y_cuda).item():.6f}, mean={torch.mean(y_cuda).item():.6f}")

                # 先用宽松的容差检查
                loose_match = torch.allclose(y_triton, y_cuda, rtol=1e-1, atol=1e-1)
                print(f"   宽松容差匹配 (rtol=1e-1, atol=1e-1): {loose_match}")

                _compare_tensors_and_exit_if_different(
                    y_triton, y_cuda, "LoRA Expand", rtol=1e-1, atol=1e-1
                )

           
                y.copy_(y_cuda)
            else:
                print("⚠️  CUDA LoRA expand 失败")
                y.copy_(y_triton)

        # 如果只有 Triton 可用，直接使用 Triton
        elif HAS_TRITON:
            lora_expand(
                x,
                lora_b_stacked,
                y,
                *self.token_mapping_meta.meta_args(num_tokens),
                offset_start=offset_start,
                add_inputs=True,
            )
            print("🔵 仅使用 Triton LoRA expand")

        # 如果只有 CUDA 可用，直接使用 CUDA
        elif CUDA_LORA_AVAILABLE:
            if self._try_cuda_expand(y, x, lora_b_stacked, offset_start):
                print("🟢 仅使用 CUDA LoRA expand")
            else:
                raise RuntimeError("CUDA LoRA expand 失败且无 Triton 备选方案")
        else:
            raise RuntimeError("无可用的 LoRA expand 实现 (Triton 和 CUDA 均不可用)")

        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked, ),
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...],
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: tuple[int, ...],
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            y = self._apply_bias(token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros(  # type: ignore
                (len(output_slices), x.size(0), r),
                dtype=torch.float32,
                device=x.device,
            )
        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs)
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            None,
            output_slices,
            add_inputs=True,
            **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)


        if HAS_TRITON:
            lora_shrink(x, [lora_a_stacked], buffer.unsqueeze(dim=0),
                        *self.prompt_mapping_meta.meta_args(x.size(0)), scale)

            lora_expand(buffer.unsqueeze(dim=0), [lora_b_stacked],
                        y,
                        *self.prompt_mapping_meta.meta_args(buffer.size(0)),
                        add_inputs=True)
            print("🔵 使用 Triton LoRA logits 处理")
        else:
            raise RuntimeError("LoRA logits 处理需要 Triton 支持")

        y = y.view_as(y_org)
