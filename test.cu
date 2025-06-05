#include <stdio.h>

// 你的 warp_reduce_sum 函数
__device__ inline float warp_reduce_sum(float val) {
    unsigned int activemask = __activemask(); // 获取调用此函数时活跃的线程掩码
    // 打印一下进入函数的线程的掩码，以及每个线程的初始值 (可选，但对调试有帮助)
    // if (threadIdx.x == 0) {
    //     printf("Inside warp_reduce_sum by warp %d, initial activemask: 0x%x\n", blockIdx.x * blockDim.x / 32 + threadIdx.x / 32, activemask);
    // }
    // printf("Thread %d (lane %d) in warp_reduce_sum, initial val: %f, current activemask: 0x%x\n",
    //        blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x % 32, val, activemask);

    for (int offset = warpSize / 2; offset > 0; offset /= 2) { // warpSize 通常是 32
        float partner_val = __shfl_down_sync(activemask, val, offset);
        // 让我们看看从伙伴那里得到了什么
        // printf("Thread %d (lane %d), offset %d: my val_before_add=%f, received_from_partner=%f (source lane %d)\n",
        //        blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x % 32, offset, val, partner_val, (threadIdx.x % 32) + offset);
        
        val += partner_val;
        
        // printf("Thread %d (lane %d), offset %d: my val_after_add=%f\n",
        //        blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x % 32, offset, val);
    }
    // printf("Thread %d (lane %d) exiting warp_reduce_sum, final val: %f\n",
    //        blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x % 32, val);
    return val;
}

__global__ void test_kernel(float* output_vals, int active_thread_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % warpSize; // 当前线程在warp内的ID (0-31)

    float my_val = 0.0f;

    // 只有部分线程会参与计算
    if (lane_id < active_thread_count) {
        my_val = (float)(lane_id + 1); // 给参与的线程赋初值 1, 2, 3, ...
        
        // 打印线程的初始状态
        printf("Kernel: Thread %d (lane %d) IS ACTIVE. Initial val: %f. Calling warp_reduce_sum.\n", tid, lane_id, my_val);
        
        my_val = warp_reduce_sum(my_val);
        
        printf("Kernel: Thread %d (lane %d) IS ACTIVE. Final val after reduce: %f\n", tid, lane_id, my_val);
        
        // 通常只有leader线程（比如lane 0）的结果是我们关心的归约总和
        // 但这里为了演示，我们让所有参与的线程都输出它们计算的最终值
        output_vals[tid] = my_val;
    } else {
        // 这些线程不参与，它们的my_val保持0.0f，它们也不在warp_reduce_sum内的activemask里
        // printf("Kernel: Thread %d (lane %d) IS INACTIVE.\n", tid, lane_id);
        output_vals[tid] = -1.0f; // 标记为未参与
    }
}

int main() {
    int num_threads_per_block = 32; // 一个warp
    int num_blocks = 1;
    int total_threads = num_threads_per_block * num_blocks;

    float* h_output = (float*)malloc(total_threads * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, total_threads * sizeof(float));

    // 测试场景1: 只有 warp 的前10个线程参与归约
    int active_threads_in_warp = 10;
    printf("\n--- Test Case: First %d threads active ---\n", active_threads_in_warp);
    cudaMemset(d_output, 0, total_threads * sizeof(float)); // 清零
    test_kernel<<<num_blocks, num_threads_per_block>>>(d_output, active_threads_in_warp);
    cudaDeviceSynchronize(); // 等待kernel完成
    cudaMemcpy(h_output, d_output, total_threads * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Results for first %d active threads:\n", active_threads_in_warp);
    float expected_sum_if_correct_sparse_reduce = 0.0f;
    for(int i=0; i < active_threads_in_warp; ++i) {
        expected_sum_if_correct_sparse_reduce += (i+1);
    }
    printf("Theoretical sum of 1 to %d is %f\n", active_threads_in_warp, expected_sum_if_correct_sparse_reduce);

    for (int i = 0; i < total_threads; i++) {
        // if (h_output[i] != -1.0f) { // 只打印参与的线程
            printf("Thread %d (lane %d): output_val = %f\n", i, i % 32, h_output[i]);
        // }
    }
    printf("Lane 0's final val (often used as the sum for the active group): %f\n", h_output[0]);


    // 你可以增加更多测试场景，比如让奇数/偶数线程参与，或者active_thread_count = 32 (全员参与)
    // 测试场景2: 全员参与
    active_threads_in_warp = 32;
    printf("\n--- Test Case: All %d threads active ---\n", active_threads_in_warp);
    cudaMemset(d_output, 0, total_threads * sizeof(float));
    test_kernel<<<num_blocks, num_threads_per_block>>>(d_output, active_threads_in_warp);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, total_threads * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Results for all %d active threads:\n", active_threads_in_warp);
    expected_sum_if_correct_sparse_reduce = 0.0f;
    for(int i=0; i < active_threads_in_warp; ++i) {
        expected_sum_if_correct_sparse_reduce += (i+1);
    }
    printf("Theoretical sum of 1 to %d is %f\n", active_threads_in_warp, expected_sum_if_correct_sparse_reduce);
    for (int i = 0; i < total_threads; i++) {
         printf("Thread %d (lane %d): output_val = %f\n", i, i % 32, h_output[i]);
    }
    printf("Lane 0's final val (sum of all 32): %f\n", h_output[0]);


    free(h_output);
    cudaFree(d_output);

    return 0;
}