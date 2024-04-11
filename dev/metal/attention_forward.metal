/*
Kernels for attention forward pass on Apple Silicon using NEON.

Compile example:
xcrun -sdk macosx metal -c attention_forward.metal -o attention_forward.air
xcrun -sdk macosx metallib attention_forward.air -o attention_forward.metallib
*/

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------------
// Metal utils

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void attention_forward_cpu(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Metal kernels

kernel void attention_query_key_kernel1(device float* preatt, device float* inp,
                                           int B, int T, int C, int NH,
                                           uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        int t2 = idx % T;
        int t = (idx / T) % T;
        if (t2 > t) {
            // autoregressive mask
            preatt[idx] = -INFINITY;
            return;
        }
        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int C3 = C*3;
        int hs = C / NH; // head size
        device float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        device float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float4 val = 0.0f;
        for (int i = 0; i < hs; i += 4) {
            float4 q = float4(query_t[i], query_t[i+1], query_t[i+2], query_t[i+3]);
            float4 k = float4(key_t2[i], key_t2[i+1], key_t2[i+2], key_t2[i+3]);
            val += q * k;
        }
        val.x += val.y + val.z + val.w;
        val.x *= 1.0 / sqrt(hs);

        preatt[idx] = val.x;
    }
}

kernel void attention_softmax_kernel1(device float* att, device float* preatt,
                                         int B, int T, int NH,
                                         uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        device float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
        device float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        // find maxval
        float maxval = -10000.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            maxval = max(maxval, preatt_bth[t2]);
        }

        // calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = exp(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                att_bth[t2] *= expsum_inv;
            } else {
                // causal attention mask. not strictly necessary to set to zero here
                // only doing this explicitly for debugging and checking to PyTorch
                att_bth[t2] = 0.0f;
            }
        }
    }
}

// SIMD-level reduction for finding the maximum value
inline float simd_reduce_max(float4 val) {
    val = max(val, shuffle(val, 1));
    val = max(val, shuffle(val, 2));
    return max(val.x, val.y);
}

// SIMD-level reduction for summing values
inline float simd_reduce_sum(float4 val) {
    val += shuffle(val, 1);
    val += shuffle(val, 2);
    return val.x + val.y;
}

kernel void softmax_forward_kernel4(device float* out, device float* inp, int N, int C,
                                    uint2 tid [[thread_position_in_grid]],
                                    uint tgid [[threadgroup_position_in_grid]],
                                    uint gid [[thread_index_in_threadgroup]]) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 4)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in SIMDs of 4 threads

    // special reduction operations simd_reduce_max/simd_reduce_sum are used for intra-SIMD reductions
    // threadgroup memory is used for inter-SIMD reduction
    threadgroup float shared[64];
    int idx = tgid;
    int simdId = gid / 4; // SIMD index within a threadgroup
    int laneId = gid % 4; // thread index within a SIMD

    // the number of SIMDs per threadgroup.
    int simdsPerGroup = C / 4;

    // shared[] must be allocated to have 2 * simdsPerGroup elements
    // first half for max values, the second half for sum values
    threadgroup float* maxvals = shared;
    threadgroup float* sumvals = &shared[simdsPerGroup];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    device float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = gid; i < C; i += C) {
        maxval = max(maxval, x[i]);
    }
    // now within-SIMD reductions for maxval
    float4 maxval4 = maxval;
    maxval = simd_reduce_max(maxval4);

    // the 0th thread of each SIMD writes the maxval of that SIMD to threadgroup memory
    if (laneId == 0) maxvals[simdId] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // now the 0th thread reduces the maxvals in threadgroup memory, i.e. across SIMDs
    if (gid == 0) {
        float val = maxvals[0];
        for (int i = 1; i < simdsPerGroup; i++) {
            val = max(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute exp and write the result to global memory
    for (int i = gid; i < C; i += C) {
        // subtract max for numerical stability
        out[idx * C + i] = exp(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = gid; i < C; i += C) {
        sumval += x[i];
    }
    // within-SIMD reduction for sumval
    float4 sumval4 = sumval;
    sumval = simd_reduce_sum(sumval4);

    // write sumval to threadgroup memory
    if (laneId == 0) sumvals[simdId] = sumval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // inter-thread reduction of sum
    if (gid == 0) {
        float val = sumvals[0];
        for (int i = 1; i < simdsPerGroup; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = gid; i < C; i += C) {
        out[idx * C + i] /= sum;
    }
}

kernel void attention_value_kernel1(device float* out, device float* att, device float* inp,
                                      int B, int T, int C, int NH,
                                      uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    int total_threads = B * NH * T * C;

    if (idx < total_threads) {
        int c = idx % C;
        int t = (idx / C) % T;
        int h = (idx / (C * T)) % NH;
        int b = idx / (NH * T * C);

        int C3 = C*3;
        int hs = C / NH; // head size
        device float* out_bthc = out + b * T * C + t * C + h * hs + c;
        device float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        // accumulate weighted values into the output of attention
        *out_bthc = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            device float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2 + c; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            *out_bthc += att_btht2 * *value_t2;
        }
    }
}

kernel void permute_kernel(device float* q, device float* k, device float* v, device float* inp,
                           int B, int N, int NH, int d,
                           uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    int total_threads = B * NH * N * d;
    
    if (idx < total_threads) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) 
                    + (n * 3 * NH * d)
                    + (0 * NH * d) 
                    + (nh_ * d)
                    + d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}
kernel void unpermute_kernel(device float* inp, device float* out,
                             constant int& B, constant int& N, constant int& NH, constant int& d,
                             uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

kernel void scale_kernel(device float* preatt, float scale, 
                         int B, int NH, int T,
                         uint2 tid [[thread_position_in_grid]]) {
    int idx = tid.x;
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        preatt[idx] *= scale;
    }
}

// kernel version dispatch
kernel void attention_forward3(device float* out, device float* vaccum, device float* qkvr, device float* preatt, device float* att,
                       device float* inp,
                       int B, int T, int C, int NH,
                       uint2 tid [[thread_position_in_grid]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gid [[thread_index_in_threadgroup]]) {
    // inp is (B, T, 3C) QKV 
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    permute_kernel(q, k, v, inp, B, T, NH, HS, tid);

    // batched matrix multiply with MPSMatrixMultiplication
    MPSMatrixMultiplication *mpsMatMul = [[MPSMatrixMultiplication alloc] initWithDevice:MTLCreateSystemDefaultDevice()];
    MPSMatrixDescriptor *qDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:HS rowBytes:T*HS dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *kDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:HS columns:T rowBytes:T*HS dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *preattDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:T rowBytes:T*T dataType:MPSDataTypeFloat32];

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            MPSMatrix *qMat = [[MPSMatrix alloc] initWithBuffer:q + b*NH*T*HS + h*T*HS descriptor:qDesc];
            MPSMatrix *kMat = [[MPSMatrix alloc] initWithBuffer:k + b*NH*T*HS + h*T*HS descriptor:kDesc];
            MPSMatrix *preattMat = [[MPSMatrix alloc] initWithBuffer:preatt + b*NH*T*T + h*T*T descriptor:preattDesc];

            [mpsMatMul encodeToCommandBuffer:MTLCreateSystemDefaultDevice().newCommandQueue.commandBuffer 
                                    inputMatrix:qMat
                                    inputMatrix:kMat
                                   resultMatrix:preattMat];
        }
    }

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    total_threads = B * NH * T * T;
    scale_kernel(preatt, scale, B, NH, T, tid);

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    total_threads = B * NH * T;
    softmax_forward_kernel4(att, preatt, B * NH * T, T, tid, tgid, gid);

    // batched matrix multiply with MPSMatrixMultiplication
    MPSMatrixDescriptor *attDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:T rowBytes:T*T dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *vDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:HS rowBytes:T*HS dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *outDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:HS rowBytes:T*HS dataType:MPSDataTypeFloat32];

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            MPSMatrix *attMat = [[MPSMatrix alloc] initWithBuffer:att + b*NH*T*T + h*T*T descriptor:attDesc];
            MPSMatrix *vMat = [[MPSMatrix alloc] initWithBuffer:v + b*NH*T*HS + h*T*HS descriptor:vDesc];
            MPSMatrix *outMat = [[MPSMatrix alloc] initWithBuffer:out + b*T*C + h*T*HS descriptor:outDesc];

            [mpsMatMul encodeToCommandBuffer:MTLCreateSystemDefaultDevice().newCommandQueue.commandBuffer
                                    inputMatrix:attMat
                                    inputMatrix:vMat
                                   resultMatrix:outMat];
        }
    }

    // unpermute output
    total_threads = B * T * C;
    unpermute_kernel(out, vaccum, B, T, NH, HS, tid);
    
    // copy from vaccum to out
    for (int i = tid.x; i < B * T * C; i += total_threads) {
        out[i] = vaccum[i];
    }
}

kernel void attention_forward(int kernel_num,
                       device float* out, device float* vaccum, device float* qkvr, device float* preatt, device float* att,
                       device float* inp,
                       int B, int T, int C, int NH,
                       uint2 tid [[thread_position_in_grid]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gid [[thread_index_in_threadgroup]]) {
    switch (kernel_num) {
        case 1:
            // not implemented
            break;
        case 3:
            attention_forward3(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, tid, tgid, gid);
            break;
        default:
            // not implemented
            break;
    }
}