# ONNX Runtime WebGPU Shaders

## MatMul F32 (Matrix Multiplication)

### NVIDIA (5080 FE)
Uses 8×8 workgroups with 32×32 tiling and vec4 vectorization. Shared memory stores tiles of A and B matrices. Compact loop-based accumulation with standard workgroup barriers.

### Intel (Arc A770) vs NVIDIA
Intel uses fully unrolled loops with explicit `subgroupBroadcast(value, lane)` calls for all 32 lanes. Named accumulators (`acc_0` to `acc_7`) instead of arrays. Results in ~1000+ lines vs ~160 lines for NVIDIA, optimized for Intel's 32-wide SIMD.

### AMD (RX 7900 XT)
Identical to NVIDIA - same shader code, same checksum.

## MVMul F32 (Matrix Vector Multiplication)

### NVIDIA (5080 FE)
Uses scalar `f32` arrays instead of vec4 since output is a vector. 8×8 workgroups with 32×32 shared memory tiles. Loop-based accumulation with nested loops for row/column traversal.

### Intel (Arc A770) vs NVIDIA
Intel uses `vec4<f32>` vectorization even for mvmul, while NVIDIA uses scalar `f32`. Intel processes 4 elements at a time with vectorized loads/stores. Slightly fewer lines (~154 vs ~164) due to vec4 packing.

### AMD (RX 7900 XT)
Identical to NVIDIA - same shader logic, same scalar `f32` approach.
