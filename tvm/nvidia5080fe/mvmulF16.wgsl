//----------------------------------------
// Function: main_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> A : array<f16>;
@group(0) @binding(1) var<storage, read> B : array<f16>;
@group(0) @binding(2) var<storage, read_write> C : array<f16>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f16, 64>;
@compute @workgroup_size(64, 1, 1)
fn main_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var C_rf_local : array<f16, 1>;
  var B_local : array<f16, 8>;
  var C_rf_local_1 : array<f16, 1>;
  C_rf_local[0i] = 0.000000e+00h;
  B_local[0i] = B[(i32(threadIdx.x) * 8i)];
  B_local[1i] = B[((i32(threadIdx.x) * 8i) + 1i)];
  B_local[2i] = B[((i32(threadIdx.x) * 8i) + 2i)];
  B_local[3i] = B[((i32(threadIdx.x) * 8i) + 3i)];
  B_local[4i] = B[((i32(threadIdx.x) * 8i) + 4i)];
  B_local[5i] = B[((i32(threadIdx.x) * 8i) + 5i)];
  B_local[6i] = B[((i32(threadIdx.x) * 8i) + 6i)];
  B_local[7i] = B[((i32(threadIdx.x) * 8i) + 7i)];
  C_rf_local[0i] = fma(A[((v__1 * 1024i) + (i32(threadIdx.x) * 8i))], B_local[0i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 1i)], B_local[1i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 2i)], B_local[2i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 3i)], B_local[3i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 4i)], B_local[4i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 5i)], B_local[5i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 6i)], B_local[6i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 7i)], B_local[7i], C_rf_local[0i]);
  B_local[0i] = B[((i32(threadIdx.x) * 8i) + 512i)];
  B_local[1i] = B[((i32(threadIdx.x) * 8i) + 513i)];
  B_local[2i] = B[((i32(threadIdx.x) * 8i) + 514i)];
  B_local[3i] = B[((i32(threadIdx.x) * 8i) + 515i)];
  B_local[4i] = B[((i32(threadIdx.x) * 8i) + 516i)];
  B_local[5i] = B[((i32(threadIdx.x) * 8i) + 517i)];
  B_local[6i] = B[((i32(threadIdx.x) * 8i) + 518i)];
  B_local[7i] = B[((i32(threadIdx.x) * 8i) + 519i)];
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 512i)], B_local[0i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 513i)], B_local[1i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 514i)], B_local[2i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 515i)], B_local[3i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 516i)], B_local[4i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 517i)], B_local[5i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 518i)], B_local[6i], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + (i32(threadIdx.x) * 8i)) + 519i)], B_local[7i], C_rf_local[0i]);
  C_rf_local_1[0i] = 0.000000e+00h;
  C_rf_local_1[0i] = (C_rf_local_1[0i] + C_rf_local[0i]);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = C_rf_local_1[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 32i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    C[v__1] = red_buf0[0i];
  }
}

