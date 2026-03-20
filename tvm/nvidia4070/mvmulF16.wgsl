//----------------------------------------
// Function: main_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> A : array<f16>;
@group(0) @binding(1) var<storage, read_write> C : array<f16>;
@group(0) @binding(2) var<storage, read> b : array<f16>;

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
  C_rf_local[0i] = 0.000000e+00h;
  C_rf_local[0i] = fma(A[((v__1 * 1024i) + i32(threadIdx.x))], b[i32(threadIdx.x)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 64i)], b[(i32(threadIdx.x) + 64i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 128i)], b[(i32(threadIdx.x) + 128i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 192i)], b[(i32(threadIdx.x) + 192i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 256i)], b[(i32(threadIdx.x) + 256i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 320i)], b[(i32(threadIdx.x) + 320i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 384i)], b[(i32(threadIdx.x) + 384i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 448i)], b[(i32(threadIdx.x) + 448i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 512i)], b[(i32(threadIdx.x) + 512i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 576i)], b[(i32(threadIdx.x) + 576i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 640i)], b[(i32(threadIdx.x) + 640i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 704i)], b[(i32(threadIdx.x) + 704i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 768i)], b[(i32(threadIdx.x) + 768i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 832i)], b[(i32(threadIdx.x) + 832i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 896i)], b[(i32(threadIdx.x) + 896i)], C_rf_local[0i]);
  C_rf_local[0i] = fma(A[(((v__1 * 1024i) + i32(threadIdx.x)) + 960i)], b[(i32(threadIdx.x) + 960i)], C_rf_local[0i]);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = C_rf_local[0i];
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

