// MatMul: M=4096, K=2048, N=2048, dtype=f16
// Dispatch: (64, 128, 1)
// Workgroup: (8, 8, 1)
// Tile: A_outer=32, B_outer=32, inner=32
// Elements/thread: (4, 4, 1)
// Vec4 path, innerElementSize=4

enable f16;

const workgroup_size_x: u32 = 8u;
const workgroup_size_y: u32 = 8u;
const workgroup_size_z: u32 = 1u;

struct Uniforms {
  dim_a_outer: u32,  // M = 4096
  dim_b_outer: u32,  // N = 2048
  dim_inner: u32,    // K = 2048
  logical_dispatch_x: u32,
  logical_dispatch_y: u32,
  logical_dispatch_z: u32,
  // Shapes and strides for a, b, output
  a_shape: vec3<u32>,    // (1, 4096, 512)
  a_strides: vec3<u32>,
  b_shape: vec3<u32>,    // (1, 2048, 512)
  b_strides: vec3<u32>,
  output_shape: vec3<u32>,  // (1, 4096, 512)
  output_strides: vec3<u32>,
  batch_dims_shape: u32,
};

@group(0) @binding(0) var<storage, read> a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

alias a_element_t = f16;
alias a_value_t = vec4<f16>;
alias b_value_t = vec4<f16>;
alias output_element_t = f16;
alias output_value_t = vec4<f16>;
alias a_indices_t = vec3<u32>;
alias b_indices_t = vec3<u32>;
alias output_indices_t = vec3<u32>;
alias batch_dims_indices_t = u32;

fn i2o_a(indices: vec3<u32>) -> u32 {
  return indices[0] * uniforms.a_strides[0] + indices[1] * uniforms.a_strides[1] + indices[2];
}

fn i2o_b(indices: vec3<u32>) -> u32 {
  return indices[0] * uniforms.b_strides[0] + indices[1] * uniforms.b_strides[1] + indices[2];
}

fn i2o_output(indices: vec3<u32>) -> u32 {
  return indices[0] * uniforms.output_strides[0] + indices[1] * uniforms.output_strides[1] + indices[2];
}

fn mm_readA(batch: i32, row: i32, colIn: i32, batch_indices: batch_dims_indices_t) -> vec4<f16> {
  var value = vec4<f16>(0);
  let col = colIn * 4;
  if (row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_inner)) {
    var a_indices: a_indices_t;
    a_indices[0] = u32(batch);
    a_indices[1] = u32(row);
    a_indices[2] = u32(colIn);
    value = a[i2o_a(a_indices)];
  }
  return value;
}

fn mm_readB(batch: i32, row: i32, colIn: i32, batch_indices: batch_dims_indices_t) -> vec4<f16> {
  var value = vec4<f16>(0);
  let col = colIn * 4;
  if (row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_b_outer)) {
    var b_indices: b_indices_t;
    b_indices[0] = u32(batch);
    b_indices[1] = u32(row);
    b_indices[2] = u32(colIn);
    value = b[i2o_b(b_indices)];
  }
  return value;
}

fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: output_value_t) {
  let col = colIn * 4;
  if (row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_b_outer)) {
    var value = valueIn;
    let coords = vec3<u32>(u32(batch), u32(row), u32(colIn));
    output[i2o_output(coords)] = value;
  }
}

var<workgroup> mm_Asub: array<array<vec4<f16>, 8>, 32>;
var<workgroup> mm_Bsub: array<array<vec4<f16>, 8>, 32>;

const rowPerThread = 4;
const colPerThread = 4;
const innerElementSize = 4;
const tileInner = 32;

@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let workgroup_idx = workgroup_id.z * num_workgroups.x * num_workgroups.y + workgroup_id.y * num_workgroups.x + workgroup_id.x;
  let logical_workgroup_id_z = workgroup_idx / (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y);
  let logical_workgroup_id_y = (workgroup_idx % (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y)) / uniforms.logical_dispatch_x;
  let logical_workgroup_id_x = (workgroup_idx % (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y)) % uniforms.logical_dispatch_x;
  let logical_workgroup_id = vec3u(logical_workgroup_id_x, logical_workgroup_id_y, logical_workgroup_id_z);
  const workgroupSize = vec3u(workgroup_size_x, workgroup_size_y, workgroup_size_z);
  let logical_global_id = logical_workgroup_id * workgroupSize + local_id;

  let localRow = i32(local_id.y);
  let tileRow = localRow * rowPerThread;
  let tileCol = i32(local_id.x);
  let globalRow = i32(logical_global_id.y) * rowPerThread;
  let globalCol = i32(logical_global_id.x);
  let globalRowStart = i32(logical_workgroup_id.y) * 32;
  let globalColStart = i32(logical_workgroup_id.x) * 32;
  var acc: array<vec4<f16>, rowPerThread>;

  let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;
  var kStart = 0;
  let batch = i32(logical_global_id.z);
  let batchIndices = u32(batch);
  let tileRowB = localRow * 4;

  for (var t = 0; t < i32(num_tiles); t = t + 1) {
    // Load tile of A into shared memory
    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      let inputRow = tileRow + innerRow;
      let inputCol = tileCol;
      mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRow + innerRow, kStart / innerElementSize + inputCol, batchIndices);
    }

    // Load tile of B into shared memory
    for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) {
      let inputRow = tileRowB + innerRow;
      let inputCol = tileCol;
      mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol, batchIndices);
    }
    kStart = kStart + tileInner;
    workgroupBarrier();

    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {
      let BCached0 = mm_Bsub[k * innerElementSize][tileCol];
      let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];
      let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];
      let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];
      for (var i = 0; i < rowPerThread; i = i + 1) {
        let ACached = mm_Asub[tileRow + i][k];
        acc[i] = BCached0 * ACached.x + acc[i];
        acc[i] = BCached1 * ACached.y + acc[i];
        acc[i] = BCached2 * ACached.z + acc[i];
        acc[i] = BCached3 * ACached.w + acc[i];
      }
    }
    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
    mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
  }
}
