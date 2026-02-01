enable subgroups;
const workgroup_size_x: u32 = 8;
const workgroup_size_y: u32 = 8;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
struct Uniforms {
  a_shape: vec2<u32>,
  a_stride: u32,
  b_shape: vec2<u32>,
  b_stride: u32,
  output_shape: vec3<u32>,
  output_stride: vec2<u32>,
  dim_a_outer: u32,
  dim_b_outer: u32,
  dim_inner: u32,
  logical_dispatch_x: u32,
  logical_dispatch_y: u32,
  logical_dispatch_z: u32
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

alias a_value_t = f32;
alias a_indices_t = vec2<u32>;
alias a_element_t = f32;
fn i2o_a(indices : a_indices_t)->u32 {
  return indices[0] * uniforms.a_stride + indices[1];
}
fn get_a_by_indices(indices_fnarg: a_indices_t)->a_value_t {
  return a[i2o_a(indices_fnarg)];
}
alias b_value_t = f32;
alias b_indices_t = vec2<u32>;
fn i2o_b(indices : b_indices_t)->u32 {
  return indices[0] * uniforms.b_stride + indices[1];
}
fn get_b_by_indices(indices_fnarg: b_indices_t)->b_value_t {
  return b[i2o_b(indices_fnarg)];
}
alias output_value_t = f32;
alias output_indices_t = vec3<u32>;
alias output_element_t = f32;
fn i2o_output(indices : output_indices_t)->u32 {
  return indices[0] * uniforms.output_stride[0] + indices[1] * uniforms.output_stride[1] + indices[2];
}
fn set_output_by_indices(indices: output_indices_t, value: output_value_t) {
  output[i2o_output(indices)]=value;
}
alias batch_dims_indices_t = u32;

fn mm_readA(batch: i32, row: i32, colIn: i32 , batch_indices: batch_dims_indices_t) -> output_element_t {
     var value = output_element_t(0);
    let col = colIn * 1;
    if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_inner)) {
        var a_indices: a_indices_t;

a_indices[0]=u32(row);
a_indices[1]=u32(colIn);
        value = get_a_by_indices(a_indices);
    }
    return value;
}

fn mm_readB(batch: i32, row: i32, colIn: i32 , batch_indices: batch_dims_indices_t) -> output_element_t {
     var value = output_element_t(0);
    let col = colIn * 1;
    if(row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_b_outer)) {
        var b_indices: b_indices_t;

b_indices[0]=u32(row);
b_indices[1]=u32(colIn);
        value = get_b_by_indices(b_indices);
    }
    return value;
}

fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: output_value_t) {
  let col = colIn * 1;
  if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_b_outer)) {
    var value = valueIn;
    let coords = vec3(u32(batch), u32(row), u32(colIn));
    
    set_output_by_indices(coords, value);
  }
}

var<workgroup> mm_Asub: array<array<a_element_t, 32>, 32>;
var<workgroup> mm_Bsub: array<array<a_element_t, 32>, 32>;
const rowPerThread = 4;
const colPerThread = 4;
const tileInner = 32;
@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32,
        @builtin(num_workgroups) num_workgroups : vec3<u32>) {
  let workgroup_idx = workgroup_id.z * num_workgroups[0] * num_workgroups[1] + workgroup_id.y * num_workgroups[0] + workgroup_id.x;
  let global_idx = workgroup_idx * (workgroup_size_x * workgroup_size_y * workgroup_size_z) + local_idx;
  let logical_workgroup_id_z = workgroup_idx / (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y);
  let logical_workgroup_id_y = (workgroup_idx % (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y)) / uniforms.logical_dispatch_x;
  let logical_workgroup_id_x = (workgroup_idx % (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y)) % uniforms.logical_dispatch_x;
  let logical_workgroup_id = vec3u(logical_workgroup_id_x, logical_workgroup_id_y, logical_workgroup_id_z);
  const workgroupSize = vec3u(workgroup_size_x, workgroup_size_y, workgroup_size_z);
  let logical_global_id = logical_workgroup_id * workgroupSize + local_id;
 let batch = i32(logical_global_id.z);
  let batchIndices = u32(batch);
 let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;
 var kStart = 0;
 var acc: array<array<a_element_t, colPerThread>, rowPerThread>;
let tileRow = i32(local_id.y) * rowPerThread;
let tileCol = i32(local_id.x) * colPerThread;
let globalRow = i32(logical_global_id.y) * rowPerThread;
let globalCol = i32(logical_global_id.x) * colPerThread;
let globalRowStart = i32(logical_workgroup_id.y) * 32;
let globalColStart = i32(logical_workgroup_id.x) * 32;
let tileRowA = i32(local_id.y) * 4;
let tileColA = i32(local_id.x) * 4;
let tileRowB = i32(local_id.y) * 4;
for (var t = 0; t < i32(num_tiles); t = t + 1) {
  for (var innerRow = 0; innerRow < i32(4); innerRow = innerRow + 1) {
    for (var innerCol = 0; innerCol < i32(4); innerCol = innerCol + 1) {
      let inputRow = tileRowA + innerRow;
      let inputCol = tileColA + innerCol;
      mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRowStart + inputRow, kStart + inputCol, batchIndices);

    }
  }
  for (var innerRow = 0; innerRow < i32(4); innerRow = innerRow + 1) {
    for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {
      let inputRow = tileRowB + innerRow;
      let inputCol = tileCol + innerCol;
           mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalColStart + inputCol, batchIndices);

     }
  }
  kStart = kStart + tileInner;
  workgroupBarrier();
var BCached: array<a_element_t, colPerThread>;
  for (var k = 0; k < tileInner; k = k + 1) {
    for (var inner = 0; inner < i32(colPerThread); inner = inner + 1) {
      BCached[inner] = mm_Bsub[k][tileCol + inner];
    }
    for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {
      let ACached = mm_Asub[tileRow + innerRow][k];
      for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {
        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
      }
    }
  }
  workgroupBarrier();
}
for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {
  for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {
    mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);
  }
}

}
