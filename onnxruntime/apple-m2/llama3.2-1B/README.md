# How We Generate MatMul and MatVecMul WGSL Shaders From ONNX Runtime

ONNX Runtime generates WGSL shader code at runtime when it runs a model on the WebGPU backend. It doesn't have a "dump shader" API. But when you set the log level to VERBOSE, it prints the full shader source code to stderr between marker lines. We capture that output and extract the shaders. Different matrix sizes can trigger different kernels, so we create a separate ONNX model and run inference for each (M, N, K) size to capture the exact shader ONNX Runtime generates for that size.

We target the MatMul sizes that appear in Llama 3.2 1B running in fp16. The model has a hidden dimension of 2048 and an intermediate (FFN) dimension of 8192. The M dimension depends on the input prompt token size. We chose 32 and 64 as sizes commonly seen in chat prompts, and 4096 to represent a large prompt.

We also generate shaders for matrix-vector multiply (M=1), which is what happens when the model generates one token at a time. With M=1, ONNX Runtime picks a different tiling strategy that processes one row at a time instead of four.

ONNX Runtime only uses its subgroup optimized MatMul kernel for Intel GPUs. All other GPUs, including Apple, AMD, and NVIDIA, get the same packed kernel. The check is in `onnxruntime/core/providers/webgpu/vendor/intel/math/gemm_subgroup.cc` at line 87:

```cpp
bool CanApplySubgroup(const ComputeContext& context, int64_t M, int64_t N, int64_t K, bool transA, bool transB) {
  if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
    bool use_subgroup = context.HasFeature(wgpu::FeatureName::Subgroups) &&
                        M >= 64 && N >= 512 && K >= 32 && !transA && !transB;
    return use_subgroup;
  }

  return false;
}
```

If the vendor is not "intel", it returns false and the packed kernel is used instead.

## The steps

### Step 1: For each (M, N, K) size, create a tiny ONNX model with just one MatMul node

```python
A = make_tensor_value_info('A', FLOAT16, [M, K])
B = make_tensor_value_info('B', FLOAT16, [K, N])
C = make_tensor_value_info('C', FLOAT16, [M, N])
node = make_node('MatMul', ['A', 'B'], ['C'])
model = make_model(make_graph([node], 'matmul', [A, B], [C]))
```

This is the smallest possible model that triggers a MatMul kernel. The data type (FLOAT16) determines whether we get an fp16 or fp32 shader. The shapes must match the exact sizes we want, because the kernel selection logic depends on them.

### Step 2: Trigger shader compilation by running the model

```python
os.environ["ORT_LOG_LEVEL"] = "VERBOSE"
ort.set_default_logger_severity(0)
sess = ort.InferenceSession(model_path, providers=["WebGpuExecutionProvider"])
outputs = sess.run(None, {
    "A": np.random.randn(M, K).astype(np.float16),
    "B": np.random.randn(K, N).astype(np.float16),
})
```

We call `sess.run()` to force ONNX Runtime to compile the MatMul shader for these dimensions. With verbose logging enabled, ONNX Runtime logs the full WGSL shader source before sending it to the GPU.

### Step 3: Extract the shader from the log output

ONNX Runtime prints shaders between these markers:
```
=== WebGPU Shader code [MatMul] Start ===
... WGSL code here ...
=== WebGPU Shader code [MatMul] End ===
```

We regex match these blocks and strip the timestamp prefixes from each line. We save one `.wgsl` file per (M, N, K) size.
