# First vLLM Offline-Inference Run

## Objective

Run one prompt through the public vLLM `LLM` interface and prove that the request reaches real GPU model execution and returns structured generated output.

## Environment

```text
Operating system: Ubuntu 22.04
GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
GPU memory: 8188 MiB
Compute capability: 8.9
vLLM revision: 05781e21dd4af5ed042d4cc19e833a3ee333e92f
vLLM version: 0.23.1rc1.dev1362+g05781e21d
Model: facebook/opt-125m
Model architecture: OPTForCausalLM
Execution device: CUDA
Model runner: V2 Model Runner
Attention backend: TRITON_ATTN
```

## Command

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python experiments/vllm/offline_inference_smoke.py
```

`VLLM_USE_FLASHINFER_SAMPLER=0` disabled the FlashInfer top-k/top-p sampler for this run. Sampling still completed through another supported path.

## Input

```text
The future of artificial intelligence is
```

The tokenizer produced seven prompt-token IDs:

```text
[2, 133, 499, 9, 7350, 2316, 16]
```

## Configuration

```text
dtype: float32
maximum model length: 128
GPU memory utilization target: 0.70
eager execution: enabled
maximum generated tokens: 16
temperature: 0.0
```

Eager mode disabled `torch.compile` and CUDA Graph execution for this learning-oriented run.

## Broad Execution Path

```text
Python program
    ↓
vllm.LLM
    ↓
model and tokenizer configuration
    ↓
request preparation
    ↓
EngineCore
    ↓
scheduler
    ↓
V2 Model Runner
    ↓
OPT transformer execution on CUDA
    ↓
TRITON_ATTN
    ↓
sampling and output processing
    ↓
RequestOutput
```

This is intentionally a broad trace. Internal scheduler, KV-cache, model-runner, and attention branches are outside the current trace boundary.

## Model Initialization Evidence

vLLM resolved:

```text
OPTForCausalLM
```

The model weights required approximately:

```text
0.47 GiB
```

The engine initialized a single-GPU process:

```text
world size: 1
tensor-parallel rank: 0
pipeline-parallel rank: 0
data-parallel rank: 0
```

## KV-Cache Evidence

The engine reported:

```text
available KV-cache memory: 4.45 GiB
GPU KV-cache capacity: 64,752 tokens
maximum reported concurrency at 128 tokens: 505.88x
```

These values are recorded as runtime observations only. Their allocation algorithm and capacity math have not yet been independently validated.

## Generated Output

```text
 in the hands of the people.

The future of artificial intelligence is in
```

Generated token IDs:

```text
[11, 5, 1420, 9, 5, 82, 4, 50118, 50118, 133, 499, 9, 7350, 2316, 16, 11]
```

Result state:

```text
generated token count: 16
finish reason: length
request finished: True
```

`finish_reason=length` means the request stopped after reaching the configured 16-token output limit.

## Performance Interpretation

The first request triggered Triton JIT compilation for:

```text
kernel_unified_attention
reduce_segments
```

Therefore, the observed timing includes cold-start compilation and must not be represented as steady-state latency or throughput.

No performance claim is made from this run.

## Warning: Optional `deep_gemm` Import

An optional `deep_gemm` component could not identify a local CUDA Toolkit through `CUDA_HOME`.

This did not block the selected runtime path:

```text
attention backend: TRITON_ATTN
request completed: True
```

No CUDA Toolkit installation was performed solely to suppress this warning.

## Warning: Shutdown Message

After the request completed successfully, cleanup emitted:

```text
engine core exited unexpectedly; starting cleanup
```

Current classification:

```text
onboarding-friction observation
```

It is not yet classified as a confirmed defect.

Required follow-up before considering a contribution:

1. reproduce it with a minimal command;
2. confirm the process exit code;
3. determine whether the warning is expected;
4. search current issues and tests;
5. identify the shutdown-state owner;
6. define a deterministic assertion.

## Success Criteria

The smoke test succeeded because:

* the model configuration resolved;
* weights loaded onto the GPU;
* EngineCore initialized;
* KV-cache capacity was allocated;
* a prompt was tokenized;
* GPU inference executed;
* 16 output tokens were produced;
* a finished `RequestOutput` was returned.

## Remaining Questions

1. Which `LLM` method converts the public prompt into an engine request?
2. Which object owns the request after it enters EngineCore?
3. Where is scheduled work handed to the V2 Model Runner?
4. Which model-runner symbol invokes the model?
5. Why does normal script termination emit the unexpected-exit warning?
6. Is the optional `deep_gemm` warning expected without a local CUDA Toolkit?

## Trace Stop Boundary

The current trace stops at:

```text
public LLM API
→ engine/request boundary
→ V2 Model Runner boundary
→ model execution
```

Attention kernels, detailed scheduling, persistent request rows, KV block management, and output-processing internals are explicitly deferred.
