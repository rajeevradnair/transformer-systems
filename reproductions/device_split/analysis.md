# CPU/GPU Device-Split Failure

## Summary

An intermediate hidden-state tensor was moved to the GPU while the attention projection parameters remained on the CPU.

The tensor shapes and dtypes were valid, but the operation failed because matrix multiplication requires the input and model parameters to reside on the same device.

## Expected Behavior

All tensors participating in an operation should reside on the same execution device.

For the attention projection:

```text
hidden_states.device == projection.weight.device
```

A correctly placed GPU execution path should look like:

```text
token IDs:              cuda:0
embedding parameters:   cuda:0
hidden states:          cuda:0
attention parameters:   cuda:0
```

Alternatively, a valid CPU path should keep every component on the CPU.

## Observed Behavior

The reproduction created the following placement:

```text
token IDs:              cpu
embedding parameters:   cpu
embedding output:       cpu
hidden states after move: cuda:0
attention parameters:   cpu
```

The attention projection attempted to execute using:

```text
input:   cuda:0
weight:  cpu
```

PyTorch raised a device-placement error during the linear projection.

The complete raw output is preserved in:

```text
reproductions/device_split/raw_output.txt
```

## Trigger

The trigger was moving only the intermediate hidden-state tensor:

```python
hidden_states = hidden_states.to("cuda")
```

while leaving the attention module unchanged:

```python
attention = AttentionPlaceholder(hidden_width=8)
```

Modules are created on the CPU unless explicitly moved.

Therefore:

```text
hidden_states.device = cuda:0
attention.projection.weight.device = cpu
```

## Deterministic Failing Operation

The failure can be reproduced with:

```python
embedding = TokenEmbedding(
    vocab_size=20,
    hidden_width=8,
)

attention = AttentionPlaceholder(
    hidden_width=8,
)

token_ids = torch.tensor(
    [[1, 4, 7, 3]],
    dtype=torch.long,
    device="cpu",
)

hidden_states = embedding(token_ids)
hidden_states = hidden_states.to("cuda")

attention(hidden_states)
```

The operation deterministically fails on a CUDA-capable system because the projection input and weight occupy different devices.

## Behavioral Contract

The attention placeholder accepts:

```text
hidden states: [batch, sequence, hidden_width]
```

It additionally requires:

```text
hidden_states.device == attention_parameter_device
```

The input in this reproduction satisfies the shape contract:

```text
[1, 4, 8]
```

It violates only the device-placement contract.

## First Causal Error

The first causal error occurs when the execution graph is split by moving only the intermediate tensor:

```python
hidden_states = hidden_states.to("cuda")
```

The later matrix-multiplication exception is the resulting symptom.

The first causal error is not:

* the linear operation itself;
* the hidden-state shape;
* the hidden width;
* the token values;
* the tensor dtype.

## State Owner

The immediate state owner is:

```text
AttentionPlaceholder.forward
```

This boundary owns:

* the incoming hidden states;
* the projection parameters;
* the transition into the attention projection.

The broader execution-device owner is normally the parent model or runtime responsible for placing the complete model and its inputs.

For example:

```python
model = model.to("cuda")
token_ids = token_ids.to("cuda")
```

Moving the parent module recursively moves its registered child-module parameters.

## Root-Cause Hypothesis

The root cause is incomplete model-device placement.

Only one intermediate activation was moved to CUDA, while the module parameters required by the next operation remained on the CPU.

Potential real-world causes include:

* moving input tensors without moving the model;
* moving only part of a composed model;
* constructing a new module after the rest of the model was moved;
* loading a checkpoint onto an unexpected device;
* manually moving an intermediate activation;
* inconsistent device configuration across pipeline stages.

For this reproduction, the confirmed cause is the deliberate movement of only `hidden_states`.

## Confirming Evidence

The hypothesis is confirmed by:

1. The hidden-state shape is valid:

   ```text
   [1, 4, 8]
   ```

2. The hidden-state dtype is valid:

   ```text
   torch.float32
   ```

3. The attention input width matches the configured hidden width:

   ```text
   observed width = 8
   expected width = 8
   ```

4. The hidden states reside on:

   ```text
   cuda:0
   ```

5. The projection weight resides on:

   ```text
   cpu
   ```

6. Moving the attention module to CUDA allows the projection to execute:

   ```python
   attention = attention.to("cuda")
   ```

7. Keeping both the input and projection on the CPU also allows execution.

## Rejecting Evidence

### Token bounds failure

All token IDs are within the vocabulary:

```text
1, 4, 7, 3
```

For `vocab_size = 20`, each value is valid.

### Hidden-width mismatch

The attention layer expects width `8`, and the input width is `8`.

### Batch or sequence shape failure

The hidden-state shape follows the required contract:

```text
[batch, sequence, hidden_width] = [1, 4, 8]
```

### Dtype mismatch

The hidden states and projection parameters both use `torch.float32`.

### Invalid parameter values

The failure occurs before parameter values affect the numerical result. The issue is their location, not their contents.

## Smallest Safe Fix

At the local attention boundary, check the parameter and input devices before executing the projection:

```python
parameter_device = self.projection.weight.device

if hidden_states.device != parameter_device:
    raise ValueError(
        "attention device mismatch: "
        f"hidden_states={hidden_states.device}, "
        f"projection_weight={parameter_device}"
    )
```

This does not automatically move either operand.

Automatic movement inside `forward()` would be unsafe because it could:

* hide incorrect runtime configuration;
* introduce repeated CPU-to-GPU transfers;
* cause synchronization and performance regressions;
* move data to an unintended GPU;
* interfere with distributed or pipeline-parallel placement.

The safest response is to fail early and require the caller to correct the placement.

## Correct Usage

Move the complete model before execution:

```python
device = torch.device("cuda")

model = TinyLanguageModel(
    vocab_size=20,
    hidden_width=8,
).to(device)

token_ids = token_ids.to(device)

logits = model(token_ids)
```

This preserves a consistent execution graph:

```text
input → embedding → attention → MLP → logits
cuda     cuda        cuda        cuda    cuda
```

## Validation

The focused regression test is:

```text
tests/test_tiny_block.py::test_attention_rejects_cpu_weight_and_cuda_input_split
```

Run it with:

```bash
pytest -q \
  tests/test_tiny_block.py::test_attention_rejects_cpu_weight_and_cuda_input_split
```

Expected result:

```text
1 passed
```

Run the complete toy test suite with:

```bash
pytest -q tests/test_tiny_block.py
```

Expected result at this checkpoint:

```text
4 passed
```

## Before-and-After Behavior

### Before explicit validation

```text
CUDA hidden states
        +
CPU projection parameters
        ↓
low-level PyTorch device error
```

### After explicit validation

```text
CUDA hidden states
        +
CPU projection parameters
        ↓
AttentionPlaceholder device-contract check
        ↓
descriptive ValueError
```

## Remaining Uncertainty

This reproduction proves the local device-placement invariant but does not cover:

* multiple-GPU placement;
* tensor parallelism;
* pipeline parallelism;
* CPU offloading;
* lazy parameter initialization;
* model sharding;
* distributed process ownership;
* asynchronous device transfers.

A production runtime may intentionally place different modules on different devices, but every individual operation must still receive operands on compatible devices.

## Reproduction Artifacts

```text
reproductions/device_split/
├── analysis.md
├── raw_output.txt
└── reproduce_raw.py
```
