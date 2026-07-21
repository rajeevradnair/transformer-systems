# Tiny Transformer Execution from First Principles

## Objective

Implement and validate one transformer-style execution path:

```text
token IDs
    ↓
token embeddings
    ↓
attention placeholder
    ↓
residual connection
    ↓
MLP
    ↓
residual connection
    ↓
vocabulary logits
```

The implementation is located at:

```text
src/transformer/tiny_block.py
```

The attention component intentionally preserves the transformer shape contract without implementing Q, K, V, masking, or real token mixing. Those attention internals are outside this implementation’s current scope.

## Configuration Used

```text
batch size:       2
sequence length:  4
vocabulary size:  20
hidden width:     8
MLP expansion:    4
intermediate width: 32
```

## Tensor Shape Trace

### Token IDs

```text
shape:  [2, 4]
dtype:  torch.int64
meaning: two sequences containing four token IDs each
```

Token IDs are integer indices into the embedding table.

### Embedding Table

```text
weight shape: [20, 8]
```

The table contains:

* 20 vocabulary rows;
* one eight-dimensional vector per vocabulary item.

### Embedding Output

```text
input:  [2, 4]
output: [2, 4, 8]
```

Each integer token ID becomes an eight-dimensional hidden-state vector.

### Attention Placeholder

```text
input:  [2, 4, 8]
output: [2, 4, 8]
```

The placeholder applies a learned linear projection to the final dimension.

It preserves batch size, sequence length, and hidden width so its result can participate in a residual connection.

### First Residual Connection

```text
embedding hidden states: [2, 4, 8]
attention output:         [2, 4, 8]
result:                   [2, 4, 8]
```

Conceptually:

```text
hidden_states_1 =
    hidden_states_0 + attention(hidden_states_0)
```

The shape, dtype, and device must match on both sides of the addition.

### MLP Expansion

```text
input:  [2, 4, 8]
output: [2, 4, 32]
```

The MLP expands each token representation independently:

```text
hidden width 8 → intermediate width 32
```

Attention is responsible for token-to-token interaction in a real transformer. The MLP instead transforms the feature vector at each token position.

### MLP Contraction

```text
input:  [2, 4, 32]
output: [2, 4, 8]
```

The output returns to the original hidden width so it can participate in the second residual connection.

### Second Residual Connection

```text
attention residual: [2, 4, 8]
MLP output:          [2, 4, 8]
block output:        [2, 4, 8]
```

Conceptually:

```text
hidden_states_2 =
    hidden_states_1 + mlp(hidden_states_1)
```

### Vocabulary Projection

```text
input:  [2, 4, 8]
output: [2, 4, 20]
```

The final dimension changes from hidden width to vocabulary size.

Each token position now has one raw score for every vocabulary item.

## Logits and Next-Token Prediction

The vocabulary projection produces logits:

```text
[batch, sequence, vocabulary size]
```

For this example:

```text
[2, 4, 20]
```

A logit is an unnormalized vocabulary score.

Greedy token selection uses:

```python
predicted_token_ids = logits.argmax(dim=-1)
```

This transforms:

```text
[2, 4, 20] → [2, 4]
```

The selected integer at each position is the vocabulary entry with the highest current score.

This is not yet autoregressive generation. A real decode loop would repeatedly:

1. select the logits for the newest position;
2. choose one token;
3. append that token to the sequence;
4. run another model step.

## Execution Invariants

### Token-input invariant

```text
token_ids.shape == [batch, sequence]
token_ids.dtype == torch.int64
0 <= token_id < vocab_size
```

### Hidden-state invariant

```text
hidden_states.shape == [batch, sequence, hidden_width]
```

### Residual invariant

For:

```text
residual + transformed
```

both operands must have identical:

* shapes;
* dtypes;
* devices.

### Device invariant

Every tensor participating in one operation must occupy a compatible device.

For the attention projection:

```text
hidden_states.device ==
attention.projection.weight.device
```

## Compute and Memory Map

### Weight Memory

Persistent model parameters include:

* embedding table;
* attention projection weight and bias;
* MLP up-projection weight and bias;
* MLP down-projection weight and bias;
* vocabulary-projection weight.

These tensors remain allocated across requests while the model is loaded.

### Activation Memory

Intermediate activations include:

* embeddings;
* attention output;
* attention residual;
* expanded MLP activation;
* MLP output;
* block output;
* logits.

These tensors are created as data moves through the model.

The expanded MLP activation is larger than the hidden state:

```text
[2, 4, 32] versus [2, 4, 8]
```

This demonstrates why temporary activation memory can become significant in large transformer models.

### Temporary Tensors

Examples include:

* linear-operation outputs;
* GELU output;
* residual-add results;
* logits;
* any tensors retained for autograd during training.

Inference mode can avoid retaining many training-only intermediates.

### Compute Locations

The major operations are:

```text
embedding:
    indexed table lookup

attention placeholder:
    matrix multiplication

MLP:
    matrix multiplication
    GELU
    matrix multiplication

residual connections:
    element-wise addition

vocabulary projection:
    matrix multiplication
```

## GPU Runtime Evidence

The complete model was moved with:

```python
model = model.to("cuda")
```

Token IDs were created on the same device.

The observed valid GPU path was:

```text
token IDs:             cuda:0
embeddings:            cuda:0
attention output:      cuda:0
attention residual:    cuda:0
MLP expanded tensor:   cuda:0
MLP output:            cuda:0
block output:          cuda:0
logits:                cuda:0
```

All registered model parameters were confirmed to use:

```text
{"cuda"}
```

Hardware used:

```text
GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
compute capability: 8.9
GPU memory: 8188 MiB
```

Software runtime:

```text
OS: Ubuntu 22.04
Python: 3.12.13
PyTorch: 2.13.0+cu132
CUDA available through PyTorch: True
```

## Validated Tests

The focused tests cover:

1. end-to-end tensor shapes;
2. residual-width rejection;
3. embedding vocabulary-bound rejection;
4. CPU-weight/CUDA-input device-split rejection;
5. complete CUDA model execution.

Command:

```bash
pytest -q tests/test_tiny_block.py
```

Expected checkpoint result:

```text
5 passed
```

## Reproduced Bugs

### Embedding Bounds Failure

```text
reproductions/embedding_bounds/
```

Trigger:

```text
token_id == vocab_size
```

The raw framework error and improved input-contract behavior are preserved.

### CPU/GPU Device Split

```text
reproductions/device_split/
```

Trigger:

```text
hidden states on CUDA
projection parameters on CPU
```

The raw device error and improved local diagnostic are preserved.

## Relationship to a Serving Runtime

The toy model begins once token IDs already exist.

A serving runtime surrounds model execution with additional responsibilities:

```text
prompt text
    ↓
tokenization and input validation
    ↓
request creation
    ↓
scheduling and batching
    ↓
model execution
    ↓
sampling
    ↓
detokenization
    ↓
generated text
```

The transformer block performs the numerical model work. A runtime such as vLLM manages requests, memory, execution scheduling, and output production around that work.

## Current Limitations

This implementation does not yet include:

* positional information;
* LayerNorm or RMSNorm;
* Q, K, and V projections;
* causal masking;
* multi-head attention;
* multiple transformer blocks;
* dropout;
* weight tying;
* KV caching;
* autoregressive decoding;
* sampling;
* training or loss computation;
* production-performance optimization.

These omissions are intentional because the implementation is restricted to the current transformer-execution objective.
