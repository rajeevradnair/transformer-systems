# Embedding Bounds Failure

## Summary

A token ID equal to `vocab_size` was passed to an embedding table.

For an embedding table with `vocab_size = 20`, the only valid token IDs are:

```text
0 through 19
```

Token ID `20` attempts to access a row that does not exist.

## Expected Behavior

The model input boundary should reject any token ID outside:

```text
0 <= token_id < vocab_size
```

The error should identify:

* the valid token range;
* the observed minimum token ID;
* the observed maximum token ID;
* the violated embedding-input contract.

## Observed Behavior

Calling `torch.nn.Embedding` directly with token ID `20` produces a low-level framework exception:

```text
IndexError: index out of range in self
```

The raw exception does not identify the configured vocabulary size or the offending token value.

The complete raw traceback is preserved in:

```text
reproductions/embedding_bounds/raw_output.txt
```

## Trigger

Configuration:

```text
vocab_size = 20
hidden_width = 8
```

Input:

```python
torch.tensor(
    [[1, 4, 20, 3]],
    dtype=torch.long,
)
```

The triggering value is:

```text
token_id = vocab_size
```

## Deterministic Failing Assertion

The raw behavior can be expressed as:

```python
embedding = torch.nn.Embedding(
    num_embeddings=20,
    embedding_dim=8,
)

invalid_token_ids = torch.tensor(
    [[1, 4, 20, 3]],
    dtype=torch.long,
)

embedding(invalid_token_ids)
```

This deterministically fails because the embedding table contains rows `0` through `19`, but the operation requests row `20`.

## First Causal Error

The first causal error is not inside the embedding arithmetic.

The first causal error occurs when the caller supplies a token ID that violates the vocabulary bounds:

```text
maximum_token_id >= vocab_size
```

The later `IndexError` is the resulting symptom.

## State Owner

The token-input validation boundary owns this invariant.

In the toy implementation, the narrowest owner is:

```text
TokenEmbedding.forward
```

It owns:

* the configured `vocab_size`;
* the incoming token IDs;
* the transition from token IDs to embedding vectors.

The residual block, MLP, and vocabulary projection do not own this contract.

## Root-Cause Hypothesis

The failure occurs because the input reaches `nn.Embedding` without being checked against the configured vocabulary size.

Possible upstream causes in a real system include:

* tokenizer and model vocabulary mismatch;
* malformed manually supplied token IDs;
* incorrect special-token configuration;
* stale tokenizer artifacts;
* data corruption during request preprocessing.

For this reproduction, the confirmed cause is a deliberately supplied out-of-range token ID.

## Confirming Evidence

The hypothesis is confirmed by these observations:

1. `vocab_size` is `20`.
2. The valid table indices are `0` through `19`.
3. The input contains token ID `20`.
4. Replacing `20` with `19` allows the lookup to succeed.
5. The failure occurs before attention, residual, MLP, or output projection execution.
6. Explicit validation catches the same input before `nn.Embedding` executes.

## Rejecting Evidence

The following potential explanations are rejected:

### Wrong tensor shape

The input shape is valid:

```text
[batch, sequence] = [1, 4]
```

### Wrong dtype

The input dtype is valid for embedding lookup:

```text
torch.int64
```

### Device mismatch

Both the embedding table and token IDs are on the CPU in the raw reproduction.

### Invalid model weights

The failure is independent of embedding-weight values because row `20` does not exist.

## Smallest Safe Fix

Validate token IDs immediately before embedding lookup:

```python
minimum_id = int(token_ids.min().item())
maximum_id = int(token_ids.max().item())

if minimum_id < 0 or maximum_id >= vocab_size:
    raise ValueError(
        "token ID outside embedding vocabulary: "
        f"expected every ID in [0, {vocab_size - 1}], "
        f"but observed minimum={minimum_id}, maximum={maximum_id}"
    )
```

This is the smallest safe fix because it:

* changes no valid execution behavior;
* fails at the owning boundary;
* provides actionable diagnostics;
* does not modify embedding weights or downstream model logic.

## Validation

The focused regression test is:

```text
tests/test_tiny_block.py::test_embedding_rejects_token_id_equal_to_vocab_size
```

Run it with:

```bash
pytest -q \
  tests/test_tiny_block.py::test_embedding_rejects_token_id_equal_to_vocab_size
```

Expected result:

```text
1 passed
```

The complete valid-path test suite is run with:

```bash
pytest -q tests/test_tiny_block.py
```

Expected result at this checkpoint:

```text
3 passed
```

## Before-and-After Behavior

### Before validation

```text
invalid token ID
    ↓
nn.Embedding
    ↓
raw IndexError
```

### After validation

```text
invalid token ID
    ↓
TokenEmbedding input contract
    ↓
descriptive ValueError
```

## Remaining Uncertainty

The local reproduction proves the model-side contract but does not yet determine which component would generate an invalid token ID in a production serving system.

Potential production owners could include:

* tokenizer initialization;
* request preprocessing;
* special-token configuration;
* model and tokenizer artifact compatibility.

Those broader integration paths are outside this reproduction’s scope.

## Reproduction Artifacts

```text
reproductions/embedding_bounds/
├── analysis.md
├── raw_output.txt
└── reproduce_raw.py
```
