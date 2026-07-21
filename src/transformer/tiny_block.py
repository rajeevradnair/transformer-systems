from __future__ import annotations

import torch
from torch import Tensor, nn 

HIDDEN_DIMENSION = 16
VOCAB_SIZE = 80
EXPANSION_FACTOR = 8

def trace_tensor(name: str, tensor: Tensor) -> None:
    """Print the execution metadata needed to follow one tensor."""

    print(
        f"{name:<40}"
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}"
    )

def validate_token_ids(token_ids: Tensor, vocab_size: int) -> None:
    """Validate the contract required by an embedding lookup."""

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape [batch, sequence], "
            f"but received shape {tuple(token_ids.shape)}"
        )

    if token_ids.dtype != torch.long:
        raise TypeError(
            "token_ids must use torch.long integer indices, "
            f"but received {token_ids.dtype}"
        )

    if token_ids is None or token_ids.numel() == 0:
        raise ValueError("token_ids must not be empty")

    minimum_id = int(token_ids.min().item())
    maximum_id = int(token_ids.max().item())

    if minimum_id < 0 or maximum_id >= vocab_size:
        raise ValueError(
            "token ID outside embedding vocabulary: "
            f"expected every ID in [0, {vocab_size - 1}], "
            f"but observed minimum={minimum_id}, maximum={maximum_id}"
        )


def require_same_shape(
    original: Tensor,
    transformed: Tensor,
    component_name: str,
) -> None:
    """Ensure a transformed tensor can participate in a residual add."""

    if original.shape != transformed.shape:
        raise ValueError(
            f"{component_name} violated the residual shape contract: "
            f"expected {tuple(original.shape)}, "
            f"but received {tuple(transformed.shape)}"
        )


class TokenEmbedding(nn.Module):
    """Convert token IDs into hidden-state vectors."""

    def __init__(self, vocab_size: int, hidden_width: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        self.device = device
        self.vocab_size = vocab_size
        self.hidden_width = hidden_width
        self.embedding = nn.Embedding(vocab_size, hidden_width, device=device)

    def forward(self, token_ids: Tensor) -> Tensor:
        validate_token_ids(token_ids, self.vocab_size)

        hidden_states = self.embedding(token_ids)

        trace_tensor("Layer / Embeddings lookup", hidden_states)

        return hidden_states


class AttentionPlaceholder(nn.Module):
    """Preserve the shape contract of a transformer attention layer."""

    def __init__(self, hidden_width: int, *, device: torch.device | str | None = None,) -> None:
        super().__init__()

        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        self.hidden_width = hidden_width
        self.device=device
        self.layer = nn.Linear(hidden_width, hidden_width, device=self.device)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch, sequence, hidden_width], "
                f"but received {tuple(hidden_states.shape)}"
            )

        observed_width = hidden_states.shape[-1]

        if observed_width != self.hidden_width:
            raise ValueError(
                "attention input width mismatch: "
                f"expected {self.hidden_width}, "
                f"but received {observed_width}"
            )


        parameter_device = self.layer.weight.device

        if hidden_states.device != parameter_device:
            raise ValueError(
                "attention device mismatch: "
                f"hidden_states={hidden_states.device}, "
                f"projection_weight={parameter_device}"
            )

        attention_output = self.layer(hidden_states)

        if attention_output.shape != hidden_states.shape:
            raise RuntimeError(
                "attention output must preserve the complete hidden-state shape: "
                f"input={tuple(hidden_states.shape)}, "
                f"output={tuple(attention_output.shape)}"
            )

        trace_tensor("Layer / Attention output", attention_output)

        return attention_output


def add_residual(
    residual: Tensor,
    transformed: Tensor,
    *,
    name: str,
) -> Tensor:
    """Add a transformed tensor back to its residual input."""

    if residual.shape != transformed.shape:
        raise ValueError(
            f"{name} residual shape mismatch: "
            f"residual={tuple(residual.shape)}, "
            f"transformed={tuple(transformed.shape)}"
        )

    if residual.device != transformed.device:
        raise ValueError(
            f"{name} residual device mismatch: "
            f"residual={residual.device}, "
            f"transformed={transformed.device}"
        )

    if residual.dtype != transformed.dtype:
        raise ValueError(
            f"{name} residual dtype mismatch: "
            f"residual={residual.dtype}, "
            f"transformed={transformed.dtype}"
        )

    output = residual + transformed
    trace_tensor(name, output)

    return output


class TinyMLP(nn.Module):
    """Expand and contract each token's hidden representation."""

    def __init__(
        self,
        hidden_width: int,
        expansion_factor: int = 4,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")

        self.hidden_width = hidden_width
        self.intermediate_width = hidden_width * expansion_factor

        self.up_projection = nn.Linear(
            hidden_width,
            self.intermediate_width,
            device=device
        )
        self.activation = nn.GELU()
        self.down_projection = nn.Linear(
            self.intermediate_width,
            hidden_width,
            device=device
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "MLP input must have shape "
                "[batch, sequence, hidden_width], "
                f"but received {tuple(hidden_states.shape)}"
            )

        observed_width = hidden_states.shape[-1]

        if observed_width != self.hidden_width:
            raise ValueError(
                "MLP input width mismatch: "
                f"expected {self.hidden_width}, "
                f"but received {observed_width}"
            )

        trace_tensor("Layer / input_to_mlp", hidden_states)

        expanded = self.up_projection(hidden_states)
        trace_tensor("Layer / mlp_expansion", expanded)

        activated = self.activation(expanded)

        mlp_output = self.down_projection(activated)

        if mlp_output.shape != hidden_states.shape:
            raise RuntimeError(
                "MLP output must preserve the input shape: "
                f"input={tuple(hidden_states.shape)}, "
                f"output={tuple(mlp_output.shape)}"
            )

        trace_tensor("Layer / mlp_contraction", mlp_output)

        return mlp_output


class TinyTransformerBlock(nn.Module):
    """Apply attention and MLP updates with residual connections."""

    def __init__(
        self,
        hidden_width: int,
        expansion_factor: int = 4,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.attention = AttentionPlaceholder(hidden_width, device=device)
        self.mlp = TinyMLP(
            hidden_width=hidden_width,
            expansion_factor=expansion_factor,
            device=device
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention_output = self.attention(hidden_states)

        hidden_states = add_residual(
            hidden_states,
            attention_output,
            name="Layer / attention_+_residual"
        )

        mlp_output = self.mlp(hidden_states)

        hidden_states = add_residual(
            hidden_states,
            mlp_output,
            name="Layer / mlp_residual",
        )

        return hidden_states


class VocabularyProjection(nn.Module):
    """Project each hidden state into vocabulary-sized logits."""

    def __init__(
        self,
        hidden_width: int,
        vocab_size: int,
        *,
        device:torch.device | None = None
    ) -> None:
        super().__init__()

        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.hidden_width = hidden_width
        self.vocab_size = vocab_size

        self.projection_layer = nn.Linear(
            hidden_width,
            vocab_size,
            bias=False,
            device=device
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "vocabulary projection input must have shape "
                "[batch, sequence, hidden_width], "
                f"but received {tuple(hidden_states.shape)}"
            )

        observed_width = hidden_states.shape[-1]

        if observed_width != self.hidden_width:
            raise ValueError(
                "vocabulary projection width mismatch: "
                f"expected {self.hidden_width}, "
                f"but received {observed_width}"
            )

        logits = self.projection_layer(hidden_states)

        expected_shape = (
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.vocab_size,
        )

        if logits.shape != expected_shape:
            raise RuntimeError(
                "unexpected logits shape: "
                f"expected {expected_shape}, "
                f"received {tuple(logits.shape)}"
            )

        trace_tensor("Layer / logits", logits)

        return logits


class TinyLanguageModel(nn.Module):
    """Run token IDs through one transformer-style block into logits."""

    def __init__(
        self,
        vocab_size: int,
        hidden_width: int,
        expansion_factor: int = 4,
        *,
        device:torch.device | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_width = hidden_width

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_width=hidden_width,
            device=device
        )

        self.block = TinyTransformerBlock(
            hidden_width=hidden_width,
            expansion_factor=expansion_factor,
            device=device
        )

        self.output_projection = VocabularyProjection(
            hidden_width=hidden_width,
            vocab_size=vocab_size,
            device=device
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        trace_tensor("token_ids", token_ids)

        hidden_states = self.token_embedding(token_ids)
        hidden_states = self.block(hidden_states)
        logits = self.output_projection(hidden_states)

        return logits



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    token_ids = torch.tensor(
        [
            [27,63,19],
            [42,27,69]
        ],
        dtype=torch.long,
        device=device,
    )
    trace_tensor("Raw token ids", token_ids)

    embedding_lookup = TokenEmbedding(VOCAB_SIZE, HIDDEN_DIMENSION, device=device)

    embedded_token_ids = embedding_lookup(token_ids)

    trace_tensor("Embedded token ids", embedded_token_ids)

    attention_layer = AttentionPlaceholder(HIDDEN_DIMENSION, device=device)

    attention_output = attention_layer(embedded_token_ids)

    trace_tensor("Attended tokens", attention_output)

    residual_output = add_residual(embedded_token_ids, attention_output, name = "Residual + Attention")

    trace_tensor("Residual output", residual_output)

    mlp_layer = TinyMLP(HIDDEN_DIMENSION, expansion_factor=EXPANSION_FACTOR, device=device)

    mlp_output = mlp_layer(residual_output)

    trace_tensor("MLP output", mlp_output)

    transformer_block = TinyTransformerBlock(HIDDEN_DIMENSION, EXPANSION_FACTOR, device=device)

    block_output = transformer_block(embedded_token_ids)

    trace_tensor("Block output", block_output)

    project_to_vocab = VocabularyProjection(HIDDEN_DIMENSION, VOCAB_SIZE, device=device)

    vocab_projected_tensor: torch.Tensor = project_to_vocab(block_output)

    trace_tensor("Vocabulary projected tensor", vocab_projected_tensor)

    predicted_token_ids = vocab_projected_tensor.argmax(dim=-1)

    trace_tensor("Predicted token ids", predicted_token_ids)

    print(predicted_token_ids)

    print()
    print("**********************")
    print("Composite model")
    print("**********************")
    
    model = TinyLanguageModel(VOCAB_SIZE, HIDDEN_DIMENSION, EXPANSION_FACTOR, device=device)

    model_parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Model parameter count: {model_parameter_count}")

    logits:torch.Tensor = model(token_ids)

    trace_tensor("Logit for each token", logits)

    next_token_ids = logits.argmax(dim=-1)

    trace_tensor("Next token ids", next_token_ids)



