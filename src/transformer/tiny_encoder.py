import torch
from torch import nn

from transformer.tiny_block import TinyMLP, add_residual, trace_tensor, TokenEmbedding


class TinyEncoderLayer(nn.Module):
    """One bidirectional Transformer encoder layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads: "
                f"hidden_size={hidden_size}, num_heads={num_heads}"
            )

        self.hidden_size = hidden_size

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            device=device
        )

        self.mlp = TinyMLP(hidden_size, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "encoder hidden states must have shape "
                "[batch, sequence, hidden]"
            )

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "encoder hidden width mismatch: "
                f"expected {self.hidden_size}, "
                f"received {hidden_states.shape[-1]}"
            )

        trace_tensor("Layer / encoder.input", hidden_states)

        # since this is a self attention, the embedded tensor to use to derive Q, K, V is the same 
        attention_output, _ = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            need_weights=False,
        )

        trace_tensor("LAyer / encoder.attention_output", attention_output)

        hidden_states = add_residual(
            hidden_states,
            attention_output,
            name="Residual + Attention"
        )

        mlp_output = self.mlp(hidden_states)

        trace_tensor("Layer / encoder.mlp_output", mlp_output)

        hidden_states = add_residual(
            hidden_states,
            mlp_output,
            name="Residual + Attention",
        )

        trace_tensor("Layer / encoder.output", hidden_states)

        return hidden_states

class TinyEncoder(nn.Module):
    """Token embedding followed by one bidirectional encoder layer."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_width=hidden_size,
            device=device,
        )

        self.encoder_layer = TinyEncoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            device=device,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        trace_tensor("encoder.token_ids", token_ids)

        hidden_states = self.embedding(token_ids)

        trace_tensor("encoder.embeddings", hidden_states)

        encoder_output = self.encoder_layer(hidden_states)

        return encoder_output