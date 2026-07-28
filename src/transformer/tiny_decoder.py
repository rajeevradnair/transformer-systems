import torch
from torch import nn

from transformer.tiny_block import (
    TinyMLP,
    TokenEmbedding,
    add_residual,
    trace_tensor,
)


def build_causal_mask(
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a Boolean mask where True means blocked."""

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    return torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )

class TinyDecoderLayer(nn.Module):
    """One causal Transformer decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        device:torch.device | None = None,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads: "
                f"hidden_size={hidden_size}, "
                f"num_heads={num_heads}"
            )

        self.device=device

        self.hidden_size = hidden_size

        self.self_attention:nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.mlp = TinyMLP(hidden_size, device=self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "decoder hidden states must have shape "
                "[batch, sequence, hidden]"
            )

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "decoder hidden width mismatch: "
                f"expected {self.hidden_size}, "
                f"received {hidden_states.shape[-1]}"
            )

        trace_tensor(
            "decoder.input",
            hidden_states,
        )

        sequence_length = hidden_states.shape[1]

        causal_mask = build_causal_mask(
            sequence_length=sequence_length,
            device=hidden_states.device,
        )

        trace_tensor(
            "decoder.causal_mask",
            causal_mask,
        )

        attention_output, _ = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,
            need_weights=False,
        )

        trace_tensor(
            "decoder.attention_output",
            attention_output,
        )

        hidden_states = add_residual(
            hidden_states,
            attention_output,
            name="Residual + Attention"
        )


        mlp_output = self.mlp(hidden_states)

        trace_tensor(
            "decoder.mlp_output",
            mlp_output,
        )

        hidden_states = add_residual(
            hidden_states,
            mlp_output,
            name="Residual + Attention",
        )

        trace_tensor(
            "decoder.output",
            hidden_states,
        )

        return hidden_states


class TinyDecoder(nn.Module):
    """Token IDs in, causal decoder states out."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_width=hidden_size,
        )

        self.layer = TinyDecoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        trace_tensor(
            "decoder.token_ids",
            token_ids,
        )

        hidden_states = self.embedding(token_ids)

        trace_tensor(
            "decoder.embeddings",
            hidden_states,
        )

        return self.layer(hidden_states)


if __name__=="__main__":
    device:torch.device = torch.device("cpu")
    causal_mask = build_causal_mask(8, device=device)
    print(causal_mask)