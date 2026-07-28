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

        self.cross_attention:nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )       

        self.mlp = TinyMLP(hidden_size, device=self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "decoder hidden states must have shape "
                "[batch, target_sequence, hidden]"
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

        self_attention_output, _ = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,
            need_weights=False,
        )

        trace_tensor(
            "decoder.self_attention_output",
            self_attention_output,
        )

        hidden_states = add_residual(
            hidden_states,
            self_attention_output,
            name="Residual + Self Attention"
        )

        if encoder_memory is not None:
            hidden_states = self._apply_cross_attention(
                hidden_states=hidden_states,
                encoder_memory=encoder_memory,
            )

        mlp_output = self.mlp(hidden_states)

        trace_tensor(
            "decoder.mlp_output",
            mlp_output,
        )

        hidden_states = add_residual(
            hidden_states,
            mlp_output,
            name="Residual + MLP",
        )

        trace_tensor(
            "decoder.output",
            hidden_states,
        )

        return hidden_states

    def _apply_cross_attention(
        self,
        hidden_states: torch.Tensor,
        encoder_memory: torch.Tensor,
    ) -> torch.Tensor:
        if encoder_memory.ndim != 3:
            raise ValueError(
                "encoder memory must have shape "
                "[batch, source_sequence, hidden]"
            )

        if encoder_memory.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "encoder and decoder batch sizes must match: "
                f"encoder batch={encoder_memory.shape[0]}, "
                f"decoder batch={hidden_states.shape[0]}"
            )

        if encoder_memory.shape[-1] != self.hidden_size:
            raise ValueError(
                "encoder memory width mismatch: "
                f"expected {self.hidden_size}, "
                f"received {encoder_memory.shape[-1]}"
            )

        if encoder_memory.device != hidden_states.device:
            raise ValueError(
                "encoder memory and decoder states must be "
                "on the same device: "
                f"encoder={encoder_memory.device}, "
                f"decoder={hidden_states.device}"
            )

        trace_tensor(
            "decoder.encoder_memory",
            encoder_memory,
        )

        cross_attention_output, _ = self.cross_attention(
            query=hidden_states,
            key=encoder_memory,
            value=encoder_memory,
            need_weights=False,
        )

        trace_tensor(
            "decoder.cross_attention_output",
            cross_attention_output,
        )

        return add_residual(
            hidden_states,
            cross_attention_output,
            name="Residual + Cross Attention"
        )


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
        encoder_memory: torch.Tensor | None = None,
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

        return self.layer(
            hidden_states=hidden_states,
            encoder_memory=encoder_memory,
        )


if __name__=="__main__":
    device:torch.device = torch.device("cpu")
    causal_mask = build_causal_mask(8, device=device)
    print(causal_mask)