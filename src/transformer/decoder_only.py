import torch
from torch import nn

from transformer.tiny_decoder import TinyDecoder
from transformer.tiny_block import trace_tensor


class DecoderOnlyModel(nn.Module):
    """A tiny GPT-style causal language model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.decoder = TinyDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        decoder_states = self.decoder(
            token_ids=token_ids,
        )

        trace_tensor(
            "decoder_only.states",
            decoder_states,
        )

        logits = self.output_projection(
            decoder_states,
        )

        trace_tensor(
            "decoder_only.logits",
            logits,
        )

        return logits