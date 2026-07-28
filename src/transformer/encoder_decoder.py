import torch
from torch import nn

from transformer.tiny_decoder import TinyDecoder
from transformer.tiny_encoder import TinyEncoder


class EncoderDecoderModel(nn.Module):
    """A tiny encoder-decoder Transformer."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.encoder = TinyEncoder(
            vocab_size=source_vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.decoder = TinyDecoder(
            vocab_size=target_vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            target_vocab_size,
        )

    def forward(
        self,
        source_token_ids: torch.Tensor,
        decoder_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_memory = self.encoder(
            source_token_ids
        )

        decoder_states = self.decoder(
            token_ids=decoder_token_ids,
            encoder_memory=encoder_memory,
        )

        logits = self.output_projection(
            decoder_states
        )

        return logits