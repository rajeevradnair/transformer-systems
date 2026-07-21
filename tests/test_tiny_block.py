import pytest

import torch

from src.transformer.tiny_block import TinyLanguageModel, add_residual


def test_tiny_language_model_produces_expected_shapes() -> None:
    torch.manual_seed(7)

    batch_size = 2
    sequence_length = 4
    vocab_size = 20
    hidden_width = 8

    model = TinyLanguageModel(
        vocab_size=vocab_size,
        hidden_width=hidden_width,
        expansion_factor=4,
    )

    token_ids:torch.Tensor = torch.tensor(
        [
            [1, 4, 7, 3],
            [2, 9, 5, 6],
        ],
        dtype=torch.long,
    )

    assert not token_ids.is_floating_point() and not token_ids.is_complex()

    logits = model(token_ids)

    assert token_ids.shape == (
        batch_size,
        sequence_length,
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size,
    )

    assert logits.is_floating_point()

    
    predicted_token_ids = logits.argmax(dim=-1)

    assert not predicted_token_ids.is_floating_point() and not predicted_token_ids.is_complex()

    assert predicted_token_ids.shape == token_ids.shape
    assert int(predicted_token_ids.min()) >= 0
    assert int(predicted_token_ids.max()) < vocab_size

def test_residual_rejects_hidden_width_mismatch() -> None:
    residual = torch.zeros(
        2,
        4,
        8,
        dtype=torch.float32,
    )

    transformed = torch.zeros(
        2,
        4,
        9,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="residual shape mismatch"):
        add_residual(residual=residual, transformed=transformed, name="attention + residual")