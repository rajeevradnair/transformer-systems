import pytest

import torch

from src.transformer.tiny_block import TinyLanguageModel, add_residual, TokenEmbedding, AttentionPlaceholder


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


def test_embedding_rejects_token_id_equal_to_vocab_size() -> None:
    vocab_size = 20

    embedding = TokenEmbedding(
        vocab_size=vocab_size,
        hidden_width=8,
    )

    invalid_token_ids = torch.tensor(
        [[1, 4, vocab_size, 3]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError) as error:
        embedding(invalid_token_ids)

    diagnostic = str(error.value)

    assert "outside embedding vocabulary" in diagnostic
    assert "expected every ID in [0, 19]" in diagnostic
    assert "maximum=20" in diagnostic


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Device-split reproduction requires CUDA",
)
def test_attention_reproduces_cpu_gpu_device_split() -> None:
    hidden_width = 8

    embedding = TokenEmbedding(
        vocab_size=20,
        hidden_width=hidden_width,
    ).cpu()

    attention = AttentionPlaceholder(
        hidden_width=hidden_width,
    ).cpu()

    token_ids = torch.tensor(
        [[1, 4, 7, 3]],
        dtype=torch.long,
        device="cpu",
    )

    hidden_states = embedding(token_ids)
    hidden_states = hidden_states.to("cuda")

    assert hidden_states.shape == (1, 4, hidden_width)
    assert hidden_states.device.type == "cuda"
    assert attention.layer.weight.device.type == "cpu"

    with pytest.raises(RuntimeError) as error:
       attention(hidden_states)

    diagnostic = str(error.value).lower()

    '''
    assert "device" in diagnostic
    assert "cpu" in diagnostic
    assert "cuda" in diagnostic
    '''