import torch

from transformer.tiny_encoder import TinyEncoderLayer, TinyEncoder


def test_encoder_layer_preserves_hidden_shape() -> None:
    torch.manual_seed(7)
    device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    sequence_length = 4
    hidden_size = 8

    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        device=device,
    )

    encoder = TinyEncoderLayer(
        hidden_size=hidden_size,
        num_heads=2,
        device=device,
    )

    output = encoder(hidden_states)

    assert output.shape == (
        batch_size,
        sequence_length,
        hidden_size,
    )

    assert torch.isfinite(output).all()

def test_encoder_converts_token_ids_to_contextual_states() -> None:
    torch.manual_seed(11)
    device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")

    vocab_size = 20
    hidden_size = 8

    token_ids = torch.tensor(
        [
            [1, 4, 5, 2],
            [1, 8, 9, 2],
        ],
        dtype=torch.long,
    )

    encoder = TinyEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=2,
        device=device
    )

    output = encoder(token_ids)

    assert output.shape == (2, 4, hidden_size)
    assert torch.isfinite(output).all()