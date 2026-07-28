import torch

from transformer.tiny_decoder import (
    TinyDecoder,
    TinyDecoderLayer,
    build_causal_mask,
)

def test_causal_mask_blocks_only_future_positions() -> None:
    mask = build_causal_mask(
        sequence_length=4,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [False, True,  True,  True],
            [False, False, True,  True],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)

def test_decoder_layer_preserves_hidden_shape() -> None:
    torch.manual_seed(7)

    hidden_states = torch.randn(
        2,
        4,
        8,
        device=torch.device("cpu"),)

    decoder = TinyDecoderLayer(
        hidden_size=8,
        num_heads=2,
        device=torch.device("cpu"),
    )

    output = decoder(hidden_states)

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()

def test_decoder_converts_token_ids_to_hidden_states() -> None:
    torch.manual_seed(7)

    token_ids = torch.tensor(
        [
            [1, 4, 5, 6],
            [1, 7, 8, 9],
        ],
        dtype=torch.long,
    )

    decoder = TinyDecoder(
        vocab_size=20,
        hidden_size=8,
        num_heads=2,
    )

    output = decoder(token_ids)

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()

def test_decoder_early_states_ignore_future_token() -> None:
    torch.manual_seed(7)

    decoder = TinyDecoder(
        vocab_size=20,
        hidden_size=8,
        num_heads=2,
    )
    decoder.eval()

    first_input = torch.tensor(
        [[1, 4, 5, 6]],
        dtype=torch.long,
    )

    second_input = torch.tensor(
        [[1, 4, 5, 9]],
        dtype=torch.long,
    )

    with torch.inference_mode():
        first_output = decoder(first_input)
        second_output = decoder(second_input)

    assert torch.allclose(
        first_output[:, :3, :],
        second_output[:, :3, :],
        atol=1e-6,
    )

    assert not torch.allclose(
        first_output[:, 3, :],
        second_output[:, 3, :],
    )