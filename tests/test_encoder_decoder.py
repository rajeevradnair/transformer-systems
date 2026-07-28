import torch

from transformer.encoder_decoder import EncoderDecoderModel


def test_encoder_decoder_produces_target_logits() -> None:
    torch.manual_seed(7)

    model = EncoderDecoderModel(
        source_vocab_size=20,
        target_vocab_size=30,
        hidden_size=8,
        num_heads=2,
    )

    source_token_ids = torch.tensor(
        [[4, 5]],
        dtype=torch.long,
    )

    decoder_token_ids = torch.tensor(
        [[1, 4, 6]],
        dtype=torch.long,
    )

    logits = model(
        source_token_ids=source_token_ids,
        decoder_token_ids=decoder_token_ids,
    )

    assert logits.shape == (1, 3, 30)
    assert torch.isfinite(logits).all()

def test_encoder_decoder_output_changes_with_source() -> None:
    torch.manual_seed(7)

    model = EncoderDecoderModel(
        source_vocab_size=20,
        target_vocab_size=30,
        hidden_size=8,
        num_heads=2,
    )
    model.eval()

    first_source = torch.tensor(
        [[4, 5]],
        dtype=torch.long,
    )

    second_source = torch.tensor(
        [[4, 9]],
        dtype=torch.long,
    )

    decoder_token_ids = torch.tensor(
        [[1, 4, 6]],
        dtype=torch.long,
    )

    with torch.inference_mode():
        first_logits = model(
            first_source,
            decoder_token_ids,
        )

        print("***************************************")

        second_logits = model(
            second_source,
            decoder_token_ids,
        )

    assert not torch.allclose(
        first_logits,
        second_logits,
    )