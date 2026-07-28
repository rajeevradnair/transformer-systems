import torch

from transformer.decoder_only import DecoderOnlyModel


def test_decoder_only_produces_vocabulary_logits() -> None:
    torch.manual_seed(7)

    model = DecoderOnlyModel(
        vocab_size=20,
        hidden_size=8,
        num_heads=2,
    )

    token_ids = torch.tensor(
        [[1, 4, 5, 6]],
        dtype=torch.long,
    )

    logits = model(token_ids)

    print("logits shape", logits.shape)

    last_token_logit=logits[:,-1,:]

    print(last_token_logit.shape)
    print(last_token_logit)
    print(torch.argmax(last_token_logit, dim=-1))

    assert logits.shape == (1, 4, 20)
    assert torch.isfinite(logits).all()