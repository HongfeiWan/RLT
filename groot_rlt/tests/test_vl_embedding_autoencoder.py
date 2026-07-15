import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from groot_rlt.representation.train_vl_embedding_autoencoder import (
    RL_TOKEN_ARCHITECTURE,
    RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
    VLTokenAutoencoder,
    VLTokenAutoencoderConfig,
    compact_cached_vl_tokens,
    make_arg_parser,
    make_ema_state,
    masked_mse_loss,
    maybe_load_checkpoint,
    save_checkpoint,
    update_ema_state,
    update_learning_rate,
    validate_prefix_corruption_args,
    validate_strict_checkpoint_payload,
)


def make_autoencoder() -> VLTokenAutoencoder:
    torch.manual_seed(7)
    return VLTokenAutoencoder(
        VLTokenAutoencoderConfig(
            input_dim=16,
            model_dim=16,
            rl_token_dim=16,
            max_vl_tokens=8,
            encoder_layers=2,
            decoder_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            use_decoder_cross_attention=True,
        )
    )


def test_openpi_encoder_decoder_data_dependencies() -> None:
    model = make_autoencoder().eval()
    mask = torch.tensor([[True, True, True, False, False, False, False, False]])
    prefix_a = torch.randn(1, 8, 16)
    prefix_b = prefix_a.clone()
    prefix_b[:, 0] += 3.0

    with torch.no_grad():
        token_a = model.encode_rl_token(prefix_a, mask)
        token_b = model.encode_rl_token(prefix_b, mask)
        assert token_a.shape == (1, 16)
        assert not torch.equal(token_a, token_b)

        target_a = torch.zeros(1, 8, 16)
        target_b = torch.randn(1, 8, 16)
        decoded_a = model.decode_from_rl_token(token_a, target_a, mask)
        decoded_b = model.decode_from_rl_token(token_a, target_b, mask)
        torch.testing.assert_close(decoded_a, decoded_b, rtol=0.0, atol=0.0)

        decoded_other_token = model.decode_from_rl_token(token_b, target_a, mask)
        assert not torch.equal(decoded_a, decoded_other_token)

    model.train()
    output = model(prefix_a, mask)
    loss = masked_mse_loss(output["reconstruction"], prefix_a, mask)
    loss.backward()
    assert model.query_token.grad is not None
    assert model.decoder_query.grad is not None


def test_openpi_decoder_rejects_teacher_forced_prefix() -> None:
    model = make_autoencoder()
    mask = torch.ones(1, 8, dtype=torch.bool)
    target = torch.randn(1, 8, 16)
    rl_token = torch.randn(1, 16)

    try:
        model.decode_from_rl_token(
            rl_token,
            target,
            mask,
            decoder_prefix_embeddings=target[:, :-1],
        )
    except ValueError as exc:
        assert "does not accept ground-truth prefix" in str(exc)
    else:
        raise AssertionError("Strict openpi-RLT decoder accepted teacher-forced embeddings")


def test_openpi_encoder_ignores_padding_and_rejects_empty_samples() -> None:
    model = make_autoencoder().eval()
    mask = torch.tensor([[True, True, False, False, False, False, False, False]])
    prefix = torch.randn(1, 8, 16)
    changed_padding = prefix.clone()
    changed_padding[:, 2:] = torch.randn_like(changed_padding[:, 2:]) * 1000.0

    with torch.no_grad():
        expected = model.encode_rl_token(prefix, mask)
        actual = model.encode_rl_token(changed_padding, mask)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    try:
        model.encode_rl_token(prefix, torch.zeros_like(mask))
    except ValueError as exc:
        assert "at least one valid prefix token" in str(exc)
    else:
        raise AssertionError("Strict encoder accepted a sample with no valid prefix tokens")


def test_strict_cli_rejects_decoder_prefix_corruption() -> None:
    args = make_arg_parser().parse_args(["--decoder-prefix-corruption"])
    try:
        validate_prefix_corruption_args(args)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("Strict training accepted decoder prefix corruption")


def test_strict_checkpoint_schema_is_fail_closed() -> None:
    state = make_autoencoder().state_dict()
    validate_strict_checkpoint_payload({"autoencoder": state})
    validate_strict_checkpoint_payload(
        {
            "schema_version": RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
            "architecture": RL_TOKEN_ARCHITECTURE,
            "autoencoder": state,
        }
    )

    for invalid in (
        {"autoencoder": {"query_token": state["query_token"]}},
        {
            "schema_version": RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
            "architecture": "unknown",
            "autoencoder": state,
        },
    ):
        try:
            validate_strict_checkpoint_payload(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid strict checkpoint payload: {invalid.keys()}")


def test_compact_cached_vl_tokens_selects_only_images_and_actual_length() -> None:
    packed = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    packed_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    packed_image_mask = torch.tensor(
        [
            [False, True, False, True, True, False],
            [True, False, True, False, False, False],
        ]
    )

    compact, mask, image_mask, original_counts, selected_counts = compact_cached_vl_tokens(
        packed,
        packed_mask,
        packed_image_mask,
        token_scope="image",
        max_tokens=4,
        token_sampling="head",
    )

    assert compact.shape == (2, 3, 2)
    torch.testing.assert_close(compact[0], packed[0, [1, 3, 4]])
    torch.testing.assert_close(compact[1, :2], packed[1, [0, 2]])
    torch.testing.assert_close(compact[1, 2], torch.zeros(2))
    assert torch.equal(mask, torch.tensor([[True, True, True], [True, True, False]]))
    assert torch.equal(image_mask, mask)
    assert original_counts == [3, 2]
    assert selected_counts == [3, 2]


def test_compact_cached_vl_tokens_subsamples_non_image_tokens() -> None:
    packed = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    packed_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    packed_image_mask = torch.tensor(
        [
            [False, True, False, True, True, False],
            [True, False, True, False, False, False],
        ]
    )

    compact, mask, image_mask, original_counts, selected_counts = compact_cached_vl_tokens(
        packed,
        packed_mask,
        packed_image_mask,
        token_scope="non_image",
        max_tokens=2,
        token_sampling="head",
    )

    assert compact.shape == (2, 2, 2)
    torch.testing.assert_close(compact[0], packed[0, [0, 2]])
    torch.testing.assert_close(compact[1], packed[1, [1, 3]])
    assert bool(mask.all())
    assert not bool(image_mask.any())
    assert original_counts == [2, 4]
    assert selected_counts == [2, 2]


def test_reference_learning_rate_schedule_keeps_30k_horizon() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    args = SimpleNamespace(
        learning_rate=2.5e-5,
        min_learning_rate=2.5e-6,
        warmup_steps=1000,
        lr_decay_steps=30000,
        max_steps=10000,
    )

    assert update_learning_rate(optimizer, 1, args) == args.learning_rate / 1001
    assert update_learning_rate(optimizer, 1001, args) == args.learning_rate

    progress = (9999 - args.warmup_steps) / (args.lr_decay_steps - args.warmup_steps)
    expected_10k = args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    assert math.isclose(
        update_learning_rate(optimizer, 10000, args),
        expected_10k,
        rel_tol=1e-12,
    )
    assert update_learning_rate(optimizer, 30001, args) == args.min_learning_rate


def test_ema_checkpoint_uses_ema_for_inference_and_raw_for_resume() -> None:
    model = make_autoencoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema_state = make_ema_state(model)
    initial_ema = {name: value.clone() for name, value in ema_state.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    update_ema_state(ema_state, model, decay=0.9)

    raw_state = model.state_dict()
    for name, value in ema_state.items():
        if torch.is_floating_point(value):
            expected = initial_ema[name] * 0.9 + raw_state[name] * 0.1
            torch.testing.assert_close(value, expected)

    args = SimpleNamespace(ema_decay=0.9)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "000001.pt"
        save_checkpoint(
            checkpoint_path,
            1,
            model,
            ema_state,
            optimizer,
            model.config,
            args,
            0.5,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["ema_decay"] == 0.9
        assert "autoencoder_raw" in checkpoint
        for name in ema_state:
            torch.testing.assert_close(checkpoint["autoencoder"][name], ema_state[name])
            torch.testing.assert_close(checkpoint["autoencoder_raw"][name], raw_state[name])

        restored_model = make_autoencoder()
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
        step, restored_ema = maybe_load_checkpoint(
            str(checkpoint_path),
            restored_model,
            restored_optimizer,
            torch.device("cpu"),
        )
        assert step == 1
        assert restored_ema is not None
        for name, value in restored_model.state_dict().items():
            torch.testing.assert_close(value, raw_state[name])
            torch.testing.assert_close(restored_ema[name], ema_state[name])
