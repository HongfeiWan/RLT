"""Serve GR00T ``z_rl``/reference chunks using the RLT Machine-A protocol."""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from groot_rlt.groot_repo import ensure_groot_repo
from groot_rlt.serving import msgpack_numpy
from groot_rlt.serving.groot_feature_policy import (
    FeatureContract,
    GrootN1d7FeatureBackend,
    MachineAFeaturePolicy,
)

LOGGER = logging.getLogger(__name__)


def _parse_key_map(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    targets: set[str] = set()
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"expected SOURCE=TARGET, got {value!r}")
        source, target = (part.strip() for part in value.split("=", 1))
        if not source or not target or source in result or target in targets:
            raise ValueError(f"invalid or duplicate key mapping: {value!r}")
        result[source] = target
        targets.add(target)
    return result


def _parse_state_layout(values: list[str] | None) -> tuple[tuple[str, int], ...] | None:
    if not values:
        return None
    layout = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=DIM, got {value!r}")
        key, raw_dim = (part.strip() for part in value.split("=", 1))
        dim = int(raw_dim)
        if not key or dim <= 0 or any(existing == key for existing, _ in layout):
            raise ValueError(f"invalid or duplicate state field: {value!r}")
        layout.append((key, dim))
    return tuple(layout)


def _groot_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class WebsocketFeatureServer:
    """Small msgpack-numpy server compatible with ``MachineAFeatureClient``."""

    def __init__(
        self,
        policy: MachineAFeaturePolicy,
        *,
        host: str,
        port: int,
        metadata: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = metadata

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            import websockets.asyncio.server as websocket_server
        except ImportError as exc:
            raise RuntimeError(
                "WebSocket serving requires the 'serve' extra: pip install -e '.[serve]'"
            ) from exc

        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: Any) -> None:
        import websockets

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        previous_total = None
        while True:
            try:
                started = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv())
                inference_started = time.monotonic()
                response = self.policy.infer(request)
                inference_ms = (time.monotonic() - inference_started) * 1000.0
                response["server_timing"] = {"infer_ms": inference_ms}
                if previous_total is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total * 1000.0
                await websocket.send(packer.pack(response))
                previous_total = time.monotonic() - started
            except websockets.ConnectionClosed:
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal feature server error")
                raise

    @staticmethod
    def _health_check(connection: Any, request: Any) -> Any | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-repo-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--vlm-model-path", type=str, required=True)
    encoder_source = parser.add_mutually_exclusive_group(required=True)
    encoder_source.add_argument(
        "--rl-token-encoder-artifact",
        type=str,
        help="Verified encoder-only EMA artifact (recommended serving path).",
    )
    encoder_source.add_argument(
        "--legacy-rl-token-checkpoint",
        "--rl-token-checkpoint",
        dest="legacy_rl_token_checkpoint",
        type=str,
        help=(
            "Explicit trusted full strict checkpoint compatibility path; this is never used "
            "as a fallback for an artifact load failure."
        ),
    )
    parser.add_argument("--prefix-cache-manifest", type=str, required=True)
    parser.add_argument("--expected-checkpoint-fingerprint", type=str, required=True)
    parser.add_argument("--expected-cache-fingerprint", type=str, required=True)
    parser.add_argument(
        "--expected-vlm-content-fingerprint",
        type=str,
        required=True,
        help=(
            "Deployment-time content hash for --vlm-model-path. This protects the current "
            "deployment but is not retroactively part of representation-training lineage."
        ),
    )
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--z-dim", type=int, default=2048)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument(
        "--proprio-dim",
        type=int,
        default=19,
        help=(
            "Actor/critic proprio dimension. Nero defaults to EEF9+hand10 while the "
            "frozen 400k model still receives its complete 26D state."
        ),
    )
    parser.add_argument("--proprio-key", action="append", default=None)
    parser.add_argument(
        "--image-key",
        action="append",
        default=None,
        metavar="SOURCE=TARGET",
        help="Map a flat RLT observation image key to a GR00T video key.",
    )
    parser.add_argument(
        "--flat-state-field",
        action="append",
        default=None,
        metavar="KEY=DIM",
        help="Ordered split for flat RLT state; Nero defaults to eef9+hand10+arm7.",
    )
    parser.add_argument("--default-instruction", type=str, default=None)
    parser.add_argument(
        "--num-inference-timesteps",
        "--denoise-steps",
        dest="num_inference_timesteps",
        type=int,
        default=None,
        help="Override checkpoint flow denoising steps (use 32 for the validated Nero setup).",
    )
    parser.add_argument("--token-scope", choices=("all", "image", "non_image"), default=None)
    parser.add_argument("--token-sampling", choices=("head", "tail", "uniform"), default=None)
    parser.add_argument("--max-vl-tokens", type=int, default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = make_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    repo_root = ensure_groot_repo(args.groot_repo_path)

    from groot_rlt.integration.checkpoint_policy_utils import resolve_processor_path

    model_path = Path(args.model_path).expanduser().resolve()
    processor_path = (
        Path(args.processor_path).expanduser().resolve()
        if args.processor_path
        else resolve_processor_path(model_path)
    )
    contract = FeatureContract(
        z_dim=args.z_dim,
        chunk_len=args.chunk_len,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
    )
    backend = GrootN1d7FeatureBackend(
        model_path=model_path,
        processor_path=processor_path,
        vlm_model_path=args.vlm_model_path,
        prefix_cache_manifest=args.prefix_cache_manifest,
        expected_checkpoint_fingerprint=args.expected_checkpoint_fingerprint,
        expected_cache_fingerprint=args.expected_cache_fingerprint,
        expected_vlm_content_fingerprint=args.expected_vlm_content_fingerprint,
        rl_token_encoder_artifact=args.rl_token_encoder_artifact,
        legacy_rl_token_checkpoint=args.legacy_rl_token_checkpoint,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        contract=contract,
        strict=args.strict,
        token_scope=args.token_scope,
        token_sampling=args.token_sampling,
        max_vl_tokens=args.max_vl_tokens,
        proprio_keys=None if args.proprio_key is None else tuple(args.proprio_key),
        image_key_map=_parse_key_map(args.image_key),
        flat_state_layout=_parse_state_layout(args.flat_state_field),
        default_instruction=args.default_instruction,
        num_inference_timesteps=args.num_inference_timesteps,
    )
    metadata = {
        "backend": "groot-n1.7",
        "groot_repo": str(repo_root),
        "groot_commit": _groot_commit(repo_root),
        "model_path": str(model_path),
        "rl_token_contract": dict(backend.rl_token_serving_metadata),
        # The backend currently evaluates one observation at a time. Advertising
        # false keeps replay prefetch from sending a 16-item serial request that
        # exceeds the online client's receive timeout.
        "supports_batch": False,
        "z_dim": contract.z_dim,
        "chunk_len": contract.chunk_len,
        "action_dim": contract.action_dim,
        "proprio_dim": contract.proprio_dim,
        "reference_action_space": "processor_decoded_physical",
        # Only advertise a rotation contract that the backend recognized from
        # the checkpoint modalities. Machine B rejects None for Nero runs.
        "rot6d_convention": backend.rot6d_convention,
        "num_inference_timesteps": backend.num_inference_timesteps,
        "stateless_reference": True,
        "rtc_applied": False,
        "token_scope": backend.token_scope,
        "token_sampling": backend.token_sampling,
        "max_vl_tokens": backend.max_vl_tokens,
        "action_layout": backend.action_layout,
        "action_layout_hash": backend.action_layout_hash,
        "proprio_layout": backend.proprio_layout,
        "proprio_layout_hash": backend.proprio_layout_hash,
    }
    LOGGER.info("Serving GR00T RLT features on ws://%s:%d", args.host, args.port)
    WebsocketFeatureServer(
        MachineAFeaturePolicy(backend, contract, supports_batch=False),
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
