# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.offline_bridge_training import OfflineBridgeTrainConfig
from rlt_online_rl.offline_bridge_training import run_offline_bridge_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate a LeRobot-v3 bridge pickle, build C=10/stride=2 replay, "
            "and run a bounded offline actor/critic smoke train."
        )
    )
    parser.add_argument("--bridge-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rl-config",
        type=Path,
        default=None,
        help="Optional production online_rl YAML. Without it, a small non-production network is used.",
    )
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-fps", type=float, default=10.0)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--metrics-interval", type=int, default=1)
    parser.add_argument("--actor-hidden-dim", type=int, default=64)
    parser.add_argument("--actor-num-layers", type=int, default=1)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-num-layers", type=int, default=1)
    parser.add_argument("--bc-weight", type=float, default=None)
    parser.add_argument("--q-weight", type=float, default=None)
    parser.add_argument("--delta-weight", type=float, default=None)
    parser.add_argument(
        "--features-are-real",
        action="store_true",
        help="Explicitly assert that z_rl came from the approved real feature provider.",
    )
    parser.add_argument(
        "--expected-feature-contract-fingerprint",
        default=None,
        help="Required with --features-are-real and checked against the bridge manifest.",
    )
    parser.add_argument(
        "--production-run",
        action="store_true",
        help="Mark artifacts production only with real verified features and an explicit --rl-config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rl_config = None
    system_config = None
    if args.rl_config is not None:
        system_config = load_system_config_yaml(str(args.rl_config))
        rl_config = system_config.rl
    config = OfflineBridgeTrainConfig(
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        expected_fps=args.expected_fps,
        replay_capacity=args.replay_capacity,
        metrics_interval=args.metrics_interval,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        bc_weight=args.bc_weight,
        q_weight=args.q_weight,
        delta_weight=args.delta_weight,
        features_are_real=args.features_are_real,
        expected_feature_contract_fingerprint=args.expected_feature_contract_fingerprint,
        production_run=args.production_run,
    )
    summary = run_offline_bridge_training(
        args.bridge_pickle,
        args.output_dir,
        config=config,
        rl_config=rl_config,
        system_config=system_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
