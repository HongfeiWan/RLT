"""Unified, GR00T-first command line entry point for ``groot-rlt``."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class _Command:
    module: str
    help: str


_COMMANDS = {
    "train-token": _Command(
        "groot_rlt.representation.train_vl_embedding_autoencoder",
        "train the GR00T RL-token representation",
    ),
    "evaluate-token": _Command(
        "groot_rlt.representation.evaluate_vl_embedding_autoencoder_ablation",
        "evaluate a trained RL-token representation",
    ),
    "precompute": _Command(
        "groot_rlt.representation.precompute_rl_tokens_and_vla_actions",
        "precompute RL tokens and GR00T VLA actions",
    ),
    "visualize-token": _Command(
        "groot_rlt.representation.visualize_rl_token_umap",
        "visualize the learned RL-token space",
    ),
    "serve-features": _Command(
        "groot_rlt.serving.groot_feature_server",
        "serve online GR00T features and reference actions",
    ),
    "export-online-stats": _Command(
        "groot_rlt.integration.online_stats",
        "export action statistics for online RLT",
    ),
}


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groot-rlt",
        description="GR00T-first entry point for the RLT training lifecycle.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="lifecycle commands",
    )
    for name, command in _COMMANDS.items():
        # The legacy command owns its complete parser.  These parsers exist only
        # to produce the unified lifecycle overview and must not consume child
        # arguments (especially child ``--help``).
        subparsers.add_parser(name, add_help=False, help=command.help)
    return parser


def _load_main(module_name: str) -> Callable[[], int | None]:
    command_main = getattr(importlib.import_module(module_name), "main")
    if not callable(command_main):
        raise TypeError(f"{module_name}.main is not callable")
    return command_main


def main(argv: Sequence[str] | None = None) -> int:
    """Route a lifecycle subcommand to its existing entry point.

    Only the selected command module is imported.  Its arguments are forwarded
    unchanged through ``sys.argv`` because the legacy entry points intentionally
    retain ownership of their argument parsers.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    parser = _make_parser()

    if not args or args[0] in {"-h", "--help"}:
        print(parser.format_help(), end="")
        return 0

    command_name, child_args = args[0], args[1:]
    command = _COMMANDS.get(command_name)
    if command is None:
        parser.error(f"unknown command: {command_name!r}")

    command_main = _load_main(command.module)
    original_argv = sys.argv
    try:
        sys.argv = [f"groot-rlt {command_name}", *child_args]
        result = command_main()
    finally:
        sys.argv = original_argv

    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
