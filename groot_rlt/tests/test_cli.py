from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from groot_rlt import cli

COMMAND_MODULES = {
    "train-token": "groot_rlt.representation.train_vl_embedding_autoencoder",
    "evaluate-token": ("groot_rlt.representation.evaluate_vl_embedding_autoencoder_ablation"),
    "precompute": "groot_rlt.representation.precompute_rl_tokens_and_vla_actions",
    "visualize-token": "groot_rlt.representation.visualize_rl_token_umap",
    "serve-features": "groot_rlt.serving.groot_feature_server",
    "export-online-stats": "groot_rlt.integration.online_stats",
}


@pytest.mark.parametrize("argv", [[], ["--help"], ["-h"]])
def test_lifecycle_help_is_dependency_free(monkeypatch, capsys, argv):
    def fail_import(_module_name):
        raise AssertionError("top-level help must not load a command module")

    monkeypatch.setattr(cli.importlib, "import_module", fail_import)

    assert cli.main(argv) == 0

    output = capsys.readouterr().out
    assert "lifecycle commands" in output
    for command in COMMAND_MODULES:
        assert command in output


@pytest.mark.parametrize(("command", "module_name"), COMMAND_MODULES.items())
def test_routes_each_command_lazily_and_forwards_arguments(monkeypatch, command, module_name):
    original_argv = ["pytest", "unchanged"]
    monkeypatch.setattr(sys, "argv", original_argv)
    imports = []
    received_argv = []

    def legacy_main():
        received_argv.append(list(sys.argv))
        return 7

    def fake_import(requested_module):
        imports.append(requested_module)
        return SimpleNamespace(main=legacy_main)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    child_args = ["--help", "--model-path", "path with spaces", "--flag=value"]

    assert cli.main([command, *child_args]) == 7
    assert imports == [module_name]
    assert received_argv == [[f"groot-rlt {command}", *child_args]]
    assert sys.argv is original_argv


def test_restores_sys_argv_when_child_exits(monkeypatch):
    original_argv = ["groot-rlt", "before"]
    monkeypatch.setattr(sys, "argv", original_argv)

    def legacy_main():
        raise SystemExit(0)

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _module_name: SimpleNamespace(main=legacy_main),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["train-token", "--help"])

    assert exc_info.value.code == 0
    assert sys.argv is original_argv


def test_unknown_command_does_not_import(monkeypatch, capsys):
    def fail_import(_module_name):
        raise AssertionError("an unknown command must not load a command module")

    monkeypatch.setattr(cli.importlib, "import_module", fail_import)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["not-a-command"])

    assert exc_info.value.code == 2
    assert "unknown command" in capsys.readouterr().err
