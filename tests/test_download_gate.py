"""Tests for the dataset download approval gate.

The gate is a safety control: these tests ensure no code path can download
anything while DATASET_APPROVED is false.
"""

from __future__ import annotations

import pytest

from roadvision.config import Config, dataset_approved, load_config
from roadvision.data.download import DownloadBlocked, check_approval, main


def _cfg(approved: bool) -> Config:
    return Config(
        {
            "dataset": {"approved": approved, "root": "data/raw"},
        }
    )


def test_gate_closed_by_default(monkeypatch):
    monkeypatch.delenv("DATASET_APPROVED", raising=False)
    cfg = load_config()  # shipped config has approved: false
    assert dataset_approved(cfg) is False


def test_gate_requires_both_signals(monkeypatch):
    # env true, config false -> blocked
    monkeypatch.setenv("DATASET_APPROVED", "true")
    assert dataset_approved(_cfg(False)) is False
    # env false, config true -> blocked
    monkeypatch.setenv("DATASET_APPROVED", "false")
    assert dataset_approved(_cfg(True)) is False
    # both true -> open
    monkeypatch.setenv("DATASET_APPROVED", "true")
    assert dataset_approved(_cfg(True)) is True


def test_gate_env_var_variants(monkeypatch):
    for value, expected in [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("maybe", False),
    ]:
        monkeypatch.setenv("DATASET_APPROVED", value)
        assert dataset_approved(_cfg(True)) is expected, value


def test_check_approval_raises_when_closed(monkeypatch):
    monkeypatch.delenv("DATASET_APPROVED", raising=False)
    with pytest.raises(DownloadBlocked, match="BLOCKED"):
        check_approval(_cfg(False))


def test_cli_returns_exit_code_2_when_blocked(monkeypatch, caplog):
    import logging

    monkeypatch.delenv("DATASET_APPROVED", raising=False)
    with caplog.at_level(logging.ERROR, logger="roadvision.data.download"):
        code = main([])  # no countries needed; must refuse before any network I/O
    assert code == 2
    assert "BLOCKED" in caplog.text


def test_cli_never_touches_network_when_blocked(monkeypatch):
    """If the gate is closed, urllib.request.urlopen must never be called."""
    monkeypatch.delenv("DATASET_APPROVED", raising=False)

    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("network access attempted while gate closed")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    code = main(["--countries", "Czech"])
    assert code == 2
