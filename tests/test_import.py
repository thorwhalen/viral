"""Importing viral must not need its data directory (viral#3)."""

import importlib

import pytest


def test_import_without_a_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("viral_sources_dir", str(tmp_path / "does-not-exist"))
    import viral.base as base

    importlib.reload(base)
    with pytest.raises(NotADirectoryError, match="viral_sources_dir"):
        base.get_data_sources()
