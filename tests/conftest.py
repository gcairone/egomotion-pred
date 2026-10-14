import importlib

import pytest


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Point every data/output root at a temporary directory."""
    from egopred import paths

    monkeypatch.setenv("EGOPRED_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EGOPRED_OUT", str(tmp_path / "out"))
    importlib.reload(paths)
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(paths)
