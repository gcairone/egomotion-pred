import pytest

from egopred.manifest import Manifest


def test_states_info_merge_and_persistence(tmp_path):
    m = Manifest(tmp_path / "m.sqlite")
    assert m.state("frames", "a") is None
    m.set("frames", "a", "pending", src="u")
    m.set("frames", "a", "done", frames=3)
    m.set("frames", "b", "verified")
    assert m.get("frames", "a") == ("done", {"src": "u", "frames": 3})
    assert m.keys("frames") == ["a", "b"]
    assert m.keys("frames", "verified") == ["b"]
    m.close()
    m2 = Manifest(tmp_path / "m.sqlite")
    assert m2.summary() == {"frames": {"done": 1, "verified": 1}}
    with pytest.raises(ValueError):
        m2.set("frames", "a", "bogus")


def test_merge_keeps_newest(tmp_path):
    a = Manifest(tmp_path / "a.sqlite")
    b = Manifest(tmp_path / "b.sqlite")
    a.set("frames", "x", "pending")
    b.set("frames", "x", "verified", frames=3)  # newer
    b.set("frames", "y", "done")
    a.set("meta", "z", "verified")  # only local
    b.close()
    assert a.merge(tmp_path / "b.sqlite") == 2
    assert a.get("frames", "x") == ("verified", {"frames": 3})
    assert a.state("frames", "y") == "done" and a.state("meta", "z") == "verified"
    a.set("frames", "y", "verified")  # local newer than other: kept
    a.merge(tmp_path / "b.sqlite")
    assert a.state("frames", "y") == "verified"
