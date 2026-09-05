from f1_predictor.cache import DataCache


def test_set_get_and_expiry(tmp_path):
    cache = DataCache(str(tmp_path))
    cache.set("k", {"a": 1}, ttl=3600)
    assert cache.get("k") == {"a": 1}
    assert cache.is_valid("k")
    cache.set("k", {"a": 2}, ttl=-1)
    assert cache.get("k") is None
    assert cache.get("k", ignore_ttl=True) == {"a": 2}
    assert not cache.is_valid("k")


def test_clear_with_prefix(tmp_path):
    cache = DataCache(str(tmp_path))
    cache.set("season_results_2025", [1], 10)
    cache.set("season_results_2024", [1], 10)
    cache.set("schedule_2025", [1], 10)
    assert cache.clear(prefix="season_results_") == 2
    assert cache.get("schedule_2025") == [1]
    assert cache.clear() == 1


def test_corrupt_file_is_removed(tmp_path):
    cache = DataCache(str(tmp_path))
    (tmp_path / "bad.json").write_text("{not json")
    assert cache.get("bad") is None
    assert not (tmp_path / "bad.json").exists()
