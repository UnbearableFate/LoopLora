from looplora.infra.seeds import derive_seed, stable_hash


def test_derive_seed_is_deterministic_and_label_sensitive():
    assert derive_seed(17, "train", 3) == derive_seed(17, "train", 3)
    assert len({derive_seed(17, "train", i) for i in range(1000)}) == 1000
    assert derive_seed(17, "train", 3) != derive_seed(17, "val", 3)
    assert derive_seed(17, "train", 3) != derive_seed(29, "train", 3)
    assert 0 <= derive_seed(1, "x") < 2**63


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
