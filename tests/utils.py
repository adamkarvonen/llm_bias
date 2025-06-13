import numbers

import pytest


def assert_dict_approx_equal(d1, d2, rel=None, abs=None):
    """Asserts that two dictionaries are approximately equal.

    Compares dictionaries recursively, using pytest.approx for numeric values
    and direct equality for other types.
    """
    assert d1.keys() == d2.keys(), f"Keys mismatch: {d1.keys()} != {d2.keys()}"

    for key in d1:
        v1 = d1[key]
        v2 = d2[key]

        if isinstance(v1, dict) and isinstance(v2, dict):
            assert_dict_approx_equal(v1, v2, rel=rel, abs=abs)
        elif isinstance(v1, numbers.Number) and isinstance(v2, numbers.Number):
            assert v1 == pytest.approx(v2, rel=rel, abs=abs), (
                f"Value mismatch for key '{key}': {v1} != approx({v2})"
            )
        else:
            assert v1 == v2, f"Value mismatch for key '{key}': {v1} != {v2}"
