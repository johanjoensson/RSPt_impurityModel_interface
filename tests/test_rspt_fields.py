"""``RsptFields``: RSPt's field table as a name -> numpy view mapping."""

import numpy as np
import pytest

pytest.importorskip("cffi")
from rspt_table import build_table, make_ffi  # noqa: E402

from impurityModel_interface.rspt_fields import RsptFields  # noqa: E402


@pytest.fixture
def ffi():
    return make_ffi()


def fields_of(ffi, arrays, logical=()):
    table, n, keep = build_table(ffi, arrays, logical=logical)
    return RsptFields(ffi, table, n), keep


def test_arrays_keep_dtype_shape_and_fortran_order(ffi):
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    c = np.arange(6) * (1 + 2j)
    i = np.arange(5, dtype=np.int32)
    fields, _ = fields_of(ffi, {"a": a, "c": c.reshape(2, 3), "i": i})
    np.testing.assert_array_equal(fields["a"], a)
    assert fields["a"].flags.f_contiguous and fields["a"].dtype == np.float64
    assert fields["c"].dtype == np.complex128 and fields["c"].shape == (2, 3)
    assert fields["i"].dtype == np.int32


def test_views_alias_rspt_memory_and_are_read_only_unless_whitelisted(ffi):
    fields, keep = fields_of(ffi, {"ham": np.zeros((2, 2), complex), "sig": np.zeros((2, 2), complex)})
    assert not fields["ham"].flags.writeable
    fields["sig"][0, 0] = 5
    assert np.frombuffer(bytes(ffi.buffer(keep["buffers"]["sig"])), complex)[0] == 5


def test_scalars_and_strings(ffi):
    fields, _ = fields_of(
        ffi,
        {"x": np.array(2.5), "n": np.array(7, dtype=np.int32), "z": np.array(1 + 2j), "s": "Ni-3d" + " " * 13},
    )
    assert fields["x"] == 2.5 and isinstance(fields["x"], float)
    assert fields["n"] == 7 and isinstance(fields["n"], int)
    assert fields["z"] == 1 + 2j
    assert fields["s"] == "Ni-3d"


@pytest.mark.parametrize("truth", [1, -1])  # gfortran and ifx encode .true. differently
def test_logicals_are_truth_not_a_particular_integer(ffi, truth):
    table, n, keep = build_table(ffi, {"flag": np.array(0, dtype=np.int32), "mask": np.zeros(3, dtype=np.int32)}, logical=("flag", "mask"))
    keep["buffers"]["flag"][0:4] = np.int32(truth).tobytes()
    keep["buffers"]["mask"][4:8] = np.int32(truth).tobytes()
    fields = RsptFields(ffi, table, n)
    assert fields["flag"] is True
    np.testing.assert_array_equal(fields["mask"], [False, True, False])


def test_an_unallocated_field_is_absent_and_the_error_says_what_to_do(ffi):
    fields, _ = fields_of(ffi, {"ham": np.eye(2, dtype=complex), "rho": None})
    assert "ham" in fields and "rho" not in fields
    assert fields.get("rho") is None
    assert sorted(fields) == ["ham"]
    with pytest.raises(KeyError, match="rebuild RSPt"):
        fields["rho"]


def test_every_field_can_be_listed_for_the_archive(ffi):
    fields, _ = fields_of(ffi, {"b": np.ones(2), "a": np.ones(3)})
    assert {k: v.shape for k, v in fields.items()} == {"a": (3,), "b": (2,)}
