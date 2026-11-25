"""The one fixture for a cluster as RSPt's field table: a plain dict that ``ClusterCall.from_fields`` accepts.

``call_from_arrays`` builds a :class:`ClusterCall` directly from solver-side arrays, for tests that drive
``solve.solve`` without a field table.
"""

import numpy as np

from impurityModel_interface.cluster import ClusterCall, Outputs, Shells

N = 4


def cluster(n=N, extra=None, n_full=None):
    n_full = n_full or n
    rng = np.random.default_rng(3)
    c = lambda *shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)  # noqa: E731
    fields = {
        "label": "Ni-3d",
        "solver_paramstr": "8 10",
        "dc_paramstr": "gap 0.5",
        "ham": c(n, n),
        "harm2corr": c(n_full, n),
        "corr2cf": c(n, n),
        "sig": np.zeros((n, n, 3), complex),
        "sig_real": np.zeros((n, n, 5), complex),
        "sig_static": np.zeros((n, n), complex),
        "sig_dc": np.zeros((n, n), complex),
        "ctx%u4": np.zeros((n,) * 4, complex),
        "ctx%hyb": np.zeros((n, n, 5), complex),
        "ctx%wmsl": np.arange(3.0),
        "ctx%emesh": np.arange(5.0),
        "ctx%green_mu": 0.25,
        "ctx%eim": 0.1,
        "ctx%ttt": 0.01,
        "ctx%verbosity": 1,
        "ctx%dc_flag": 1,
    }
    fields.update(extra or {})
    return fields


def call_from_arrays(
    *,
    label,
    solver_line,
    dc_line="",
    task="selfenergy",
    h_dft,
    u4,
    hyb,
    iw,
    w,
    sig,
    sig_real,
    sig_static,
    sig_dc,
    corr_to_spherical_in,
    n_rot_cols=None,
    eim,
    tau,
    verbosity=0,
    operators=None,
):
    """A :class:`ClusterCall` from the arrays the solver side sees (Fortran-ordered, ``hyb`` frequency last)."""
    n_orb = h_dft.shape[0]
    n_orb_full = corr_to_spherical_in.shape[1]
    shells = Shells.from_layout(n_orb, n_orb if n_rot_cols is None else n_rot_cols, n_orb_full)
    return ClusterCall(
        label=label,
        solver_line=solver_line,
        dc_line=dc_line,
        task=task,
        iw=iw,
        w=w,
        eim=eim,
        tau=tau,
        verbosity=verbosity,
        h_dft=h_dft,
        u4=u4,
        hyb=hyb,
        corr_to_spherical=shells.corr_to_spherical(corr_to_spherical_in, n_orb),
        shells=shells,
        out=Outputs(sig, sig_real, sig_static, sig_dc),
        operators=operators,
    )
