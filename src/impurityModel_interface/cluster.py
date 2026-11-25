"""One RSPt call as a single object: what RSPt asked for, the data it sent, and the arrays it reads back.

:class:`ClusterCall` is built from the field mapping of :mod:`rspt_fields` (any ``Mapping``, so a test
can feed it a plain dict) and is the only thing :func:`solve.solve` takes.  Everything RSPt used to
compute on the Fortran side lives here: the chemical potential shift of ``h_dft``, the
harmonic-to-correlated rotation, the dipole from RSPt's observables, the stacked symmetry
representations, and the orbital layout (:class:`Shells`) from ``offset`` / ``preoffset``.

The arrays are RSPt's own memory (see :mod:`entry`): ``u4``, ``hyb`` and the four members of
:class:`Outputs` are views, everything else is a copy RSPt never reads back.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from impurityModel.api import LocalOperators

#: What a call can ask for.  ``"spectra"`` is only reachable once RSPt sends ``ctx%task``.
TASKS = ("dc", "selfenergy", "spectra")


@dataclass(frozen=True)
class Shells:
    """The orbital layout of the correlated sets, from RSPt's ``offset`` and ``preoffset``.

    Attributes
    ----------
    nspmat : int
        Spin blocks RSPt dimensions its rotation matrices with (1: only the down block is sent).
    preoffset : tuple of int
        Boundaries of the spherical column blocks, one per correlated set, as RSPt sent them.
    offset : tuple of int
        Boundaries of the correlated row segments; each is ``[dn k | up k]`` with ``k`` up to ``2l+1``
        (a set need not be a full shell).
    shell_ls : tuple of int
        ``l`` of each spherical block.  Repeated values are legitimate: sets of one ``l`` are never merged.
    """

    nspmat: int
    preoffset: tuple
    offset: tuple
    shell_ls: tuple

    @classmethod
    def from_layout(cls, n_orb, n_rot_cols, n_orb_full, *, nspmat=None, preoffset=None, offset=None):
        """Validate the layout RSPt sent.  ``None`` means one set covering everything (the single-shell case)."""
        inferred = cls._infer_nspmat(n_orb, n_rot_cols, n_orb_full)
        if nspmat is None:
            nspmat = inferred
        elif nspmat != inferred:
            raise RuntimeError(f"RSPt sent nspmat={nspmat}, but the shapes (n_orb={n_orb}, n_rot_cols={n_rot_cols}) say {inferred}.")
        pre = (0, n_orb_full) if preoffset is None else tuple(int(x) for x in preoffset)
        off = (0, n_orb) if offset is None else tuple(int(x) for x in offset)
        cls._check_boundaries("preoffset", pre, n_orb_full)
        cls._check_boundaries("offset", off, n_orb)
        shell_ls = []
        for a in range(len(pre) - 1):
            width = pre[a + 1] - pre[a]
            if width % nspmat or (width // nspmat) % 2 == 0:
                if preoffset is not None:
                    raise RuntimeError(
                        f"preoffset block {a} is {width} columns wide; with nspmat={nspmat} that is not a shell of 2l+1 orbitals."
                    )
                # No layout was sent: the single block is whatever the call carries, not necessarily a shell.
                shell_ls = []
                break
            shell_ls.append((width // nspmat - 1) // 2)
        for s in range(len(off) - 1 if offset is not None else 0):
            if (off[s + 1] - off[s]) % 2:
                raise RuntimeError(f"offset segment {s} is {off[s + 1] - off[s]} spin-orbitals wide; expected an even number.")
        return cls(nspmat=nspmat, preoffset=pre, offset=off, shell_ls=tuple(shell_ls))

    @staticmethod
    def _infer_nspmat(n_orb, n_rot_cols, n_orb_full):
        if n_rot_cols == n_orb:
            return 2
        if 2 * n_rot_cols == n_orb:
            return 1
        raise RuntimeError(
            f"Inconsistent rotation shapes from RSPt: n_orb={n_orb}, "
            f"n_rot_cols={n_rot_cols}, n_orb_full={n_orb_full}; "
            "expected n_rot_cols == n_orb or 2*n_rot_cols == n_orb."
        )

    @staticmethod
    def _check_boundaries(name, bounds, total):
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != total or any(b <= a for a, b in zip(bounds, bounds[1:])):
            raise RuntimeError(f"RSPt's {name} {list(bounds)} is not a strictly increasing partition of [0, {total}].")

    def corr_to_spherical(self, corr_to_spherical_in, n_orb):
        """``corr_to_spherical`` with both spin blocks present, ``(n_orb, nspmat-free 2 * n_orb_full)``.

        RSPt dimensions the rotation with ``nspmat`` spins (1 for non-spin-polarized calculations
        without SOC), while the Hamiltonian, hybridization and self-energies always carry both.  With
        ``nspmat == 2`` the matrix is passed through.  With ``nspmat == 1`` only the down rows of each
        ``offset`` segment are populated; they are copied onto the up rows, and each spherical block
        ``a`` gets its columns ``[dn m | up m]``.  The correlated rows are ordered per segment
        ``[dn k | up k]``, and a segment may hold fewer than ``2l+1`` orbitals per spin.

        ``corr_to_spherical_in`` is ``(n_orb, n_orb_full)``.  RSPt's ``corr2cf`` is not used.
        """
        if self.nspmat == 2:
            return np.array(corr_to_spherical_in)
        n_full = self.preoffset[-1]
        out = np.zeros((n_orb, 2 * n_full), dtype=complex)
        for s in range(len(self.offset) - 1):
            start, half = self.offset[s], (self.offset[s + 1] - self.offset[s]) // 2
            down, up = slice(start, start + half), slice(start + half, start + 2 * half)
            for a in range(len(self.preoffset) - 1):
                p0, p1 = self.preoffset[a], self.preoffset[a + 1]
                width = p1 - p0
                out[down, 2 * p0 : 2 * p0 + width] = corr_to_spherical_in[down, p0:p1]
                out[up, 2 * p0 + width : 2 * p0 + 2 * width] = corr_to_spherical_in[down, p0:p1]
        return out


@dataclass
class Outputs:
    """The four arrays RSPt reads its result from, all written by in-place assignment (never copies)."""

    sig: np.ndarray
    sig_real: np.ndarray
    sig_static: np.ndarray
    sig_dc: np.ndarray

    @property
    def sig_python(self):
        """``sig`` with the frequency first, as the solver has it (a view)."""
        return np.moveaxis(self.sig, -1, 0)

    @property
    def sig_real_python(self):
        return np.moveaxis(self.sig_real, -1, 0)

    def write_selfenergy(self, sigma_static, sigma, sigma_real, comm):
        """Fill the self-energy from the solver's results (RSPt's basis) on the root and broadcast it."""
        if comm.rank == 0:
            self.sig_static[:, :] = sigma_static
            self.sig_python[:, :, :] = sigma
            self.sig_real_python[:, :, :] = sigma_real
        comm.Bcast(self.sig_static, root=0)
        comm.Bcast(self.sig_real, root=0)
        comm.Bcast(self.sig, root=0)

    def write_double_counting(self, dc):
        self.sig_dc[:, :] = dc

    def poison(self):
        """Mark the self-energy as failed: infinity in the imaginary part at the last Matsubara frequency."""
        self.sig[:, :, -1] += 1j * np.inf


@dataclass
class ClusterCall:
    """Everything one call to ``run_impmod_cluster`` carries, in the names the solver side uses."""

    label: str
    solver_line: str
    dc_line: str
    #: One of :data:`TASKS`.
    task: str
    iw: np.ndarray
    w: np.ndarray
    eim: float
    tau: float
    verbosity: int
    #: ``ham - mu``, a copy.
    h_dft: np.ndarray
    #: RSPt's views: broadcast through on the non-root ranks.
    u4: np.ndarray
    hyb: np.ndarray
    #: ``(n_orb, 2 n_orb_full)``, both spins, columns ``[dn | up]`` per spherical block.
    corr_to_spherical: np.ndarray
    shells: Shells
    out: Outputs
    operators: Optional[LocalOperators] = None
    #: A malformed operator array, reported on every rank by :meth:`synchronise`.
    operator_error: Optional[str] = field(default=None, repr=False)

    @property
    def n_orb(self):
        return self.h_dft.shape[0]

    @classmethod
    def from_fields(cls, fields):
        """Derive the call from RSPt's cluster fields (a mapping, see :class:`rspt_fields.RsptFields`).

        ``h_dft`` is a copy: RSPt's ``ham`` is never written.  Every other array is the view RSPt owns.
        The output shapes are checked rather than reshaped: a mismatch would write past RSPt's allocation.
        """
        ham = fields["ham"]
        n_orb = int(ham.shape[0])
        if ham.shape != (n_orb, n_orb):
            raise ValueError(f"RSPt's ham has shape {ham.shape}; expected a square matrix.")
        iw = fields["ctx%wmsl"]
        w = fields["ctx%emesh"]
        expected = {
            "sig": (n_orb, n_orb, len(iw)),
            "sig_real": (n_orb, n_orb, len(w)),
            "sig_static": (n_orb, n_orb),
            "sig_dc": (n_orb, n_orb),
            "ctx%u4": (n_orb,) * 4,
            "ctx%hyb": (n_orb, n_orb, len(w)),
        }
        for name, shape in expected.items():
            if fields[name].shape != shape:
                raise ValueError(f"RSPt field {name!r} has shape {fields[name].shape}, expected {shape}.")

        h_dft = np.array(ham, order="F")
        h_dft[np.diag_indices(n_orb)] -= fields["ctx%green_mu"]

        task = fields.get("ctx%task")
        task = ("dc" if int(fields["ctx%dc_flag"]) == 1 else "selfenergy") if task is None else str(task).strip().lower()

        harm2corr = fields["harm2corr"]
        shells = Shells.from_layout(
            n_orb,
            int(fields["corr2cf"].shape[1]),
            int(harm2corr.shape[0]),
            nspmat=None if fields.get("ctx%nspmat") is None else int(fields["ctx%nspmat"]),
            preoffset=fields.get("preoffset"),
            offset=fields.get("offset"),
        )
        # RSPt's harm2corr is (n_orb_full, n_orb); the solver wants corr -> harmonic.
        corr_to_spherical = shells.corr_to_spherical(np.asfortranarray(harm2corr.conj().T), n_orb)

        try:
            operators, operator_error = _operators(fields, n_orb), None
        except ValueError as error:
            operators, operator_error = None, str(error)

        return cls(
            label=fields["label"],
            solver_line=fields["solver_paramstr"],
            dc_line=fields["dc_paramstr"],
            task=task,
            iw=iw,
            w=w,
            eim=fields["ctx%eim"],
            tau=fields["ctx%ttt"],
            verbosity=int(fields["ctx%verbosity"]),
            h_dft=h_dft,
            u4=fields["ctx%u4"],
            hyb=fields["ctx%hyb"],
            corr_to_spherical=corr_to_spherical,
            shells=shells,
            out=Outputs(fields["sig"], fields["sig_real"], fields["sig_static"], fields["sig_dc"]),
            operators=operators,
            operator_error=operator_error,
        )

    def synchronise(self, comm):
        """Make every rank hold rank 0's data, writing through RSPt's views where they are RSPt's.

        The operators are built on every rank (cheap) but rank 0's answer, or its failure, is the one all
        ranks use: a malformed input must fail everywhere rather than leave the other ranks waiting.
        """
        comm.Bcast(self.corr_to_spherical)
        comm.Bcast(self.h_dft)
        comm.Bcast(self.u4)
        comm.Bcast(self.out.sig_dc)
        comm.Bcast(self.hyb)
        self.verbosity = comm.bcast(self.verbosity)
        self.operators, self.operator_error = comm.bcast((self.operators, self.operator_error))
        if self.operator_error is not None:
            raise ValueError(self.operator_error)


# --- RSPt's observables as LocalOperators ---------------------------------------------------


_OPERATOR_NAMES = {"ms": "sz", "msp": "s_plus", "ml": "lz", "mlp": "l_plus", "msaxis": "s_axis"}


def _square(fields, name, n_orb):
    """``fields[name]`` cut to the first ``n_orb`` rows and columns, as the Fortran side did."""
    value = np.asarray(fields[name])
    if value.ndim != 2 or value.shape[0] < n_orb or value.shape[1] < n_orb:
        raise ValueError(f"RSPt field {name!r} has shape {value.shape}, smaller than the {n_orb}x{n_orb} cluster.")
    return value[:n_orb, :n_orb]


def _operator(fields, name, n_orb):
    """A one-body operator in RSPt's correlated basis, or ``None`` if absent (not allocated, or all zero)."""
    if name not in fields:
        return None
    value = np.array(_square(fields, name, n_orb), dtype=complex)
    return value if np.any(value) else None


def _dipole(fields, n_orb):
    """Cartesian dipole ``(3, n_orb, n_orb)`` from RSPt's observables, or ``None``.

    The observables line lists Dxyz and/or Dpm. Dxyz is used as is; Dpm (Dp, Dm, Dz) is converted
    back: ``Dx = (Dp + Dm)/sqrt(2)``, ``Dy = (Dp - Dm)/(i sqrt(2))``.
    """
    names = []
    while f"observables%obs({len(names) + 1})%name" in fields:
        names.append(fields[f"observables%obs({len(names) + 1})%name"].strip())

    def block(first):
        keys = [f"observables%obs({first + i})%t" for i in range(3)]
        if first + 2 > len(names) or not all(key in fields for key in keys):
            return None
        return [np.array(_square(fields, key, n_orb), dtype=complex) for key in keys]

    for k, name in enumerate(names, start=1):
        if name == "(Dipole) Dx" and (parts := block(k)) is not None:
            return np.stack(parts, axis=0)
    for k, name in enumerate(names, start=1):
        if name == "(Dipole) Dp" and (parts := block(k)) is not None:
            dp, dm, dz = parts
            half = np.sqrt(0.5)
            return np.stack([half * (dp + dm), -1j * half * (dp - dm), dz], axis=0)
    return None


def _symmetries(fields, n_orb):
    """``((rep, antiunitary), ...)`` over the cluster's point-group operations, or ``None``.

    A cluster where any operation has no representation counts as having none.
    """
    n_symm = int(fields.get("symm%n", 0))
    keys = [f"symm({i})%rep" for i in range(1, n_symm + 1)]
    if n_symm == 0 or not all(key in fields for key in keys):
        return None
    return tuple(
        (np.array(_square(fields, key, n_orb), dtype=complex), bool(fields.get(f"symm({i})%timerev", False)))
        for i, key in enumerate(keys, start=1)
    )


def _operators(fields, n_orb):
    """RSPt's one-body operators as a :class:`LocalOperators` in its correlated basis, or ``None``."""
    kwargs = {}
    for name, target in _OPERATOR_NAMES.items():
        value = _operator(fields, f"observables%{name}", n_orb)
        if value is not None:
            kwargs[target] = value
    dipole = _dipole(fields, n_orb)
    if dipole is not None:
        kwargs["dipole"] = dipole
    symmetry_ops = _symmetries(fields, n_orb)
    if symmetry_ops is not None:
        kwargs["symmetry_ops"] = symmetry_ops
    if not kwargs:
        return None
    return LocalOperators(source="rspt", frame="global", **kwargs)
