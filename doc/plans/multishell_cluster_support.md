# Multi-Shell RSPt Cluster Support Across the RSPt ED Interface, `impmod_interface`, `rspt2spectra` and `impurityModel`

## 0. Design decisions (settled 2026-10-25)

- **D1** `rot_to_spherical` stays a single 2D array `(n_imp, Σ_a 2(2l_a+1))`; a new `shell_ls` tuple
  carries the shell structure. A dict of per-shell embeddings cannot survive `prepare_solver_basis`
  (orbitals are regrouped by block structure into new integer keys), see §2.
- **D2** The Fortran side additionally passes `nspmat, n_shells, preoffset, n_offset, offset`;
  `reconstruct_rotations` becomes segment-aware. `Cf` with several shells is accepted.
- **D3** The core-shell plumbing needed for an XAS cluster in the Σ loop is in scope: per-shell
  nominal occupation (a bath-less core shell is an ordinary shell the ground-state solver optimises, **not frozen**:
  review J1, 2026-11-02), non-hybridizing shells excluded
  from the bath pool, valence-only DC shifts, `fll/amf/nominal` refused for several shells.
- **D4** Scope is the **self-energy path only**. A separate C-visible entry point that runs the XAS
  machinery (not the Σ machinery) comes later (follow-up F1).
- **D5** RSPt delivers the complete multi-shell `u4`, inter-shell Slater integrals included. The
  interface consumes it as-is and does no U-matrix construction or override.
- **D6** Correlated sets need not be full shells (`t2g`/`eg`/`Cf`-truncated). Nothing may infer `l`
  or the spin split from a row count. Sets with the same `l` are **never merged**: RSPt separates
  them for a reason.

### 1. Context and scope

> Written 2026-10-25 from a three-repo audit; supersedes the earlier draft. Line numbers are as of
> RSPt `feature/cmake-build` c5f5b7dc, impmod_interface `development` 1dd4dd0, impurityModel
> `master` e858ef8d.

An RSPt `cluster` block may carry several correlated orbital sets (`ncorr > 1`): e.g. a Ni 2p
semicore set plus the 3d set for XAS, or 4f + 5d on a rare-earth site. RSPt already builds
`ham`, `Δ(ω)` and the 4-index `u4` over all correlated sets (inter-shell F^k/G^k included), but
the external ED hook refuses `ncorr ≠ 1`, and the wrapper/solver silently mis-handle the
orbital layout when it gets through (which it does today under `Cf`, see §2).

**Scope:** the self-energy (DMFT) path through `run_impmod_ed` only. The XAS/RIXS machinery will get
its own C-visible entry point later (follow-up F1) and is not touched here. The U matrix is taken
from RSPt unchanged: it is already complete for multi-shell clusters.

**Hard requirement (user):** a correlated set need not be a full shell. `t2g`-only or `eg`-only
sets of a d shell, or a `Cf`-truncated set, MUST be handled correctly everywhere. In this
design a *shell* is identified by its **spherical column block** (RSPt always sends the full
`nspmat·(2l+1)` spherical functions per set, in `preoffset`), while the **correlated rows** that
carry weight in that block may be any subset, `n_a ≤ 2(2l_a+1)`. Nothing may infer `l` or the
spin split from a row count. Per-shell observables that are well defined on a subset
(occupation, `ρ_sph,a`, `S_z`, `S_+` spin pairs, nominal/frozen occupation, DC projector) work on
`n_a`; Casimirs (`L²`, `S²`, `J²`, term symbol) are evaluated only for shells with
`R_a†R_a = I` and skipped with a clear log line otherwise.

### 2. Verified facts and corrections to the draft

RSPt (`/home/johan/Programming/rspt/rsptDir/src`):
- `green_solver.F90:1411-1413` stops on `acluster%ncorr .ne. 1`; `:1418-1421` stops on
  `usize .ne. hsize`. `usize = offset(size(slater,2)+1)` (all correlated spin-orbitals, `:1345-1349`),
  `hsize = size(alocal,1)` (`:1350`). **`Cf` already defeats both guards**: `locust_symmetrize_proj`
  collapses `offset` to `[0, msize]`, sets `ncorr = 1`, rebuilds `qn/uavg/javg` as 1×1 and resets
  `corr2cf` to identity (`green_locust.F90:4724-4752, 4783-4786`); `preoffset` is never touched.
- `green_impmod_interface.F90` (151 lines) passes `conjg(transpose(harm2corr))` `(n_orb, n_orb_full)`,
  `corr2cf` `(n_orb, n_rot_cols)`, `n_orb = size(ham,1)` (always 2 spins, all sets),
  `n_rot_cols = msize·nspmat/2`, `n_orb_full = preoffset(end) = Σ_a nspmat·(2l_a+1)`. It does **not**
  pass `nspmat`, `preoffset`, `offset`, `ncorr`, `l`, or Slater integrals (`slater(4)` at `:47` is dead).
  `solver_param(100)` (`:37`) vs `solver_paramstr` len **120** (`green_config.F90:205`) → truncation.
- `harm2corr(premsize, msize)`: rows spherical (per `preoffset` segment: nspmat=2 → `[dn m=-l..l | up m=-l..l]`,
  nspmat=1 → `[m=-l..l]`), columns correlated (per `offset` segment `[dn k_s | up k_s]`, `k_s ≤ 2l+1`;
  for nspmat=1 only the dn half of each segment is populated, `green_trunk_interface.F90:2373-2375`).
  Block placed at `harm2corr(preoffset(n)+1:preoffset(n+1), offset(n)+1:offset(n+1))`
  (`green_locust.F90:1037-1056`). With `Cf`: `harm2corr ← harm2corr·alocal`, columns become globally
  spin-packed `[all dn cf | all up cf]` (`:4642, 4673`).
- `u4`: `locust_u4 → harmonic_u4_intra` (`green_harmonics.f90:464-559`) **includes inter-shell**
  direct F^k and exchange G^k terms (inter-shell values from radial integrals,
  `green_trunk_interface.F90:1584ff`). `u4` is `usize⁴` in the correlated basis, declared `n_orb⁴` at
  the C boundary — equal because of the `usize == hsize` guard. The interface uses it as-is (D5).
- No core flag, no per-shell SOC: SOC is inside `acluster%ham` when `fulrel` (⇒ nspmat = 2). A 2p set
  must exist as a (semicore) energy set in the LMTO basis (`lda_id2exist`, `green_init.F90:1853`);
  correlated lines must come first (`:1849`). `Δ(ω)` is computed for the whole block, core rows included.
- Internal HIA/ED pass `offset(1:ncorr+1)` and per-set `qn`; RSPt's own DC uses `uavg(s,t)` per set
  pair (`green_double_counting.F90:183-460`).

impmod_interface (`src/impurityModel_interface/lib.py`):
- `reconstruct_rotations` `:557-597`; `_run_impmod_ed` `:947-1073` (Fortran-order `np.ndarray` views
  over `ffi.buffer`, not `np.frombuffer`); `run_impmod_ed(*args)` `:1573-1603` needs **no** change;
  `_solve` `:1076-1532` (reconstruct + Bcast `:1109-1119`, N0 bounds `:1160`, sector seeding
  `:1172-1180` via `_split_total_over_groups` `:737-760`, shape heuristic `:1193-1198`,
  `rot_to_spherical` `:1241`, `from_blocks` `:1242-1250`, archive `:1257-1292`, DC dispatch
  `:1317-1372`, Σ write-back `:1478-1488`); `parse_solver_line` `:106-426`
  (`BasisOptions(nominal_occ={0: N0})` `:397-407`); `get_ed_h0` `:1633-1778`.
- **The draft's nspmat=1 bug is real but mis-described**: `reconstruct_rotations` splits rows at a
  single global `n_half = n_orb//2`, wrong whenever rows are per-segment `[p_dn, p_up, d_dn, d_up]`.
- **The draft's "MPI buffer bug" does not exist**: all ranks run `reconstruct_rotations` with identical
  scalars before `Bcast`. Dropped.
- Hyb fit: `rspt2spectra.block_structure.build_block_structure(G, mat)` partitions by connectivity of
  `Δ ∪ H_local` at 1e-6; `rspt2spectra/hyb_fit.py:get_state_per_inequivalent_block` `:508-600`:
  `pool = B·n_blocks` (`:576`) counts zero-weight blocks (inflates valence baths), `hybridizing = weight > 0`
  (`:587`) is a strict zero test (noise-level core Δ would force ≥ n_orb states). Empty blocks are
  tolerated downstream (`:291-292`, `flatten_star_levels`, `build_H_bath_v`).
- `interface_test.py` is already broken (stale `calc_selfenergy` signature); not a place for new tests.
- Tests live in `src/impurityModel_interface/test/` (no `tests/`). Positional callers of
  `_run_impmod_ed`: `test_marshalling_aliases_rspt_memory.py:71-101`, `test_environment_file.py:95`,
  `test_entry_point_returns_failure.py`; `test_solve_is_pure_python.py:58-72` calls `_solve` directly.

impurityModel (`src/impurityModel/ed`):
- Convention in code: `O_comp = R O_sph R†`, `ρ_sph = R† ρ_comp R`; `R` rows = computational
  (impurity) orbitals, columns = spherical.
- `prepare_solver_basis` `solver_basis.py:134-283`: flattens one level (`:163`), rotates
  `R ← u_imp† R` (`:188`, ndarray only), **regroups impurity orbitals by block structure into new
  integer keys** (`:220-229`) — a dict keyed by shell can never match downstream. ⇒ D1.
- `spin_pairs.py`: `_pairs_from_rotated_splus` `:206-236` (infers l from count),
  `_impurity_pairs_per_partition` `:239-256`, `_impurity_pairs_whole_shell` `:259-272` (returns None
  for dict). There is **no guard at `:309`** (draft error).
- `observables.py`: `impurity_shell_rhos` `:542-630` (array path `:595-597` treats the whole block as
  one shell), `make_impurity_casimir_operators` `:860-949` (`:906-912` raises unless count = 2(2l+1);
  a 6-orbital t2g set passes as "l=1" and then fails on a `(6×10)@(6×6)` matmul),
  `_single_particle_lsj_matrices` `:386-410`, `get_occupations_from_rho_spherical` `:200-209`.
- `gs_report.py:183-248` catches `ValueError` only (a dict would `KeyError` at `observables.py:914`).
- `model.py`: `from_blocks` `:823-871` → `from_solver_matrix` `:764-821` (`{0: range(n_imp)}`);
  `from_shells` `:503-724` (nested `{l: [[..]]}`, dict rotation keyed by l, `u4=None`) is the spectra
  path; `_read_archive_group` `:1168-1253` (`"Rot to spherical"` `:1199`, `"nominal occupation"` `:1240`);
  `impurity_indices` `:197-199` flattens one level (`doc/known_issues.md`).
- `dc_static.py`: `_model_uj` `:63-89` raises only for a dict; `fll/nominal` `:92-153, 191-223` scalar×I
  over all impurity orbitals with total N; `amf` `:155-188` uniform (bypasses `_model_uj`);
  `sigma_inf_dc` `:226+` uses `get_Sigma_static(u4, ρ)` — the only shell-agnostic mode.
  **Draft claim "other modes work natively" is false.**
- `dc_criteria.py`: `fixed_peak/gap/occupation` shift `dc_guess + μ·I` uniformly (`:479-483`), N and
  bound over the whole impurity (`:1934, 1992, 2229, 2236-2237`); `frozen_occupations` for bath-less
  groups at `:822` and `:1940` — **not** in `selfenergy.py` (`basis_information` `:570-588`), so
  DC↔GS parity breaks for a bath-less core. `groundstate.py` threads `frozen_occupations` through
  `sector_key` `:54-77`, the walk `:614-900`, HF seed `:456-499`. `basis_restrictions.get_effective_restrictions`
  `:197-251` windows the whole impurity.
- Two-shell fixture exists: `test/symmetry/test_symmetry_observables.py:1249-1262` (dict form, core first).
  Tests live in `src/impurityModel/test/`, not `/tests`.

### 3. Shell model (new, central)

- `shell_ls: tuple[int, ...]` — the `l` of each consecutive spherical column block of
  `rot_to_spherical`, in RSPt `preoffset` order. Column block `a` has width `2(2l_a+1)` after
  reconstruction (both spins). `R_a = R[:, cols_a]`, shape `(n_imp, 2(2l_a+1))`.
- `impurity_shell_index[o] = argmax_a Σ_{c∈cols_a} |R[o,c]|²`, computed on the solver-basis `R`
  (`solver_basis.py`), raising if the max weight `< 1 − 1e-6` (an orbital straddling shells).
  `n_a = #{o : shell_index[o] = a}`; `complete_a = (n_a == 2(2l_a+1)) and allclose(R_a†R_a, I)`.
- Partial sets are the normal case for `t2g`/`eg`/`Cf`-truncated clusters: `R_a` is rectangular with
  `R_a†R_a` a rank-`n_a` projector. Per-shell occupation is `Tr(R_a†ρR_a)`, spin pairs come from
  `S_+ = Σ_a R_a S_+(l_a) R_a†` (well defined on a subset), DC/N0/frozen logic uses `n_a`.
- Two sets with the same `l` (e.g. `t2g` and `eg` as separate correlated lines) arrive as two column
  blocks with equal `l`; each is treated as its own (partial) shell, exactly as RSPt sent them.
  **The interface never merges same-`l` sets** (user decision, 2026-10-25): RSPt separates them for a
  reason. Consequence: per-set occupation, `ρ_sph`, `S_z` and spin pairs are reported per set; Casimirs
  are skipped for both (each is partial). `shell_ls` may therefore contain repeated values.
- Single-shell backward compatibility: `shell_ls=None` ⇒ one shell inferred from the column count;
  every new keyword defaults to `None` and reproduces today's behaviour bit-for-bit.
- Why not a dict or an array attribute: dict keys cannot survive the block-structure regrouping;
  `np.asarray(..., dtype=complex)` at `solver_basis.py:188`, `observables.py:915`, `spin_pairs.py:221`
  and the HDF5 round trip would strip an ndarray subclass attribute.

### 4. Orbital-ordering trace (corrected)

Shell `a`: `m_a = 2l_a+1`, `preoffset` width `w_a = nspmat·m_a`. `offset` segment `s`: width `2k_s`,
rows `[dn k_s | up k_s]`, `k_s ≤ m_s` (t2g: `k_s = 3`). `n_orb = offset(end)`,
`n_orb_full = preoffset(end)`, `n_rot_cols = n_orb·nspmat/2`.

1. `corr_to_spherical_in` `(n_orb, n_orb_full)`: rows per `offset` segment (nspmat=1: only dn rows
   non-zero), columns per `preoffset` segment (nspmat=2 `[dn|up]`, nspmat=1 `[m]`). Block-diagonal over
   (segment s, shell s) without `Cf`/`Irr`.
2. `corr_to_cf_in` `(n_orb, n_rot_cols)`: rows as above; columns globally spin-packed
   `[all dn | all up]` (nspmat=1 keeps dn half only). Example 2p+3d nspmat=1, identity CF:
   `[0:3,0:3]=I_p`, `[6:11,3:8]=I_d`.
3. Output after reconstruction: `corr_to_cf (n_orb,n_orb)`; `corr_to_spherical (n_orb, 2·n_orb_full)`
   (nspmat=1) or `(n_orb, n_orb_full)` (nspmat=2), columns per shell `[shell_a dn | shell_a up]`;
   `rot_to_spherical = corr_to_cf† @ corr_to_spherical`; `shell_ls = ((w_a/nspmat − 1)//2)_a`.
4. `Cf` case: single `offset` segment `(0, n_half)`, `corr2cf = I`, rows are CF orbitals globally
   spin-packed — the segment algorithm is exactly right for it. Accept `Cf` with `n_shells > 1`;
   shell purity is checked in impurityModel (§5 G4/T4.4) since `symm_irrdecompose` can mix p and d
   functions sharing an irrep in non-centrosymmetric site groups.

### 5. Tasks

Dependencies: G1 → G2; G4 independent (must land before G2's `_solve` is exercised end-to-end);
G3 independent; G5 last.

#### G1 RSPt — only the ED interface files

- **T1.1** `green_solver.F90:1411-1413`: delete the `ncorr .ne. 1` stop; keep `:1418-1421`.
- **T1.2** `green_impmod_interface.F90`: interface block `:8-51` gains
  `integer(c_int), value :: nspmat, n_shells, n_offset` and
  `integer(c_int) :: preoffset(n_shells+1), offset(n_offset+1)` after `n_orb_full`;
  `solver_param(100)` → `(120)`; drop dead `slater(4)`. Body: `c_preoffset = int(acluster%preoffset, c_int)`,
  `c_offset = int(acluster%offset, c_int)`; pass `int(dmft%nspmat, c_int)`, sizes−1, arrays in the call
  `:117-142`. (`use green_config` already present at `:57`.)

#### G2 impmod_interface

- **T2.1 Boundary** — `src/run_impurityModel.h` (`run_impmod_ed_py`) and `library_builder.py`
  (`run_impmod_ed` + forward): insert `int nspmat, int n_shells, int *preoffset, int n_offset, int *offset`
  after `size_t n_orb_full`. `_run_impmod_ed`: same slot; decode with
  `np.frombuffer(ffi.buffer(ptr, (n+1)*ffi.sizeof("int")), dtype=np.intc).copy()`; forward
  `nspmat=, preoffset=, offset=`. `ffi.string(..., 100)` at `:991` → 120. Update the positional
  callers listed in §2 (and the `char[100]` buffer in `test_marshalling_aliases_rspt_memory.py:60`).
- **T2.2 `reconstruct_rotations`** `:557-597` — signature
  `(corr_to_spherical_in, corr_to_cf_in, n_orb, n_rot_cols, n_orb_full, nspmat=None, preoffset=None, offset=None)`,
  same 2-tuple return. Defaults: `nspmat` from `n_rot_cols` vs `n_orb` (kept as consistency check when
  given), `preoffset=[0,n_orb_full]`, `offset=[0,n_orb]`. Validate: both start at 0, end at
  `n_orb_full`/`n_orb`, strictly increasing; `w_a % nspmat == 0` and `w_a/nspmat` odd; every `offset`
  segment even (no requirement that `2k_s == 2m_s` — partial sets allowed). nspmat=2: pass-through.
  nspmat=1:
  ```python
  n_half = n_orb // 2
  cf  = zeros((n_orb, n_orb)); sph = zeros((n_orb, 2*n_orb_full))
  for s: start, half = offset[s], (offset[s+1]-offset[s])//2
      dn, up = slice(start, start+half), slice(start+half, start+2*half)
      cf[dn, :n_half] = cf_in[dn, :];  cf[up, n_half:] = cf_in[dn, :]
      for a: p0, p1 = preoffset[a], preoffset[a+1]; m = p1-p0
          sph[dn, 2*p0:2*p0+m]     = sph_in[dn, p0:p1]
          sph[up, 2*p0+m:2*p0+2*m] = sph_in[dn, p0:p1]
  ```
  Copy dn rows only (independent of what RSPt leaves in up rows). Single shell reproduces `:586-591`.
  New helper `spherical_shell_ls(preoffset, nspmat) -> tuple[int, ...]`.
- **T2.3 `_solve`** `:1076-1532` — add `nspmat=None, preoffset=None, offset=None` (defaults keep
  `test_solve_is_pure_python.py` working). **Broadcast `(preoffset, offset, nspmat)` from rank 0 before
  validation** (a rank-local assert would hang the others at the next `Bcast`). Compute `shell_ls`.
  N0 handling: `parse_solver_line` returns `{0: N0}` or a per-shell tuple; bounds: total ≤ `n_orb`,
  tuple length = `len(shell_ls)`, `N0_a ≤ n_a` (**`n_a` from the row→shell map, not `2(2l_a+1)`**;
  in the wrapper the map is `shell_index_of_orbitals(rot_to_spherical, shell_ls)`). Sector seeding
  `:1172-1180`: `_split_total_over_groups(total, nominal_occ, movable=...)` where `movable` = shells with
  hybridization weight above `HYB_WEIGHT_REL_TOL` (helper `shell_hybridization_weights(hyb_cf, w, R, shell_ls)`;
  move the block below the CF rotation at `:1127-1129`, it only mutates `basis`, first used `:1211`).
  Replace the square/non-square heuristic `:1193-1198` with a per-shell report
  (`shell a (l=..): n_a of 2(2l+1) spin-orbitals, complete|partial`). `from_blocks(..., shell_ls=shell_ls)`.
  Archive: keep `"nominal occupation"` = total (reader compat `model.py:1240`), add datasets
  `"Spherical shell l"` (intc) and, for the tuple form, `"nominal occupation per shell"`.
  DC dispatch `:1364-1372`: raise a named `RuntimeError` for `fll/amf/nominal` when `len(shell_ls) > 1`;
  `_parse_dc_line` docstring: `occupation N` targets the hybridizing (valence) shells only.
- **T2.4 `parse_solver_line`** `:106-218, 397-407` — `N0` token accepts `6,8` → `nominal_occ=(6, 8)`
  (tuple; never confusable with a group dict); single int unchanged. `format_settings_header` /
  `solver_line_attrs` `:429-554` print the tuple as `6,8`. Docstring `:111`.
- **T2.5** Quarantine/delete `interface_test.py` (already broken).

#### G3 rspt2spectra (`src/rspt2spectra/hyb_fit.py:508-600`)

- **T3.1** Module constant `HYB_WEIGHT_REL_TOL = 1e-3` (exported, imported by lib.py so fitter and
  wrapper agree on which shells are "core"); parameter `rel_tol` on `get_state_per_inequivalent_block`
  and `fit_hyb` (`:123-141`, threaded `:268-276`). `hybridizing = weight > rel_tol·weight.max()`;
  `pool = B·hybridizing.sum()` (replaces `:576`); proportional split over hybridizing blocks only;
  coverage `:587-591` unchanged; log the excluded blocks. Rationale: a 2p semicore at −60 Ry has
  noise-level but non-zero Δ in a ±1 Ry window; `> 0` would force ≥ 6 states on it and steal from valence.

#### G4 impurityModel

- **T4.1 `model.py`** — `ImpurityModel` (`:186-194`): field `shell_ls: Optional[tuple] = None`,
  property `resolved_shell_ls` via `observables.resolve_shell_ls` (None for the dict/spectra form).
  `from_solver_matrix`/`from_blocks`: keyword `shell_ls=None`. `_read_archive_group`: read
  `"Spherical shell l"` (after `:1199`) and `"nominal occupation per shell"` (after `:1240`) when present.
  `BasisOptions.nominal_occ` docstring `:1370-1372`: dict by group | scalar total | per-shell tuple.
- **T4.2 `observables.py`** — helpers `resolve_shell_ls(R, shell_ls=None)` (infer single shell from
  column count; validate `Σ 2(2l+1) == n_cols`), `shell_column_slices(shell_ls)`,
  `shell_index_of_orbitals(R, shell_ls, tol=1e-6)`, `shell_is_complete(R_a)`.
  `impurity_shell_rhos(..., shell_ls=None)`: ndarray path — one shell ⇒ today's `(None, None, R†ρR)`;
  several ⇒ yield `(l_a, a, R_a†ρR_a)` per shell (works for partial `R_a`; `ρ_sph,a` is `2(2l_a+1)`-square,
  rank ≤ `n_a`). Dict path unchanged (spectra contract). `compute_shell_observables` `:633-700` forwards.
  `make_impurity_casimir_operators(..., shell_ls=None)`: ndarray with several shells **or a single
  partial shell** ⇒ ignore partition keys, work on `all_orbs`, per shell `complete_a` → build
  `R_a M R_a†` on `all_orbs` into `per_shell_ops[a]`; totals only if all shells complete, else
  `(None, None, None, per_shell_ops)` when `per_shell`, `ValueError` otherwise. Fixes today's t2g
  accident at `:906-912`. `_single_particle_lsj_matrices`/`get_occupations_from_rho_spherical` unchanged
  (they now only ever see full-width per-shell matrices).
- **T4.3 `spin_pairs.py`** — factor `_pairs_from_splus_matrix(orbs, sp_comp, tol)` out of `:226-236`;
  `_impurity_pairs_whole_shell(..., shell_ls=None)`: `S_+ = Σ_a R_a S_+(l_a) R_a†` (`n_imp×n_imp`),
  valid for partial shells (`S_+` maps `m↓ → m↑` inside the subset). `derive_spin_pairs`/`resolve_spin_pairs`
  get `shell_ls=None`; with several shells skip the per-partition attempt.
- **T4.4 `solver_basis.py`** — `SolverBasis` gains `shell_ls`, `impurity_shell_index`, and properties
  `shell_has_bath`, `active_impurity_indices`
  (orbitals of shells with bath; all orbitals for one shell). `prepare_solver_basis(..., shell_ls=None)`:
  resolve on the input `R`; `:188` unchanged; after grouping compute `impurity_shell_index` on the
  solver-basis `R` and **assert every derived group is shell-pure**. If `u_imp` would mix shells, fall
  back to the input basis (`rotate = False`) rather than fail; raise only if the input basis itself
  mixes shells (message names the groups and the 1e-6 coupling). `_per_group_occupation(..., shell_of_group=None)`
  `:65-101`: tuple branch — per shell `a`, energetic filling restricted to groups of shell `a` with
  total `N0_a`; raise if `N0_a > n_a`. Callers pass `shell_ls=model.shell_ls`: `selfenergy.py:536-538`,
  `dc_criteria.py:753-762` and `~:1924`, `susceptibility.py:441-443`.
- **T4.5 Threading `shell_ls` to reporters** — `calc_gs` (`groundstate.py:1066-1075`) keyword →
  `report_ground_state` (`gs_report.py:60-72`) → `make_impurity_casimir_operators` (`:184-186, 198-200`),
  `resolve_spin_pairs` (`:274`), `compute_state_summary` (`:421`), `compute_magnetic_summary` (`:436`),
  `print_thermal_expectation_values` (`:514`), `print_expectation_values` (`:532`);
  `selfenergy.py:589-596` passes `sb.shell_ls`. Restructure `gs_report.py:183-248` so per-shell
  Casimirs run whenever `per_shell_ops` is non-empty (today nested under `l_ops is not None`), and
  print "skipped (partial shell, n_a of 2(2l+1))" for incomplete shells. `susceptibility.py`
  `build_susceptibility_operators(..., shell_ls=None)` `:95-127`, pass at `:469/:483`.
- **T4.6 (removed 2026-11-02, review J1/§6.2)** No frozen core: `frozen_occupations` is for shells without orbital
  energies only (XAS core), never Σ/DC/multishell. The bath-less p shell is held at 6 by its energy (2p at −60 Ry) and
  checked by the Tr ρ_p = 6 test, not pinned.
- **T4.7 Double counting** — `dc_static.py`: `_model_uj` raises when `len(resolved_shell_ls) > 1`
  (keep dict check); add the same check to `amf_dc` and the explicit-`(u,j)` branch of `_fll` `:121-134`.
  Rationale: a scalar FLL/AMF over p+d applies the core U to a full shell and drops `U_pd N_d`
  cross terms; the correct per-shell mean field *is* `Σ_static(u4, ρ)` = `sigma_inf_dc` — document it
  as the multi-shell default. `dc_criteria.py`: replace `identity` in `_SectorContext` (`:406`) and the
  occupation context (`:1856-1857`) by projectors `shift_input` (onto `sb.active_impurity_indices`, input
  basis) and `shift_solver = u_imp† P u_imp`; use at `:479, :483, :2015, :2349`, built at `:809, :1991`;
  `total_impurity_orbitals` (`:1934, 1992, 2229, 2236-2237`) → `len(active_impurity_indices)`;
  `state_occupations` `:2070` and the DFT reference `dc_reference._noninteracting_impurity_occupation:114`
  over the same indices. Identity for a single shell ⇒ numerics unchanged.

#### G5 End-to-end

- **T5.1** Extend `/home/johan/Programming/impmod_tests/NiO/ED/green.inp`: add a Ni 2p line (own
  `Ewin`, semicore energy set must exist in the basis), solver line `6,8 10 4 …`, DC `sigma_inf` first.
  Expect: log shows `preoffset=[0,6,16]` (fulrel) or `[0,3,8]`, `offset=[0,6,16]`; group legend shows a
  6-orbital group with no bath (Tr ρ_p = 6 holds by energy, not by a pin); per-shell Casimirs for l=1 and l=2; returned `sig_dc`
  differs from RSPt's only on the 3d block in `occupation` mode.
- **T5.2** Partial-set regression: an existing t2g-only cluster (`basis = 1`) must produce identical
  Σ/DC to before and a "partial shell, 6 of 10" line instead of today's silent Casimir failure.

### 6. Test plan (concrete fixtures)

impmod_interface `test/test_reconstruct_rotations.py`:
1. 2p+3d nspmat=2 (`[0,6,16]/[0,6,16]`): pass-through; `shell_ls == (1,2)`.
2. 2p+3d nspmat=1 (`preoffset=[0,3,8]`, `offset=[0,6,16]`, `n_orb=16, n_rot_cols=8, n_orb_full=8`):
   block placement as in §4, `cf` unitary, `R_a†R_b = δ_ab I`, `Σ_a R_a R_a† = I`.
3. 4f+5d nspmat=1 (`[0,7,12]/[0,14,24]`) and nspmat=2 (`[0,14,24]/[0,14,24]`).
4. `Cf` collapsed nspmat=1 (`preoffset=[0,3,8]`, `offset=[0,16]`, rows 0:8 populated, `cf_in = I[:, :8]`).
5. **Partial sets**: t2g-only d + full p, nspmat=1 (`n_orb=12, n_rot_cols=6, n_orb_full=8`,
   `preoffset=[0,3,8]`, `offset=[0,6,12]`): `sph.shape == (12,16)`, `R_p†R_p = I_6`, `R_d†R_d` rank-6
   projector; eg-only variant; t2g and eg as two sets (`preoffset=[0,5,10]`, `offset=[0,6,10]`) →
   `shell_ls == (2,2)`, both partial, **kept as two shells** (no merge), `R_a†R_b = 0` for `a ≠ b`.
6. Invalid offsets raise; existing single-shell tests untouched.

`test_parse_solver_line.py`: `"6,8 10 2"` → `(6, 8)`; `"8 10 2"` → `{0: 8}`; `"6,,8"`, `"6,-1"` raise.
`test_solve_is_pure_python.py`: defaults; per-shell out-of-bounds against `n_a` (t2g: `N0_d = 7` raises);
`_split_total_over_groups(15, {0:6,1:8}, movable={1}) == {0:6,1:9}`.
`test_marshalling_aliases_rspt_memory.py`: new positionals, `np.intc` copies, 120-byte solver line.
Archive: `"Spherical shell l"` / `"nominal occupation per shell"` round trip via `from_hdf5`.

rspt2spectra `tests/test_state_distribution.py` (new): p-block at 1e-9 of max weight → 0 states,
`pool == B·2`; `rel_tol=0` restores old behaviour; `fit_hyb` yields empty `ebs_star[p]`.

impurityModel `src/impurityModel/test/`:
- `symmetry/test_symmetry_observables.py`: ndarray twin of `_two_shell_fixture` (`R=eye(16)`,
  `shell_ls=(1,2)`, groups `{0:[0..5],1:[6..9],2:[10..15]}`) gives the same shell rhos as the dict
  fixture; same after a block-diagonal random `u_imp`; per-shell Casimirs for `2p⁶3d¹`;
  **partial**: `R` `(12,16)` (t2g+p) → totals `None`, per-shell ops only for p, rhos for both.
- `test_spin_pairs_multishell.py` (new): pairs `(0,3),(1,4),(2,5),(6,11),…` for `eye(16)`; rotated;
  t2g-only `(6,10)` single shell → 3 pairs (today returns None).
- `basis/test_solver_basis_multishell.py` (new): 2p at −60 Ry + cubic 3d near 0, baths on d only,
  `nominal_occ=(6,8)` → shell-pure groups, `{p:6, eg:2, t2g:6}`, `shell_has_bath == (False, True)`, Tr ρ_p = 6 in the ground state; 1e-3 p–d
  element → falls back to input basis / raises as specified; t2g-only single set `nominal_occ=(5,)`.
- `test_dc_multishell.py` (new): `fll/amf/nominal` raise; `sigma_inf_dc` `(16,16)`;
  `fixed_occupation_dc` on p(6)+d(2)+2 baths changes only the d block; `calc_selfenergy` keeps
  `Tr ρ_p = 6` in every thermal state.
- `model`: `from_blocks(shell_ls=…)`, `resolved_shell_ls` inference, archive round trip.

### 7. Verification

1. `PYTHONPATH=src pytest` in impmod_interface (see memory: else a stale site-packages copy is tested);
   `pytest` in impurityModel and rspt2spectra; rebuild CFFI lib and RSPt.
2. Single-shell regression on `impmod_tests/NiO/ED` and `SMO`: archives identical apart from the two new
   datasets; DC and Σ bit-identical (projector = I).
3. T5.1 multi-shell run; T5.2 partial-set run.

### 8. Follow-ups and open questions

- **F1** Separate C-visible entry point (e.g. `run_impmod_xas`) in `run_impurityModel.h` /
  `library_builder.py` / `lib.py` that runs the XAS machinery (ground state + `spectra.simulate_spectra`)
  on the same marshalled `h_dft/Δ/u4/rotations` without the Σ machinery. Needs: a `from_shells`-style
  nested `impurity_orbitals` keyed by shell built from `R` + `shell_ls` + `impurity_shell_index`
  (bridge between the Σ-path data model and the spectra contract), `core_l`/`valence_l` from `shell_ls`,
  and `frozen_occupations={core}`. Everything in this plan (shell model, `shell_ls`, per-shell N0,
  the shell-role data) is a prerequisite for it; the XAS entry freezes only a core the user declares (review O5). Not required now.
- **F2** Spectra model from a multi-shell *archive* (`"Rot to spherical"` + `"Spherical shell l"`),
  the offline counterpart of F1.
- **F3** `impurity_indices` nesting bug (`doc/known_issues.md`).
- **Q1** `HYB_WEIGHT_REL_TOL = 1e-3` must be measured on the NiO 2p case.
- **Q2** *Answered 2026-11-02:* moot; nothing is frozen in the Σ/DC path (T4.6 removed).
- **Q3** `Cf` in non-centrosymmetric groups: does `symm_irrdecompose` keep shell purity? The purity
  check makes it a clear error; fix would be a shell-blocked decomposition in RSPt.
- **Q4** Semicore `Ewin`/energy-set requirements for the NiO 2p extension need confirming.
