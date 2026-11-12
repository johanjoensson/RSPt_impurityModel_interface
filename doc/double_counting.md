# The double-counting line

RSPt passes the criterion for the double counting in a fixed 100-character line (`rspt_dc_line`),
parsed by `impurityModel_interface.dc_line.parse_dc_line`. The line can be extended but not
restructured, because RSPt rewrites it for DC in {-4,-5,-14,-15}.

| Line | Criterion |
|---|---|
| `<peak_position>` | Place a spectral peak at that energy: `E[N+1]-E[N]` if positive, `E[N]-E[N-1]` if negative. |
| `gap [offset]` | Centre the impurity gap, i.e. put the midpoint `(E[N+1]-E[N-1])/2` of the removal and addition excitations at `offset` (default 0, the Fermi level). **Recommended for insulators:** Karolak et al. (arXiv:1004.4569) show the occupation condition "essentially breaks down" for a charge-transfer insulator, because inside a gap the occupation is flat and a whole interval of mu satisfies it. For NiO they get 25.3 eV this way against 20.4 (SC)AMF; 21 eV is qualitatively wrong. |
| `occ [N]` | Fix the thermal impurity occupation (Karolak's Eq. 2; the right criterion for metals). |
| `fll`, `amf`, `sigma_inf` | Static schemes evaluated at the DFT reference occupation/density matrix; no ED solve. |
| `nominal` | FLL at the *nominal* (integer) occupation: needs no reference filling, so it cannot saturate. The natural guess for CSC iteration 1, or a reference for checking a converged `occ` answer. |

## Modifiers (anywhere on the line)

* `alpha X` (default 0.5) — RSPt applies the returned DC verbatim, with no mixing of its own. An
  undamped full Newton step on a problem whose target (the charge density) moves each CSC iteration,
  combined with a search that can return a point at a charge-sector boundary, is a d8/d9
  limit-cycle generator. The wrapper damps against **its own** previous answer stored in the archive
  (`DC damped`): `DC_new = DC_prev + alpha (DC_converged - DC_prev)`, and `DC_new = DC_converged`
  when there is none. It does not damp against the `sig_dc` RSPt hands in: RSPt rebuilds that from
  its own FLL/AMF potential on every call and never stores our answer, so damping toward it returned
  a DC that does not satisfy the criterion just converged.
* `de2_min X` — the Epstein-Nesbet PT2 admission floor of the charge-sector CIPSI expansions
  (`gap`, `peak`). Unset, the solver's `GS_DE2_MIN` (1e-8) buys *parity* with the self-energy run,
  not accuracy: on `nio_5peeled` 1e-6 -> 1e-8 moves the gap centre 0.3 meV (about 2.7 meV in `mu`
  after the 1/|chi| amplification) at 4x the cost, while a cap change from 2,000 to 8,000 moves it
  about 54 meV. The search itself is only resolved to about 5e-2, so loosening this is a *bounded*
  approximation (the skipped weight is reported as `subthreshold_de2_mass`); lowering the
  determinant cap is not — it truncates the basis and can make the self-energy non-causal.
* `e_pt2 X` (or `e_pt2_tol X`) — the residual PT2 energy the `gap`/`peak` sector solves converge
  to: the *summed* PT2 contribution of every determinant left out, which is what the energy error
  follows (impurityModel `doc/plans/cipsi_pt2_convergence.md`). This, not `de2_min`, is the
  accuracy control. Unset, the sector solves inherit the solver line's `e_pt2`.

## What is refused, and why

* `de2_min` with a static scheme: no ground-state solve, nothing to narrow.
* `e_pt2` with `occ`: `occ` solves on the production ground-state path and converges to the solver
  line's `e_pt2`; narrowing it there would change the criterion rather than its cost.
* `e_pt2` with a static scheme: no solve.

A modifier that reports as set and does nothing is the failure mode these refusals prevent.

## Failure behaviour

A search that cannot reach its target (`DoubleCountingUnreachable`) or fails in the solver leaves
`sig_dc` as RSPt sent it, prints a banner, records `DC search unreachable` / `DC search failed` on the
iteration's archive group and returns 0, so a multi-hour CSC run carries on with the previous value.
The search raises rank-synchronously, so every rank reaches the handler together.
