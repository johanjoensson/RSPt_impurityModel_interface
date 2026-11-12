"""RSPt's 100-character double-counting line.

RSPt rewrites that line for DC in {-4,-5,-14,-15}, so the grammar can only be extended, never
restructured.  What each criterion means, and why the modifiers exist, is in ``doc/double_counting.md``.
"""

from dataclasses import dataclass
from typing import Optional

STATIC_SCHEMES = frozenset({"fll", "amf", "sigma_inf", "sigmainf", "nominal"})


@dataclass(frozen=True)
class DCRequest:
    """A parsed double-counting line."""

    #: ``gap``, ``occupation``, ``peak`` or one of :data:`STATIC_SCHEMES`.
    mode: str
    #: The criterion's number (gap offset, occupation, peak position); ``None`` where it takes none.
    target: Optional[float]
    #: Damping against the previous iteration's answer.
    alpha: float = 0.5
    #: PT2 admission floor of the charge-sector solves (``None``: the solver's default).
    de2_min: Optional[float] = None
    #: Residual PT2 energy the charge-sector solves converge to (``None``: the solver line's).
    e_pt2_tol: Optional[float] = None


def _take(words, names, convert):
    """Remove ``name value`` (any of ``names``) from ``words`` and return the converted value, or ``None``."""
    for i, word in enumerate(words):
        if word.lower() in names:
            if i + 1 >= len(words):
                raise ValueError(f"'{word}' on the double-counting line needs a value")
            value = convert(word, words[i + 1])
            del words[i : i + 2]
            return value
    return None


def _number(word, text):
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"'{word}' on the double-counting line needs a number, got {text!r}") from None


def _positive(word, text):
    value = _number(word, text)
    if not value > 0:
        raise ValueError(f"'{word}' must be positive")
    return value


def parse_dc_line(dc_line):
    """Parse the double-counting line into a :class:`DCRequest`.

    Raises
    ------
    ValueError
        An unknown or malformed criterion, or a modifier the criterion cannot honour.
    """
    words = dc_line.split("!")[0].split("#")[0].split()
    alpha = _take(words, {"alpha"}, _number)
    de2_min = _take(words, {"de2_min"}, _positive)
    e_pt2_tol = _take(words, {"e_pt2", "e_pt2_tol"}, _positive)
    head = words[0].lower() if words else None
    if head == "gap":
        if len(words) > 2:
            raise ValueError(
                "impurityModel gap double counting takes at most 1 argument, the offset of the "
                f"gap centre from the Fermi level. Got: {words}"
            )
        mode, target = "gap", (float(words[1]) if len(words) == 2 else 0.0)
    elif head in {"occ", "occupation"}:
        if len(words) > 2:
            raise ValueError(
                "impurityModel occupation double counting takes at most 1 argument, "
                f"the target occupation. Got: {words}"
            )
        mode, target = "occupation", (float(words[1]) if len(words) == 2 else None)
    elif head in STATIC_SCHEMES:
        if len(words) != 1:
            raise ValueError(f"impurityModel static double counting ({head}) takes no further arguments. Got: {words}")
        mode, target = head, None
    else:
        if len(words) != 1:
            raise ValueError(
                "impurityModel double counting takes 1 argument, peak_position, 'gap [offset]', "
                f"'occ [target impurity occupation]', or one of fll/amf/sigma_inf/nominal. Got: {words}"
            )
        mode, target = "peak", float(words[0])
    if de2_min is not None and mode in STATIC_SCHEMES:
        raise ValueError(
            f"'de2_min' sets the PT2 admission floor of the charge-sector solves, and '{mode}' runs none. "
            "Remove it, or pick a criterion that solves ('gap', 'peak' or 'occ')."
        )
    if e_pt2_tol is not None and mode not in {"gap", "peak"}:
        raise ValueError(
            "'e_pt2' on the double-counting line sets the convergence of the 'gap' and 'peak' charge-sector solves; "
            + (
                "'occ' converges to the solver line's 'e_pt2', so set it there."
                if mode == "occupation"
                else f"the static scheme '{mode}' runs no solve."
            )
        )
    return DCRequest(mode, target, 0.5 if alpha is None else alpha, de2_min, e_pt2_tol)
