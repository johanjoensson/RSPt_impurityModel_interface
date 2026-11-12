"""Rank-0-decides: run something on rank 0, give every rank the answer or the failure."""


def on_root(comm, function):
    """Return ``function()`` evaluated on rank 0 and broadcast to every rank.

    Ranks must agree on anything that steers a collective call (a block partition, a stored fit, a
    charge sector); a value read or computed rank by rank can differ in its last bits or in what a
    file system shows.  A failure on rank 0 is broadcast too and raised as ``RuntimeError`` on every
    rank, so no rank is left waiting in the broadcast.  Without a communicator the function just runs.
    """
    if comm is None:
        return function()
    outcome = None
    if comm.rank == 0:
        try:
            outcome = (None, function())
        except Exception as exc:
            outcome = (f"{type(exc).__name__}: {exc}", None)
    error, value = comm.bcast(outcome, root=0)
    if error is not None:
        raise RuntimeError(f"Failed on rank 0: {error}")
    return value
