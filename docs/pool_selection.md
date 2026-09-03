# Pool selection

The pool builder first filters ineligible loans at the chosen reporting date, calculates deterministic risk/prepayment proxies, then ranks and selects loans until the configured UPB target plus tolerance is reached. It reports every configured constraint separately; a failed constraint is never hidden.

The current implementation is intentionally a deterministic greedy v1. It is suitable for transparent analyst workflows and modest candidate universes. It is not a global mixed-integer optimum. Candidate-versus-selected metrics and concentration results must be reviewed before use.
