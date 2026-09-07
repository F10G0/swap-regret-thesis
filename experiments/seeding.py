"""Stable, domain-separated seed derivation for experiment randomness."""

from operator import index

import numpy as np


ENVIRONMENT_SEED_DOMAIN = 0x454E5652  # "ENVR"
LEARNER_SEED_DOMAIN = 0x4C454152  # "LEAR"


def domain_separated_seed(base_seed: int, replicate: int, domain: int) -> int:
    """Derive a reproducible uint64 seed for one role and replicate."""
    try:
        base_seed = index(base_seed)
        replicate = index(replicate)
        domain = index(domain)
    except TypeError as error:
        raise ValueError("seed inputs must be integers") from error
    if base_seed < 0 or replicate < 0 or domain < 0:
        raise ValueError("seed inputs must be non-negative")

    sequence = np.random.SeedSequence([base_seed, replicate, domain])
    words = sequence.generate_state(2, dtype=np.uint32)
    return (int(words[0]) << 32) | int(words[1])
