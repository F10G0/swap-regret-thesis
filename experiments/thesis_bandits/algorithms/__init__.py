from .etc_wrappers import (
    make_etc,
    make_etc_doubling,
)

from .ucb_wrappers import (
    make_ucb_standard,
    make_ucb_delta,
    make_ucb_asymptotically_optimal,
)

from .phased_ucb_wrappers import (
    make_phased_ucb_exponential,
    make_phased_ucb_count_doubling,
)

from .elimination_wrappers import (
    make_elimination,
    make_elimination_standard,
)

__all__ = [
    # Explore-Then-Commit (ETC)
    "make_etc",
    "make_etc_doubling",

    # Upper Confidence Bound (UCB)
    "make_ucb_standard",
    "make_ucb_delta",
    "make_ucb_asymptotically_optimal",

    # Phased UCB
    "make_phased_ucb_exponential",
    "make_phased_ucb_count_doubling",

    # Elimination-based algorithms
    "make_elimination",
    "make_elimination_standard",
]
