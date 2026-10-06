"""
This module serves the purpose of centering
 the default number generation in the library,
 both for unseeded and seeded Random Number Generation.

 For use case check get_rng() example
"""

import numpy as np

# Instantiates an unseeded geneator
#Never call directly, use get_rng
_rng = np.random.default_rng()


def set_seed(seed):
    """
    Changes the seed of the generator
    """
    global _rng
    _rng = np.random.default_rng(seed)

def get_rng(rng: None | int | np.random.Generator=None):
    """
    Returns a generator:

        None -> the default generator of the library
        int -> a new generator with that seed
        Generator -> just returns the generator
    
    Use case::

        from utils import _random

        rng = _random.get_rng() #returns a generator
        X_normal = rng.normal(...)
    """

    if rng is None:
        return _rng
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)