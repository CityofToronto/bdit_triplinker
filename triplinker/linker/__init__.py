"""Module for linking trips together."""

from .batched import BatchedLinker
from .bipartite import MaxCardinalityLinker

# If Google OR-Tools is available, import additional linkers.
try:
    from .bipartite_ortools import (MinWeightMaxCardinalityLinker,
                                    MinWeightMaximalLinker)
except ImportError:
    pass
