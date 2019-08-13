"""Module for linking trips together."""

from .base import BatchedLinker

from .bipartite import MaxCardinalityLinker

try:
    from .bipartite_ortools import (MinWeightMaxCardinalityLinker,
                                    MinWeight


# # If pyfftw is available, import PyfftwFFTMaker.
# try:
#     from .pyfftw import PyfftwFFTMaker
#     from os import environ
#     get_fft_maker.system_default = PyfftwFFTMaker(
#         flags=['FFTW_ESTIMATE'],
#         threads=int(environ.get('OMP_NUM_THREADS', 2)))
#     del environ
# except ImportError:
#     pass