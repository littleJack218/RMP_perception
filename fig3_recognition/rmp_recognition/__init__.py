"""RMP mixed-timescale recognition: trained models and evaluation protocol.

Reproduces the recognition conclusions of the manuscript (Fig. 3e, 3f):
joint letter (4-class) and rotation-direction (2-class) recognition,
comparing the reconfigurable macropixel (RMP, 17 channels) against an
equivalent-area uniform pixel array (81 channels), including the
Gaussian-noise robustness sweep (sigma = 0-0.4).
"""

from .data import add_pixel_noise, load_test_set
from .evaluation import evaluate_all, evaluate_scheme
from .model import MultiTaskMLP
from .utils import set_seed

__version__ = '1.1.0'

__all__ = [
    'MultiTaskMLP',
    'add_pixel_noise',
    'evaluate_all',
    'evaluate_scheme',
    'load_test_set',
    'set_seed',
]
