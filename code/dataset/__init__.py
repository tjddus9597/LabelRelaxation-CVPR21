from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .import utils
from .base import BaseDataset

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP
}

def load(name, root, mode, transform = None, is_CRD = False):
    return _type[name](root = root, mode = mode, transform = transform, is_CRD = is_CRD)