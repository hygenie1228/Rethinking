from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

util_path = osp.join(this_dir, '..', 'lib', 'utils')
add_path(util_path)

smpl_path = osp.join(this_dir, '..', 'lib', 'smplpytorch')
add_path(smpl_path)

densepose_path = osp.join(this_dir, '..', 'lib', 'densepose')
add_path(densepose_path)

data_path = osp.join(this_dir, '..', 'data')
add_path(data_path)
