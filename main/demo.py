import os
import os.path as osp
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
import shutil
import cv2
import __init_path

import torchvision.transforms as transforms

from core.config import update_config, cfg

parser = argparse.ArgumentParser(description='Test AnimateHuman')

parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--dir', default='asset/example2', help='input data directory')
parser.add_argument('--action', default='multi-view', help='input data directory')
parser.add_argument('--gpu', type=str, default='0,', help='assign multi-gpus by comma concat')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])


from core.base import Demo

predictor = Demo(args, load_dir=cfg.MODEL.weight_path, input_dir=args.dir)
predictor.inference(args.action)
