import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
from pycocotools.coco import COCO

from core.config import cfg
from core.logger import logger
from coord_utils import process_bbox
from base_dataset import BaseDataset

class MPII(BaseDataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        if self.data_split == 'train':
            self.img_dir = osp.join('data', 'MPII')
        else:
            logger.info("Unknown data subset")
            assert 0
        self.annot_path = osp.join('data', 'MPII', 'annotations')

        self.joint_set = {
            'joint_num': 16,
            'joints_name': ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Hip', 'Thorax', 'Neck', 'Head_Top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'),
            'flip_pairs': ((0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)),
            'skeleton':  ((0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (7, 12), (12, 11), (11, 10), (7, 13), (13, 14), (14, 15))
        }
        self.has_joint_cam = False
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT

        self.datalist = self.load_data()
        
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'train.json'))
        else:
            db = COCO(osp.join(self.annot_path, 'test.json'))
        
        if self.data_split == 'train' and self.has_smpl_param:
            with open(osp.join(self.annot_path, 'MPII_train_SMPL_NeuralAnnot.json')) as f:
                smpl_params = json.load(f)
        else:
            smpl_params = None
        
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            
            if ann['iscrowd'] or (ann['num_keypoints'] == 0): continue
            
            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
            
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
            joint_valid = (joint_img[:,2].copy().reshape(-1) > 0).astype(np.float32)
            joint_img = joint_img[:, :2]
            
            if smpl_params is not None:
                if str(aid) in smpl_params:
                    smpl_param = smpl_params[str(aid)]['smpl_param']
                    cam_param = smpl_params[str(aid)]['cam_param']
                else:
                    continue
            else:
                smpl_param = None
                cam_param = None
            
            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img, 
                'joint_valid': joint_valid,
                'smpl_param': smpl_param,
                'cam_param': cam_param
                })

        return datalist