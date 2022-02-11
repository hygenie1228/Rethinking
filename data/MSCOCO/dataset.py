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
from coord_utils import process_bbox, add_pelvis_and_neck
from base_dataset import BaseDataset


class MSCOCO(BaseDataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        if self.data_split == 'train':
            self.img_dir = osp.join('data', 'MSCOCO', 'images', 'train2017')
        else:
            self.img_dir = osp.join('data', 'MSCOCO', 'images', 'val2017')
        self.annot_path = osp.join('data', 'MSCOCO', 'annotations')

        self.orig_joint_set = {
            'name': 'MSCOCO',
            'joint_num': 17,
            'joints_name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'),
            'flip_pairs': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
            'skeleton': ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12),(5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
        }

        self.joint_set = {
            'name': 'MSCOCO',
            'joint_num': 19,
            'joints_name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck'),
            'flip_pairs': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
            'skeleton': ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        }
        
        self.has_joint_cam = False
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT
        
        self.datalist = self.load_data()
        
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_train_v1.0.json'))
        else:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_val_v1.0.json'))
            
        if self.has_smpl_param:
            if self.data_split == 'train':
                with open(osp.join(self.annot_path, 'MSCOCO_train_SMPL_NeuralAnnot.json')) as f:
                    smpl_params = json.load(f)
            else:
                with open(osp.join(self.annot_path, 'MSCOCO_val_SMPL_NeuralAnnot.json')) as f:
                    smpl_params = json.load(f)
        
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

            # pre-processing joint_img
            joint_img, joint_valid = add_pelvis_and_neck(joint_img, joint_valid, self.joint_set['joints_name'])

            if self.has_smpl_param:
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