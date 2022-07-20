import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json

from pycocotools.coco import COCO

from core.config import cfg
from coord_utils import world2cam, cam2pixel, process_bbox
from base_dataset import BaseDataset

class PW3D(BaseDataset):
    def __init__(self, transform, data_split):
        super(PW3D, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('data', 'PW3D', 'data', 'imageFiles')
        self.annot_path = osp.join('data', 'PW3D', 'data')

        self.joint_set = {
            'name': '3DPW',
            'joint_num': 24,
            'joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'),
            'flip_pairs': ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)),
            'skeleton': ((0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12))
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')
        
        self.has_joint_cam = True
        self.has_smpl_param = True
        self.use_pseudo_GT = False

        self.datalist = self.load_data()
        
        
        
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, '3DPW_train.json'))
        else:
            db = COCO(osp.join(self.annot_path, '3DPW_latest_test.json'))
        
        if self.use_pseudo_GT:
            with open(osp.join(self.annot_path, '3DPW_train_SMPL_NeuralAnnot.json')) as f:
                smpl_params = json.load(f)

        sampling_idx = 0
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['sequence'], img['file_name'])

            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
            
            joint_img = np.array(ann['joint_img'], dtype=np.float32).reshape(24,-1)
            joint_cam = np.array(ann['joint_cam'], dtype=np.float32).reshape(24,-1) * 1000  # milimeter
            joint_valid = np.ones((len(joint_img), ))
            joint_img = joint_img[:, :2]
            
            if self.use_pseudo_GT and self.data_split == 'train':
                if str(aid) in smpl_params:
                    cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
                    smpl_param = smpl_params[str(aid)]
                else:
                    continue
            else:
                cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
                smpl_param = ann['smpl_param']
            
            if self.data_split == 'train' and cfg.DATASET.do_subsampling:
                sampling_idx += 1
                if sampling_idx%10 != 0: continue
            
            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img, 
                'joint_cam': joint_cam/1000,
                'joint_valid': joint_valid,
                'cam_param': cam_param,
                'smpl_param': smpl_param
                })

        return datalist
