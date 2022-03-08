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
from coord_utils import process_bbox
from base_dataset import BaseDataset


class InstaVariety(BaseDataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        self.img_dir = osp.join('data', 'InstaVariety', 'images')
        self.annot_path = osp.join('data', 'InstaVariety', 'annotations')

        self.joint_set = {
            'name': 'InstaVariety',
            'joint_num': 25,
            'joints_name': ('R_Heel','R_Knee','R_Hip','L_Hip','L_Knee','L_Heel','R_Wrist','R_Elbow','R_Shoulder','L_Shoulder','L_Elbow','L_Wrist','Neck','Head_Top','Nose','L_Eye','R_Eye','L_Ear','R_Ear','L_Big_Toe','R_Big_Toe','L_Small_Toe','R_Small_Toe','L_Ankle','R_Ankle'),
            'flip_pairs': ((0,5), (1,4), (2,3), (6,11), (7,10), (8,9), (15,16), (17,18), (19,20), (21,22), (23,24)),
            'skeleton': ((0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(2,8),(3,9),(10,11),(8,12),(9,12),(12,13),(12,14),(14,15),(14,16),(15,17),(16,18),(0,20),(20,22),(5,19),(19,21),(5,23),(0,24))
        }
        
        self.has_joint_cam = False
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT

        self.datalist = self.load_data()
        
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'insta_variety_annotations_light_train.json'))
        else:
            db = COCO(osp.join(self.annot_path, 'insta_variety_annotations_light_test.json'))
        
        sampling_idx = 0
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            
            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
            
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
            joint_valid = (joint_img[:,2].copy().reshape(-1) > 0).astype(np.float32)
            joint_img = joint_img[:, :2]

            if self.has_smpl_param:
                if str(aid) in smpl_params:
                    smpl_param = smpl_params[str(aid)]['smpl_param']
                    cam_param = smpl_params[str(aid)]['cam_param']
                else:
                    continue
            else:
                smpl_param = None
                cam_param = None
            
            sampling_idx += 1
            if sampling_idx%3 != 0: continue

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