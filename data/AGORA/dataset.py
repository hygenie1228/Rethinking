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


class AGORA(BaseDataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.annot_path = osp.join('data', 'AGORA', 'data')
        self.resolution = (2160, 3840)
        
        if self.resolution == (2160, 3840):
            self.img_dir = osp.join('data', 'AGORA', 'data', '3840x2160')
        

        self.joint_set = {
            'name': 'AGORA',
            'joint_num': 22,
            'joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'),
            'flip_pairs': ((1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21)),
            'skeleton': ((0,1), (0,2))
        }

        self.has_joint_cam = False
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT
        
        self.datalist = self.load_data()
        
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'AGORA_train.json'))
        else:
            db = COCO(osp.join(self.annot_path, 'AGORA_validation.json'))

        sampling_idx = 0
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            if not ann['is_valid']:
                continue
            
            gender = ann['gender']
            joints_2d_path = osp.join(self.annot_path, ann['smplx_joints_2d_path'])
            joints_3d_path = osp.join(self.annot_path, ann['smplx_joints_3d_path'])
            verts_path = osp.join(self.annot_path, ann['smpl_verts_path'])
            smpl_param_path = osp.join(self.annot_path, ann['smpl_param_path'])

            if self.resolution == (2160, 3840): # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                img_path = osp.join(self.img_dir, img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_ann_id_' + str(aid) + '.png')
                json_path = osp.join(self.img_dir, img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_ann_id_' + str(aid) + '.json')
                if not osp.isfile(json_path):
                    continue
                with open(json_path) as f:
                    crop_resize_info = json.load(f)
                    img2bb_trans_from_orig = np.array(crop_resize_info['img2bb_trans'], dtype=np.float32)
                    resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info['resized_width']
                img_shape = (resized_height, resized_width)
                bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                
                data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'joints_2d_path': joints_2d_path, 'joints_3d_path': joints_3d_path, 'verts_path': verts_path, 'smpl_param_path': smpl_param_path, 'gender': gender, 'ann_id': str(aid)}
                datalist.append(data_dict)

        return datalist