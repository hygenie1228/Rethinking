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
        super(AGORA, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.annot_path = osp.join('data', 'AGORA', 'data')
        self.resolution = (2160, 3840)
        
        if self.resolution == (2160, 3840):
            self.img_dir = osp.join('data', 'AGORA', 'data', '3840x2160')
        

        self.joint_set = {
            'name': 'AGORA',
            'joint_num': 45,
            'joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4'),
            'flip_pairs': ((1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23), (25,26), (27,28), (29,32), (30,33), (31,34), (35,40), (36,41), (37,42), (38,43), (39,44)),
            'orig_joint_num': 127,
            'orig_joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body
                                'Jaw', 'L_Eye_SMPLH', 'R_Eye_SMPLH',  # SMPLH
                                'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',  # fingers
                                'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',  # fingers
                                'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  # face in body
                                'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # feet
                                'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4',  # finger tips
                                'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', # finger tips
                                *['Face_' + str(i) for i in range(5,56)] # face 
                                )
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')

        self.has_joint_cam = True
        self.has_smpl_param = True
        
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