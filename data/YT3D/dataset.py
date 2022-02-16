import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from core.config import cfg
from human_models import mano
from coord_utils import get_bbox, process_bbox
# from preprocessing import load_img, process_bbox, augmentation, process_human_model_output, get_bbox

from vis_utils import vis_keypoints, save_obj
from base_dataset import BaseDataset


class YT3D(BaseDataset):
    def __init__(self, transform, data_split):

        self.transform = transform
        self.data_split = data_split
        self.annot_path = osp.join('data', 'YT3D', 'annotations')
        self.image_path = osp.join('data', 'YT3D', 'images')

        self.datalist_pose2d_det = self.load_pose2d_det()
        self.datalist = self.load_data()
        
        self.joint_set = {
            'name': 'YT3D',
            'joint_num': mano.joint_num,
            'joints_name': mano.joints_name,
            'flip_pairs': (),
            'skeleton': ()
        }
        

        print("Check lengths of annotation and detection output: ", len(self.datalist), len(self.datalist_pose2d_det))
        
        

    def load_pose2d_det(self):
        datalist = {}

        for i in range(0,8):
            det_path = osp.join(self.annot_path, f'openpose_YT3D_trainset{i}.json')
            with open(det_path) as f:
                data = json.load(f)
                for item in data:
                    datalist[item['annotation_id']] = {'img_id': item['image_id'],  'img_joint': item['img_joint']}

        return datalist

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'youtube_train.json'))

        datalist = []
        for aid in self.datalist_pose2d_det.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.image_path, img['name'])
            img_shape = (img['height'], img['width'])

            if self.data_split == 'train':
                mano_mesh = np.array(ann['vertices'], dtype=np.float32)  # 2.5D
                is_left = ann['is_left']
                # openpose = self.datalist_pose2d_det[aid]['img_joint']
                if is_left:
                    mano_mesh[:, 0] = img_shape[1] - 1 - mano_mesh[:, 0]
                mano_joints = np.dot(mano.joint_regressor, mano_mesh)
                mano_joints_valid = np.ones_like(mano_joints[:,0], dtype=np.float32)
                bbox = get_bbox(mano_joints, mano_joints_valid, extend_ratio=1.2)
                hand_bbox = process_bbox(bbox, img['width'], img['height'])
                if hand_bbox is None: continue
                
                mano_joints = mano_joints[:, :2]

                datalist.append({
                    'ann_id': aid,
                    'img_id': image_id, 
                    'img_path': img_path, 
                    'img_shape': (img['height'], img['width']),
                    'bbox': hand_bbox,
                    'joint_img': mano_joints,
                    'joint_valid': mano_joints_valid, 
                    # 'cam_param': cam_param,
                    # 'mano_param': mano_param,
                    # 'mano_mesh': mano_mesh
                })

        return datalist

    def __len__(self):
        return len(self.datalist)
