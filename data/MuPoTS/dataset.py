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

class MuPoTS(BaseDataset):
    def __init__(self, transform, data_split):
        super(MuPoTS, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('data', 'MuPoTS', 'data', 'MultiPersonTestSet')
        self.annot_path = osp.join('data', 'MuPoTS', 'data')

        assert self.data_split == 'test', "Invalid data subset"

        self.joint_set = {
            'name': 'MuPoTS',
            'joint_num': 17,
            'joints_name': ('Head_Top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head'),
            'flip_pairs': ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)),
            'skeleton': ((0, 16), (16, 15), (15, 1), (1, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13), (15, 2), (2, 3), (3, 4), (15, 5), (5, 6), (6, 7))
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')
        
        self.mpii3d_joints_name = ('Head_Top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head')
        self.mpii3d_smpl_regressor = np.load(osp.join('data', 'base_data', 'J_regressor_mi_smpl.npy'))[:17]    

        self.has_joint_cam = True
        self.has_smpl_param = False
        self.use_pseudo_GT = False

        self.datalist = self.load_data()        
        
    def load_data(self):
        if self.data_split == 'test':
            db = COCO(osp.join(self.annot_path, 'MuPoTS-3D.json'))
       
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            if ann['is_valid'] == 0: continue

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
        
            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
            
            joint_img = np.array(ann['keypoints_img'])
            joint_cam = np.array(ann['keypoints_cam'])
            joint_valid = np.array(ann['keypoints_vis']).astype(np.float32)
            
            fx, fy, cx, cy = img['intrinsic']
            f = np.array([fx, fy]); c = np.array([cx, cy])
            cam_param = {'focal': f, 'princpt': c}

            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img, 
                'joint_cam': joint_cam/1000,
                'joint_valid': joint_valid,
                'cam_param': cam_param
                })

        return datalist

    def mpii_joint_groups(self):
        joint_groups = [
            ['Head', [0]],
            ['Neck', [1]],
            ['Shou', [2,5]],
            ['Elbow', [3,6]],
            ['Wrist', [4,7]],
            ['Hip', [8,11]],
            ['Knee', [9,12]],
            ['Ankle', [10,13]],
        ]
        all_joints = []
        for i in joint_groups:
            all_joints += i[1]
        return joint_groups, all_joints

    def mean(self, l):
        return sum(l) / len(l)

    def mpii_compute_3d_pck(self, seq_err):
        pck_curve_array = []
        pck_array = []
        auc_array = []
        thresh = np.arange(0, 200, 5)
        pck_thresh = 150
        joint_groups, all_joints = self.mpii_joint_groups()
        for seq_idx in range(len(seq_err)):
            pck_curve = []
            pck_seq = []
            auc_seq = []
            err = np.array(seq_err[seq_idx]).astype(np.float32)

            for j in range(len(joint_groups)):
                err_selected = err[:,joint_groups[j][1]]
                buff = []
                for t in thresh:
                    pck = np.float32(err_selected < t).sum() / len(joint_groups[j][1]) / len(err)
                    buff.append(pck) #[Num_thresholds]
                pck_curve.append(buff)
                auc_seq.append(self.mean(buff))
                pck = np.float32(err_selected < pck_thresh).sum() / len(joint_groups[j][1]) / len(err)
                pck_seq.append(pck)
            
            buff = []
            for t in thresh:
                pck = np.float32(err[:, all_joints] < t).sum() / len(err) / len(all_joints)
                buff.append(pck) #[Num_thresholds]
            pck_curve.append(buff)

            pck = np.float32(err[:, all_joints] < pck_thresh).sum() / len(err) / len(all_joints)
            pck_seq.append(pck)

            pck_curve_array.append(pck_curve)   # [num_seq: [Num_grpups+1: [Num_thresholds]]]
            pck_array.append(pck_seq) # [num_seq: [Num_grpups+1]]
            auc_array.append(auc_seq) # [num_seq: [Num_grpups]]

        return pck_curve_array, pck_array, auc_array