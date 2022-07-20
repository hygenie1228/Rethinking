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
from coord_utils import process_bbox, get_bbox
from base_dataset import BaseDataset

from aug_utils import transform_joint_to_other_db
from vis_utils import vis_keypoints, vis_keypoints_with_skeleton, vis_3d_pose, vis_heatmaps, save_obj
from human_models import smpl, coco

def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

class MPII(BaseDataset):
    def __init__(self, transform, data_split):
        super(MPII, self).__init__()
        self.transform = transform
        self.data_split = data_split
        if self.data_split == 'train':
            self.img_dir = osp.join('data', 'MPII')
        else:
            logger.info("Unknown data subset")
            assert 0
        self.annot_path = osp.join('data', 'MPII', 'annotations')

        self.use_openpose = True

        self.joint_set = {
            'name': 'MPII',
            'joint_num': 16,
            'joints_name': ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Hip', 'Thorax', 'Neck', 'Head_Top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'),
            'flip_pairs': ((0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)),
            'skeleton':  ((0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (7, 12), (12, 11), (11, 10), (7, 13), (13, 14), (14, 15))
        }
        self.has_joint_cam = False
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT

        self.datalist = self.load_data()

       
        if self.use_openpose:
            self.joint_set['joint_num'] = coco.orig_joint_num
            self.joint_set['joints_name'] = coco.orig_joints_name
            self.joint_set['flip_pairs'] = coco.orig_flip_pairs
            self.joint_set['skeleton'] = coco.orig_skeleton

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
        
        use_openpose = self.use_openpose

        if use_openpose:
            import pickle
            with open('pred_kpt.pkl', 'rb') as f:
                openpose = pickle.load(f)

        sampling_idx = 0
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
            
            #####
            if use_openpose:
                pred_poses = openpose[img_path.split('/')[-1]]
                joint_img = self.get_closest_keypoints(pred_poses, joint_img, joint_valid, img_path)

                if joint_img is None:
                    continue

                joint_valid = joint_img[:,2]
                joint_img = joint_img[:,:2]
                bbox = get_bbox(joint_img, joint_valid)

            if smpl_params is not None:
                if str(aid) in smpl_params:
                    smpl_param = smpl_params[str(aid)]['smpl_param']
                    cam_param = smpl_params[str(aid)]['cam_param']
                else:
                    continue
            else:
                smpl_param = None
                cam_param = None
            
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
                'joint_valid': joint_valid,
                'smpl_param': smpl_param,
                'cam_param': cam_param
                })

        return datalist

    def get_closest_keypoints(self, pred_poses, joint_img, joint_valid, img_path):
        match_idx = 0
        err = 60  # pixel

        joint_img = transform_joint_to_other_db(joint_img, self.joint_set['joints_name'], coco.orig_joints_name)
        joint_valid = transform_joint_to_other_db(joint_valid, self.joint_set['joints_name'], coco.orig_joints_name)

        for idx in range(len(pred_poses)):
            pred = pred_poses[idx]['keypoints']
            pred = np.array(pred)
            pred_valid = pred[:,2] > 0.1
            pred = pred[:,:2]
            
            if sum(pred_valid) < 4:
                pred_valid = None
                continue

            valid_idx = (pred_valid * joint_valid).nonzero()[0]
            
            l1_err = np.abs(pred[valid_idx] - joint_img[valid_idx])

            euc_dst = np.sqrt((l1_err**2).sum(axis=1)).mean()

            if euc_dst < err:
                match_idx = idx
                err = euc_dst

            '''import cv2
            img = cv2.imread(img_path)
            img2 = vis_keypoints_with_skeleton(img, np.concatenate([pred,pred_valid[:,None]],1), coco.orig_skeleton)
            cv2.imwrite(osp.join(cfg.vis_dir, f'debug_joint_img.png'), img2)'''
        
        try:
            joint_img = np.concatenate([pred,pred_valid[:,None]],1)
            #joint_img = transform_joint_to_other_db(joint_img, coco.orig_joints_name,  self.joint_set['joints_name'])
            #if sum(joint_img[:,2]) <4:
            #    return None

            return joint_img
        except:
            return None