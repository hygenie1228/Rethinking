import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from core.config import cfg
from core.logger import logger
from core.config import cfg
from coord_utils import world2cam, cam2pixel, process_bbox
from base_dataset import BaseDataset

class MPI_INF_3DHP(BaseDataset):
    def __init__(self, transform, data_split):
        super(MPI_INF_3DHP, self).__init__()
        self.transform = transform
        self.data_split = data_split
        if self.data_split == 'train':
            self.img_dir = osp.join('data', 'MPI_INF_3DHP', 'data', 'images_1k')
        else:
            logger.info("Unknown data subset")
            assert 0
        self.annot_path = osp.join('data', 'MPI_INF_3DHP', 'data')

        self.joint_set = {
            'name': 'MPI_INF_3DHP',
            'joint_num': 17,
            'joints_name': (('Head_Top', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Torso', 'Head')),
            'flip_pairs': ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)),
            'skeleton': ((0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13))
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')
        
        self.has_joint_cam = True
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT
        
        self.datalist = self.load_data()
    
    
    def load_data(self):
        db = COCO(osp.join(self.annot_path, 'MPI-INF-3DHP_1k.json'))
        with open(osp.join(self.annot_path, 'MPI-INF-3DHP_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.annot_path, 'MPI-INF-3DHP_camera_1k.json')) as f:
            cameras = json.load(f)
        if cfg.TRAIN.use_pseudo_GT:
            with open(osp.join(self.annot_path, 'MPI-INF-3DHP_SMPL_NeuralAnnot.json'),'r') as f:
                smpl_params = json.load(f)
        else:
            smpl_params = None

        sampling_idx = 0
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            subject_idx = img['subject_idx']
            seq_idx = img['seq_idx']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_dir, 'S' + str(subject_idx), 'Seq' + str(seq_idx), 'imageSequence', img['file_name'])
            
            # frame sampling (25 frame per sec -> 25/3 frame per sec)
            if frame_idx % 3 != 0:
                continue
            
            # check smpl parameter exist
            if smpl_params is not None:
                try:
                    smpl_param = smpl_params[str(subject_idx)][str(seq_idx)][str(frame_idx)]
                except KeyError:
                    smpl_param = None
            else:
                smpl_param = None
            
            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject_idx)][str(seq_idx)][str(cam_idx)]
            R, t, focal, princpt = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal':focal, 'princpt':princpt}
            
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject_idx)][str(seq_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]
            joint_valid = np.ones((self.joint_set['joint_num'],))

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
                'smpl_param': smpl_param,
                'cam_param': cam_param
                })

        return datalist