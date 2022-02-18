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


class FreiHAND(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        dataset_name = 'FreiHAND'
        self.data_split = data_split
        self.transform = transform
        self.data_path = osp.join('data', 'FreiHAND', 'data')

        # MANO joint set
        # self.mesh_model = MANO()
        self.face = self.mesh_model.face
        self.joint_regressor_mano = self.mesh_model.joint_regressor
        self.vertex_num = self.mesh_model.vertex_num
        self.joint_num = self.mesh_model.joint_num
        self.joints_name = self.mesh_model.joints_name
        self.skeleton = self.mesh_model.skeleton
        self.root_joint_idx = self.mesh_model.root_joint_idx
        self.joint_hori_conn = ((1,5), (5,9), (9,13), (13,17), (2,6),(6,10),(10,14), (14,18), (3,7), (7,11), (11,15), (15,19),(4,8),(8,12),(12,16),(16,20))
            # ((1,5,9,13,17),(2,6,10,14,18),(3,7,11,15,19),(4,8,12,16,20))

        self.datalist = self.load_data()
        det_path = osp.join(self.data_path, f'hrnet_output_on_{mode}set.json')
        self.datalist_pose2d_det = self.load_pose2d_det(det_path)
        print("Check lengths of annotation and detection output: ", len(self.datalist), len(self.datalist_pose2d_det))

    def load_pose2d_det(self, det_path):
        datalist = []

        with open(det_path) as f:
            data = json.load(f)
            for item in sorted(data, key=lambda d: d['image_id']):
                datalist.append({
                    'img_id': item['image_id'],
                    'annot_id': item['annotation_id'],
                    'img_joint':np.array(item['keypoints'], dtype=np.float32)
                })

        return sorted(datalist, key=lambda d: d['img_id'])

    def load_data(self):
        print('Load annotations of FreiHAND ')
        if self.data_split == 'train':
            db = COCO(osp.join(self.data_path, 'freihand_train_coco.json'))
            with open(osp.join(self.data_path, 'freihand_train_data.json')) as f:
                data = json.load(f)

        else:
            db = COCO(osp.join(self.data_path, 'freihand_eval_coco.json'))
            with open(osp.join(self.data_path, 'freihand_eval_data.json')) as f:
                data = json.load(f)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])

            if self.data_split == 'train':
                cam_param, mano_param, joint_cam = data[db_idx]['cam_param'], data[db_idx]['mano_param'], data[db_idx][
                    'joint_3d']
                joint_cam = np.array(joint_cam).reshape(-1, 3)
                bbox = process_bbox(np.array(ann['bbox']))
                if bbox is None: continue

            else:
                cam_param, scale = data[db_idx]['cam_param'], data[db_idx]['scale']
                cam_param['R'] = np.eye(3).astype(np.float32).tolist();
                cam_param['t'] = np.zeros((3), dtype=np.float32)  # dummy
                joint_cam = np.ones((self.joint_num, 3), dtype=np.float32)  # dummy
                mano_param = {'pose': np.ones((48), dtype=np.float32), 'shape': np.ones((10), dtype=np.float32)}

            datalist.append({
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': img_shape,
                'joint_cam': joint_cam,
                'cam_param': cam_param,
                'mano_param': mano_param})

        return sorted(datalist, key=lambda d: d['img_id'])

    def __len__(self):
        return len(self.datalist)
