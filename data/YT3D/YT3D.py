import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import mano
from utils.preprocessing import load_img, process_bbox, augmentation, process_human_model_output, get_bbox
from utils.vis import vis_keypoints, vis_mesh, save_obj


class YT3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.annot_path = osp.join('..', 'data', 'YT3D', 'annotations')
        self.image_path = osp.join('..', 'data', 'YT3D', 'images')

        self.datalist_pose2d_det = self.load_pose2d_det()
        self.datalist = self.load_data()

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

                datalist.append({'img_id': image_id, 'img_path': img_path, 'img_shape': img_shape, 'hand_bbox': hand_bbox,
                                 'mano_joints': mano_joints, 'mano_joints_valid': mano_joints_valid, 'mano_mesh': mano_mesh})

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_id, img_path, img_shape, hand_bbox = data['img_id'], data['img_path'], data['img_shape'], data['hand_bbox']

        # img
        img = load_img(img_path)
        hand_img, hand_img2bb_trans, hand_bb2img_trans, hand_rot, hand_do_flip, hand_translation = augmentation(img, hand_bbox, self.data_split, enforce_flip=False)
        hand_img = self.transform(hand_img.astype(np.float32)) / 255.

        # get 2.5D mesh
        mano_mesh = data['mano_mesh']

        # reflect flip augmentation
        if hand_do_flip:
            mano_mesh[:, 0] = img_shape[1] - 1 - mano_mesh[:, 0]
            # hand has no flip pairs; for pair in mano.flip_pairs:

        # x,y affine transform / cfg.input_img_shape
        mano_mesh = np.concatenate((mano_mesh[:, :2], np.ones_like(mano_mesh[:, 0:1])), 1)
        mano_mesh[:, :2] = np.dot(hand_img2bb_trans, mano_mesh.transpose(1, 0)).transpose(1, 0)[:, :2]
        mano_joints = np.dot(mano.joint_regressor, mano_mesh)

        # process GTs
        mano_mesh[:, 0] = mano_mesh[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        mano_mesh[:, 1] = mano_mesh[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        mano_joints[:, 0] = mano_joints[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        mano_joints[:, 1] = mano_joints[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        # check truncation
        mano_joints_valid = ((mano_joints[:, 0] >= 0) * (mano_joints[:, 0] < cfg.output_hm_shape[2]) * (mano_joints[:, 1] >= 0) * (mano_joints[:, 1] < cfg.output_hm_shape[1])).reshape(-1,1).astype(np.float32)

        # # for debug
        # _tmp = det_hand_joint_img.copy()
        # _tmp[:,0] = _tmp[:,0] / (cfg.output_hm_shape[1] * 8) * cfg.input_img_shape[1]
        # _tmp[:,1] = _tmp[:,1] / (cfg.output_hm_shape[0] * 8) * cfg.input_img_shape[0]
        # _img = hand_img.numpy().transpose(1,2,0)[:,:,::-1]
        # _img = vis_keypoints(_img, _tmp)
        # cv2.imshow('HRNet freihand_' + str(idx) + '.jpg', _img)
        # cv2.waitKey(0)

        if cfg.pretrain:
            data2 = copy.deepcopy(self.datalist[idx])
            img_id, img_path, img_shape, hand_bbox = data2['img_id'], data2['img_path'], data2['img_shape'], data2['hand_bbox']

            # img
            img = load_img(img_path)
            hand_img2, hand_img2bb_trans2, hand_bb2img_trans2, hand_rot2, hand_do_flip2, hand_translation2 = augmentation(img, hand_bbox, self.data_split, enforce_flip=False)
            hand_img2 = self.transform(hand_img2.astype(np.float32)) / 255.

            # get 2.5D mesh
            mano_mesh2 = data['mano_mesh']
            # x,y affine transform / cfg.input_img_shape
            mano_mesh2 = np.concatenate((mano_mesh2[:, :2], np.ones_like(mano_mesh2[:, 0:1])), 1)
            mano_mesh2[:, :2] = np.dot(hand_img2bb_trans, mano_mesh2.transpose(1, 0)).transpose(1, 0)[:, :2]
            mano_joints2 = np.dot(mano.joint_regressor, mano_mesh2)

            mano_joints2[:, 0] = mano_joints2[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            mano_joints2[:, 1] = mano_joints2[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            # check truncation
            mano_joints_valid2 = ((mano_joints2[:, 0] >= 0) * (mano_joints2[:, 0] < cfg.output_hm_shape[2]) * (mano_joints2[:, 1] >= 0) * (mano_joints2[:, 1] < cfg.output_hm_shape[1])).reshape(-1, 1).astype(np.float32)

            hand_img = np.concatenate([hand_img[None], hand_img2[None]], axis=0)
            mano_joints = np.concatenate([mano_joints[None, :, :2], mano_joints2[None, :, :2]], axis=0)
            mano_joints_valid = np.concatenate([mano_joints_valid[None], mano_joints_valid2[None]], axis=0)
            hand_rot = np.array([hand_rot, hand_rot2], dtype=np.float32)
            hand_translation = np.concatenate([hand_translation[None], hand_translation2[None]], axis=0)

            inputs = {'hand_img': hand_img, 'hand_joints': mano_joints * 8, 'hand_joints_mask': mano_joints_valid}
            targets = {}
            meta_info = {'rotation': hand_rot, 'translation': hand_translation}

        elif self.data_split == 'train':
            # # for debug
            # _tmp = mano_joints.copy()
            # _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[1] * cfg.input_img_shape[1]
            # _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[0] * cfg.input_img_shape[0]
            # _img = hand_img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            # _img = vis_keypoints(_img, _tmp)
            # cv2.imshow('GT freihand_' + str(idx) + '.jpg', _img / 255.)
            # cv2.waitKey(0)

            mano_joint_cam = np.zeros((mano_joints.shape[0], 3), dtype=np.float32)
            mano_pose = np.zeros((mano.orig_joint_num, 3), dtype=np.float32).reshape(-1)
            mano_shape = np.zeros((mano.shape_param_dim, ), dtype=np.float32)
            mano_param_valid = np.zeros((mano.orig_joint_num, 3), dtype=np.float32)

            inputs = {'hand_img': hand_img, 'hand_joints': mano_joints[:, :2] * 8, 'hand_joints_mask': mano_joints_valid}
            targets = {'hand_joint_img': mano_joints, 'mano_joint_img': mano_joints, 'mano_mesh_img': mano_mesh, 'hand_joint_cam': mano_joint_cam, 'mano_joint_cam': mano_joint_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
            meta_info = {'mano_mesh_valid': np.ones((778,1)), 'mano_param_valid': mano_param_valid, 'hand_joint_valid': np.zeros_like(mano_joints_valid, dtype=np.float32), 'hand_joint_trunc': mano_joints_valid, 'mano_joint_trunc': mano_joints_valid,
                         'is_valid_mano_fit': float(False), 'is_3D': float(False)}
        else:
            inputs = {'hand_img': hand_img, 'hand_joints': mano_joints * 8, 'hand_joints_mask': mano_joints_valid}
            targets = {}
            meta_info = {}

        return inputs, targets, meta_info
