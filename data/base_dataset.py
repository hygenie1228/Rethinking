import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
import cv2

from torch.utils.data import Dataset

from core.config import cfg
from core.logger import logger
from img_utils import load_img, annToMask
from coord_utils import generate_joint_heatmap, sampling_joint_and_non_joint, image_bound_check, sampling_non_joint
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smpl_param_processing, flip_joint, transform_joint_to_other_db
from human_models import smpl, coco

from vis_utils import vis_keypoints, vis_keypoints_with_skeleton, vis_3d_pose, vis_heatmaps, save_obj


class BaseDataset(Dataset):
    def __init__(self):
        self.transform = None
        self.data_split = None
        self.has_joint_cam = False
        self.has_smpl_param = False
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        if cfg.MODEL.type == 'contrastive':
            return self.get_item_contrastive(index)
        elif cfg.MODEL.type == '2d_joint':
            return self.get_item_2d_joint(index)
        elif cfg.MODEL.type == 'body':
            return self.get_item_body(index)
        elif cfg.MODEL.type == 'hand':
            return self.get_item_hand(index)
        else:
            logger.info('Invalid Model Type!')
            assert 0
            
    def get_item_contrastive(self, index):
        data = copy.deepcopy(self.datalist[index])
        
        img_path, img_shape = data['img_path'], data['img_shape']
        img = load_img(img_path)

        bbox = data['bbox']
        joint_img, joint_valid = data['joint_img'], data['joint_valid']
        
        img_1, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, self.data_split)
        joint_img_1 = coord2D_processing(joint_img, img2bb_trans, do_flip, cfg.MODEL.input_img_shape, self.joint_set['flip_pairs'])
        if do_flip: joint_valid_1 = flip_joint(joint_valid, None, self.joint_set['flip_pairs'])
        else: joint_valid_1 = joint_valid
        
        hm_1, joint_valid_1 = generate_joint_heatmap(joint_img_1, joint_valid_1, cfg.MODEL.input_img_shape, cfg.MODEL.img_feat_shape, sigma=cfg.TRAIN.heatmap_sigma)
        non_joint_img_1 = sampling_non_joint(hm_1, cfg.TRAIN.non_joints_num)

        img_2, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, self.data_split)
        joint_img_2 = coord2D_processing(joint_img, img2bb_trans, do_flip, cfg.MODEL.input_img_shape, self.joint_set['flip_pairs'])
        if do_flip: joint_valid_2 = flip_joint(joint_valid, None, self.joint_set['flip_pairs'])
        else: joint_valid_2 = joint_valid
            
        hm_2, joint_valid_2 = generate_joint_heatmap(joint_img_2, joint_valid_2, cfg.MODEL.input_img_shape, cfg.MODEL.img_feat_shape, sigma=cfg.TRAIN.heatmap_sigma)
        non_joint_img_2 = sampling_non_joint(hm_2, cfg.TRAIN.non_joints_num)

        # debug
        '''
        img_1 = img_1[:,:,::-1]
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_img_1.png'), img_1)
        tmp_hm = hm_1
        tmp_hm[joint_valid_1>0] /= tmp_hm.max((1,2))[joint_valid_1>0][:,None,None]
        tmp_img = vis_heatmaps(img_1[None,...], tmp_hm[None,...])
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_hm_1.png'), tmp_img)
        tmp_img = vis_keypoints_with_skeleton(img_1, np.concatenate([joint_img_1,joint_valid_1[:,None]],1), self.joint_set['skeleton'])
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_joints_1.png'), tmp_img)
        
        img_2 = img_2[:,:,::-1]
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_img_2.png'), img_2)
        tmp_hm = hm_2
        tmp_hm[joint_valid_2>0] /= tmp_hm.max((1,2))[joint_valid_2>0][:,None,None]
        tmp_img = vis_heatmaps(img_1[None,...], tmp_hm[None,...])
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_hm_2.png'), tmp_img)
        tmp_img = vis_keypoints_with_skeleton(img_2, np.concatenate([joint_img_2,joint_valid_2[:,None]],1), self.joint_set['skeleton'])
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_joints_2.png'), tmp_img) 
        '''

        img_1 = self.transform(img_1.astype(np.float32))
        img_2 = self.transform(img_2.astype(np.float32))
        
        joint_img_1[:,0] *= cfg.MODEL.img_feat_shape[1] / cfg.MODEL.input_img_shape[1]
        joint_img_1[:,1] *= cfg.MODEL.img_feat_shape[0] / cfg.MODEL.input_img_shape[0]
        joint_img_2[:,0] *= cfg.MODEL.img_feat_shape[1] / cfg.MODEL.input_img_shape[1]
        joint_img_2[:,1] *= cfg.MODEL.img_feat_shape[0] / cfg.MODEL.input_img_shape[0]

        # convert joint set
        joint_img_1 = transform_joint_to_other_db(joint_img_1, self.joint_set['joints_name'], coco.joints_name)
        joint_img_2 = transform_joint_to_other_db(joint_img_2, self.joint_set['joints_name'], coco.joints_name)
        joint_valid_1 = transform_joint_to_other_db(joint_valid_1, self.joint_set['joints_name'], coco.joints_name)
        joint_valid_2 = transform_joint_to_other_db(joint_valid_2, self.joint_set['joints_name'], coco.joints_name)
        
        joint_img_1 = np.concatenate([joint_img_1, non_joint_img_1])
        joint_img_2 = np.concatenate([joint_img_2, non_joint_img_2])
        joint_img_1, joint_img_2 = joint_img_1.astype(np.float32), joint_img_2.astype(np.float32)

        # 1: visible, 0: not visible, -1: non joint
        non_joint_valid = np.ones((cfg.TRAIN.non_joints_num,)) * -1
        joint_valid_1 = np.concatenate([joint_valid_1, non_joint_valid])
        joint_valid_2 = np.concatenate([joint_valid_2, non_joint_valid])

        batch = {
            'img': [img_1, img_2],
            'joint_img': [joint_img_1, joint_img_2],
            'joint_valid': [joint_valid_1, joint_valid_2]
        }
        
        return batch
    
    def get_item_2d_joint(self, index):
        data = copy.deepcopy(self.datalist[index])
        
        img_path = data['img_path']
        img = load_img(img_path)

        bbox = data['bbox']
        joint_img, joint_valid = data['joint_img'], data['joint_valid']
        
        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, self.data_split)
        joint_img = coord2D_processing(joint_img, img2bb_trans, do_flip, cfg.MODEL.input_img_shape, self.joint_set['flip_pairs'])
        if do_flip: joint_valid = flip_joint(joint_valid, None, self.joint_set['flip_pairs'])

        hm, joint_valid = generate_joint_heatmap(joint_img, joint_valid, cfg.MODEL.input_img_shape, cfg.MODEL.img_feat_shape)

        # debug
        '''
        tmp_img = img[:,:,::-1]
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_img.png'), tmp_img)
        #hm = np.clip(hm.sum(0)[None,...],0,1)
        img2 = vis_heatmaps(tmp_img[None,...], hm[None,...])
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_hm.png'), img2)
        img2 = vis_keypoints_with_skeleton(tmp_img, np.concatenate([joint_img,joint_valid[:,None]],1), self.joint_set['skeleton'])
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_joint_img.png'), img2)
        '''

        img = self.transform(img.astype(np.float32))
        joint_img = transform_joint_to_other_db(joint_img, self.joint_set['joints_name'], coco.joints_name)
        hm = transform_joint_to_other_db(hm, self.joint_set['joints_name'], coco.joints_name)
        joint_valid = transform_joint_to_other_db(joint_valid, self.joint_set['joints_name'], coco.joints_name) 

        if self.data_split == 'train':
            batch = {
                'img': img,
                'hm': hm,
                'hm_valid': joint_valid
            }
        else:
            batch = {
                'img': img,
                'hm': hm,
                'hm_valid': joint_valid
            }
        
        return batch

    def get_item_body(self, index):
        data = copy.deepcopy(self.datalist[index])
        
        img_path = data['img_path']
        img = load_img(img_path)

        bbox, joint_img, joint_valid = data['bbox'], data['joint_img'], data['joint_valid']
        
        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, self.data_split)
        joint_img = coord2D_processing(joint_img, img2bb_trans, do_flip, cfg.MODEL.input_img_shape, self.joint_set['flip_pairs'])
        if do_flip: joint_valid = flip_joint(joint_valid, None, self.joint_set['flip_pairs'])
        
        if self.has_joint_cam:
            joint_cam = coord3D_processing(data['joint_cam'], rot, do_flip, self.joint_set['flip_pairs'])
            joint_cam = joint_cam - joint_cam[self.root_joint_idx]
            has_3D = np.array([1])
        else:
            joint_cam = np.zeros((smpl.joint_num, 3))
            has_3D = np.array([0])
            
        if self.has_smpl_param:
            smpl_pose, smpl_shape = smpl_param_processing(data['smpl_param'], data['cam_param'], do_flip, rot)
            mesh_cam, smpl_joint_cam = self.get_smpl_coord(smpl_pose, smpl_shape)
            has_param = np.array([1])
        else:
            smpl_pose, smpl_shape = np.zeros((smpl.joint_num*3,)), np.zeros((smpl.shape_param_dim,))
            smpl_joint_cam = np.zeros((smpl.joint_num, 3))
            has_param = np.array([0])

        # debug
        '''
        tmp_img = img[:,:,::-1]
        img2 = vis_keypoints_with_skeleton(tmp_img, np.concatenate([joint_img,joint_valid[:,None]],1), self.joint_set['skeleton'])
        cv2.imwrite(osp.join(cfg.vis_dir, f'debug_{index}_joint_img.png'), img2)
        if self.has_joint_cam:
            vis_3d_pose(joint_cam*1000, self.joint_set['skeleton'], 'human36', osp.join(cfg.vis_dir, f'debug_{index}_joint_cam.png')) 
        if self.has_smpl_param:
            vis_3d_pose(smpl_joint_cam*1000, smpl.skeleton, 'smpl', osp.join(cfg.vis_dir, f'debug_{index}_smpl_joint_cam.png')) 
            save_obj(mesh_cam*1000, smpl.face, osp.join(cfg.vis_dir, f'debug_{index}_mesh_cam.obj'))
        '''
        
        if self.data_split == 'train':
            img = self.transform(img.astype(np.float32))
            
            # convert joint set
            joint_img = transform_joint_to_other_db(joint_img, self.joint_set['joints_name'], smpl.joints_name)
            joint_cam = transform_joint_to_other_db(joint_cam, self.joint_set['joints_name'], smpl.joints_name)
            joint_valid = transform_joint_to_other_db(joint_valid, self.joint_set['joints_name'], smpl.joints_name)

            batch = {
                'img': img,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'smpl_joint_cam': smpl_joint_cam,
                'joint_valid': joint_valid,
                'has_3D': has_3D,
                'pose': smpl_pose,
                'shape': smpl_shape,
                'has_param': has_param
            }
        else:
            img = self.transform(img.astype(np.float32))

            if self.joint_set['name'] == 'Human36M':
                joint_cam = joint_cam - joint_cam[smpl.h36m_root_joint_idx]
                joint_cam = joint_cam * 1000
                mesh_cam = np.zeros((smpl.vertex_num, 3))
                
            elif self.joint_set['name'] == '3DPW':
                joint_cam = np.dot(smpl.h36m_joint_regressor, mesh_cam)
                root_cam = joint_cam[smpl.h36m_root_joint_idx]
                joint_cam = joint_cam - root_cam
                mesh_cam = mesh_cam - root_cam

                # meter to milimeter
                mesh_cam, joint_cam = mesh_cam * 1000, joint_cam * 1000
            else:
                logger.info("Not support evaluation!")
                assert 0
               
            batch = {
                'img': img,
                'joint_cam': joint_cam,
                'mesh_cam': mesh_cam
            }
        
        return batch

    def get_item_hand(self, index):
        pass
    
    def get_smpl_coord(self, smpl_pose, smpl_shape):
        root_pose, body_pose, smpl_shape = torch.tensor(smpl_pose[:3]).reshape(1,-1), torch.tensor(smpl_pose[3:]).reshape(1,-1), torch.tensor(smpl_shape).reshape(1,-1)
        output = smpl.layer['neutral'](betas=smpl_shape, body_pose=body_pose, global_orient=root_pose)
        smpl_mesh_cam = output.vertices[0].numpy()
        smpl_joint_cam = np.dot(smpl.joint_regressor, smpl_mesh_cam)
        smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[smpl.root_joint_idx]
        return smpl_mesh_cam, smpl_joint_cam