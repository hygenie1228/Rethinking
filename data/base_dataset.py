import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
import cv2
import pickle

from torch.utils.data import Dataset

from core.config import cfg
from core.logger import logger
from img_utils import load_img, annToMask
from funcs_utils import convert_focal_princpt
from coord_utils import generate_joint_heatmap, sampling_non_joint, image_bound_check
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smpl_param_processing, flip_joint, transform_joint_to_other_db
from human_models import smpl, coco

from vis_utils import vis_keypoints, vis_keypoints_with_skeleton, vis_3d_pose, vis_heatmaps, save_obj


class BaseDataset(Dataset):
    def __init__(self):
        self.transform = None
        self.data_split = None
        self.has_joint_cam = False
        self.has_smpl_param = False
        self.normalize_imagenet = cfg.MODEL.normalize_imagenet
        
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
        if self.joint_set['name'] == 'MPII':
            joint_img = transform_joint_to_other_db(joint_img, coco.orig_joints_name, coco.joints_name)
            hm = transform_joint_to_other_db(hm, coco.orig_joints_name, coco.joints_name)
            joint_valid = transform_joint_to_other_db(joint_valid, coco.orig_joints_name, coco.joints_name) 
        else:
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

        if self.joint_set['name'] == 'AGORA':
            bbox = data['bbox']
            #img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, self.data_split)
            
            with open(data['joints_2d_path']) as f: 
                joint_img = np.array(json.load(f)).reshape(-1,2)
                joint_img[:,:2] = np.dot(data['img2bb_trans_from_orig'], np.concatenate((joint_img, np.ones_like(joint_img[:,:1])),1).transpose(1,0)).transpose(1,0) # transform from original image to crop_and_resize image
                joint_valid = np.ones((len(joint_img),))

                joint_img = transform_joint_to_other_db(joint_img, self.joint_set['orig_joints_name'], self.joint_set['joints_name'])
                joint_valid = transform_joint_to_other_db(joint_valid, self.joint_set['orig_joints_name'], self.joint_set['joints_name'])

            with open(data['joints_3d_path']) as f:
                joint_cam = np.array(json.load(f)).reshape(-1,3)
                joint_cam = transform_joint_to_other_db(joint_cam, self.joint_set['orig_joints_name'], self.joint_set['joints_name'])
            with open(data['smpl_param_path'], 'rb') as f: param = pickle.load(f, encoding='latin1')
            with open(data['verts_path']) as f:
                orig_mesh_cam = json.load(f)
            
            root_pose = np.array(param['root_pose'], dtype=np.float32).reshape(-1)
            body_pose = np.array(param['body_pose'], dtype=np.float32).reshape(-1)
            smpl_pose = np.concatenate((root_pose, body_pose))
            shape = np.array(param['betas'], dtype=np.float32).reshape(-1)[:10]
            trans = np.array(param['translation'], dtype=np.float32).reshape(-1)
            smpl_param = {'pose': smpl_pose, 'shape': shape, 'trans': trans}
            cam_param = {'focal': cfg.CAMERA.focal, 'princpt': cfg.CAMERA.princpt}

        else:
            bbox, joint_img, joint_valid = data['bbox'], data['joint_img'], data['joint_valid']
            if self.has_joint_cam:
                joint_cam = data['joint_cam']
            if self.has_smpl_param:
                smpl_param, cam_param = data['smpl_param'], data['cam_param']
        
        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, self.data_split)
        joint_img = coord2D_processing(joint_img, img2bb_trans, do_flip, cfg.MODEL.input_img_shape, self.joint_set['flip_pairs'])
        if do_flip: joint_valid = flip_joint(joint_valid, None, self.joint_set['flip_pairs'])
        
        if self.has_joint_cam:
            joint_cam = coord3D_processing(joint_cam, rot, do_flip, self.joint_set['flip_pairs'])
            joint_cam = joint_cam - joint_cam[self.root_joint_idx]
            has_3D = np.array([1])
        else:
            joint_cam = np.zeros((smpl.joint_num, 3))
            has_3D = np.array([0])
            
        if self.has_smpl_param:
            smpl_pose, smpl_shape = smpl_param_processing(smpl_param, cam_param, do_flip, rot)
            mesh_cam, smpl_joint_cam = self.get_smpl_coord(smpl_pose, smpl_shape)
            has_param = np.array([1])
        else:
            smpl_pose, smpl_shape = np.zeros((smpl.joint_num*3,)), np.zeros((smpl.shape_param_dim,))
            smpl_joint_cam = np.zeros((smpl.joint_num, 3))
            has_param = np.array([0])

        smpl_pose_valid = np.ones((smpl.joint_num*3), dtype=np.float32)
        if self.joint_set['name'] == 'AGORA':
            smpl_pose_valid[:3] = 0 # global orient of the provided parameter is a rotation to world coordinate system. I want camera coordinate system.

        if self.normalize_imagenet:
            img = self.transform(img.astype(np.float32))
        else:
            img = self.transform(img.astype(np.float32))/255.

        if self.data_split == 'train':
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
                'has_param': has_param,
                'smpl_pose_valid': smpl_pose_valid
            }
        else:
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
            elif self.joint_set['name'] == 'MuPoTS':
                #joint_cam = joint_cam - joint_cam[self.root_joint_idx]
                joint_cam = data['joint_cam']
                joint_cam = joint_cam * 1000
                mesh_cam = np.zeros((smpl.vertex_num, 3))

            elif self.joint_set['name'] == 'AGORA':
                mesh_cam = np.array(orig_mesh_cam)
                joint_cam = np.dot(smpl.h36m_joint_regressor, mesh_cam)
                root_cam = joint_cam[smpl.h36m_root_joint_idx]
                joint_cam = joint_cam - root_cam
                mesh_cam = mesh_cam - root_cam

                # meter to milimeter
                mesh_cam, joint_cam = mesh_cam * 1000, joint_cam * 1000
            else:
                mesh_cam = np.zeros((smpl.vertex_num, 3))
                joint_cam = np.zeros((smpl.joint_num, 3))               
            
            batch = {
                'img': img,
                'bbox': bbox,
                'joint_cam': joint_cam,
                'mesh_cam': mesh_cam
            }

            if self.joint_set['name'] == 'MuPoTS':
                batch['cam_param'] = data['cam_param']
                batch['imgname'] = data['img_path'].split('/')[-2] + '_' + data['img_path'].split('/')[-1].split('.')[0]
                batch['img_path'] = data['img_path']

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