import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy
import os.path as osp

from models import PoseResNet, PoseHighResolutionNet, Projector, PAREHead, HMRHead, HeatmapPredictor
from core.config import cfg
from core.logger import logger
from collections import OrderedDict
from funcs_utils import load_checkpoint, sample_image_feature, rot6d_to_axis_angle, soft_argmax_2d
from human_models import smpl, coco

class Model(nn.Module):
    def __init__(self, backbone, head):
        super(Model, self).__init__()
        self.backbone = backbone
        self.head = head
        self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
        
        if cfg.TRAIN.freeze_backbone:
            self.trainable_modules = [self.head]
        else:
            self.trainable_modules = [self.backbone, self.head]
        

    def forward(self, inp_img, meta_hm=None, meta_valid=None):
        if cfg.MODEL.type == 'contrastive':
            return self.forward_contrastive(inp_img, meta_hm)
        elif cfg.MODEL.type == '2d_joint':
            return self.forward_2d_joint(inp_img)
        elif cfg.MODEL.type == 'body':
            return self.forward_body(inp_img)
        elif cfg.MODEL.type == 'hand':
            return self.forward_hand(inp_img)
        else:
            logger.info('Invalid Model Type!')
            assert 0

    def forward_contrastive(self, inp_img, meta_hm):
        img_feat = self.backbone(inp_img)
        img_feat = self.head(img_feat)

        # hm normalization
        meta_hm = meta_hm.clone()
        hm_valid = (meta_hm.sum((2,3)) > 0)
        meta_hm[hm_valid] /= meta_hm.sum((2,3))[hm_valid][:,None,None]

        joint_feat = img_feat[:,None,:,:,:] * meta_hm[:,:,None,:,:]
        joint_feat = joint_feat.sum((3,4))

        batch_size, joint_num, _ = joint_feat.shape
        joint_feat = joint_feat.view(batch_size*joint_num, -1)

        joint_feat = F.normalize(joint_feat, dim=1)
        joint_feat = joint_feat.view(batch_size, joint_num, -1)
        return joint_feat

    def forward_2d_joint(self, inp_img):
        batch_size = inp_img.shape[0]
        img_feat = self.backbone(inp_img)
        
        joint_heatmap = self.head(img_feat)
        return joint_heatmap


    def forward_body(self, inp_img):
        batch_size = inp_img.shape[0]
        img_feat = self.backbone(inp_img)

        smpl_pose, smpl_shape, cam_trans = self.head(img_feat)

        smpl_pose = rot6d_to_axis_angle(smpl_pose.reshape(-1,6)).reshape(batch_size,-1)
        cam_trans = self.get_camera_trans(cam_trans)
        joint_proj, joint_cam, mesh_cam = self.get_coord(smpl_pose[:,:3], smpl_pose[:,3:], smpl_shape, cam_trans)
        
        return mesh_cam, joint_cam, joint_proj, smpl_pose, smpl_shape


    def forward_hand(self, inp_img):
        pass
    
    
    def sampling_joint_feature(self, output, joints, joints_valid):
        batch_size = joints_valid.shape[0]
        joint_feat = torch.zeros((batch_size, joints_valid.shape[1], output.shape[1]), device='cuda')
        
        for i in range(batch_size):
            feature = output[i, None]
            points = joints[i]
            points_valid = (joints_valid[i] != 0)

            points = points[points_valid]
            joint_feat[i, points_valid] = sample_image_feature(feature, points, cfg.MODEL.img_feat_shape[1]-1, cfg.MODEL.img_feat_shape[0]-1)
            
        return joint_feat
    
    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.CAMERA.focal[0]*cfg.CAMERA.focal[1]*cfg.CAMERA.camera_3d_size*cfg.CAMERA.camera_3d_size/(cfg.MODEL.input_img_shape[0]*cfg.MODEL.input_img_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans
    
    
    def get_coord(self, smpl_root_pose, smpl_pose, smpl_shape, cam_trans):
        batch_size = smpl_root_pose.shape[0]
        
        output = self.smpl_layer(global_orient=smpl_root_pose, body_pose=smpl_pose, betas=smpl_shape)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(smpl.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        root_joint_idx = smpl.root_joint_idx
        
        # project 3D coordinates to 2D space
        x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[0] + cfg.CAMERA.princpt[0]
        y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[1] + cfg.CAMERA.princpt[1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam
    
    
def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass


def get_model(is_train):
    if cfg.MODEL.backbone == 'resnet50':
        backbone = PoseResNet(50, do_upsampling=cfg.MODEL.use_upsampling_layer)
        if cfg.MODEL.use_upsampling_layer: 
            cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//4, cfg.MODEL.input_img_shape[1]//4)
            backbone_out_dim = 256
        else: 
            cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//32, cfg.MODEL.input_img_shape[1]//32)
            backbone_out_dim = 2048
        pretrained = 'data/base_data/backbone_models/resnet50-19c8e357.pth'
    elif cfg.MODEL.backbone == 'hrnetw32':
        backbone = PoseHighResolutionNet(do_upsampling=cfg.MODEL.use_upsampling_layer)
        cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//4, cfg.MODEL.input_img_shape[1]//4)
        if cfg.MODEL.use_upsampling_layer: backbone_out_dim = 480
        else: backbone_out_dim = 32
        pretrained = 'data/base_data/backbone_models/hrnet_w32-36af842e.pth'
        

    if cfg.MODEL.type == 'contrastive':
        head = nn.Conv2d(in_channels=backbone_out_dim, out_channels=cfg.MODEL.projector_out_dim, kernel_size=1, stride=1,padding=0)
    elif cfg.MODEL.type == '2d_joint':
        head = nn.Conv2d(in_channels=backbone_out_dim, out_channels=coco.joint_num, kernel_size=1, stride=1,padding=0)
    elif cfg.MODEL.type == 'body':
        if cfg.MODEL.regressor == 'pare':
            head = PAREHead(backbone_out_dim, cfg.MODEL.predictor_pose_feat_dim, cfg.MODEL.predictor_shape_feat_dim)
        elif cfg.MODEL.regressor == 'hmr':
            head = HMRHead(backbone_out_dim, smpl_mean_params=osp.join('data','base_data','smpl_mean_params.npz'))
        else:
            assert 0
    elif cfg.MODEL.type == 'hand':
        pass
    else:
        assert 0
    
    
    if is_train:
        if cfg.TRAIN.transfer_backbone:
            logger.info(f"==> Transfer from checkpoint: {cfg.MODEL.weight_path}")
            transfer_backbone(backbone, cfg.MODEL.weight_path)
        else:
            logger.info("trained from" + pretrained)
            backbone.init_weights(pretrained)
            
        head.apply(init_weights)
            
    model = Model(backbone, head)
    return model


def transfer_backbone(backbone, weight_path):   
    checkpoint = load_checkpoint(load_dir=weight_path)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if 'backbone' in k:
            name = k.replace('backbone.', '')
            new_state_dict[name] = v
    
    if len(new_state_dict) == 0:
        backbone.load_state_dict(checkpoint, strict=False)
    else: 
        backbone.load_state_dict(new_state_dict, strict=False)