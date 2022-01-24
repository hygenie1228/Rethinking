import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy

from models import PoseResNet, ResNetFPN, PoseHighResolutionNet, Projector, Predictor, BodyPredictor, HeatmapPredictor
from core.config import cfg
from core.logger import logger
from collections import OrderedDict
from utils.funcs_utils import sample_image_feature, rot6d_to_axis_angle
from utils.human_models import smpl

class Model(nn.Module):
    def __init__(self, backbone, head, loss):
        super(Model, self).__init__()
        self.backbone = backbone
        self.head = head
        self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
        self.loss = nn.ModuleDict(loss)

        if cfg.TRAIN.freeze_backbone:
            self.trainable_modules = [self.head]
        else:
            self.trainable_modules = [self.backbone, self.head]

    def forward(self, batch, mode='train'):
        if cfg.TRAIN.two_view  and mode == 'train':
            for k in batch.keys():
                batch[k] = torch.cat(batch[k])

        if cfg.MODEL.type == 'contrastive':
            inp_img, joint_img, joint_valid = batch['img'], batch['joint_img'], batch['joint_valid']

            features = self.forward_contrastive(inp_img, joint_img, joint_valid)

            half_batch_size = len(features) // 2
            features = torch.stack([features[:half_batch_size], features[half_batch_size:]])
            features = features.permute(1, 2, 0, 3)
            joint_valid = joint_valid[:half_batch_size] * joint_valid[half_batch_size:]

            # features: [hbs, joint_num+non_joint_num, n_views, feat_dim]
            # joint_valid: [hbs, joint_num+non_joint_num]

            return features, joint_valid

        elif cfg.MODEL.type == 'body':
            inp_img = batch['img'].cuda()
            pred_mesh_cam, pred_joint_cam, pred_joint_proj, pred_smpl_pose, pred_smpl_shape, pred_joint_img = self.forward_body(inp_img)

            if mode == 'train':
                tar_joint_img, tar_joint_cam, tar_smpl_joint_cam = batch['joint_img'].cuda(), batch['joint_cam'].cuda(), batch['smpl_joint_cam'].cuda()
                tar_pose, tar_shape = batch['pose'].cuda(), batch['shape'].cuda()
                meta_joint_valid, meta_has_3D, meta_has_param = batch['joint_valid'].cuda(), batch['has_3D'].cuda(), batch['has_param'].cuda()

                loss = {}
                loss['joint_cam'] = cfg.TRAIN.joint_loss_weight * self.loss['joint_cam'](pred_joint_cam, tar_joint_cam, meta_joint_valid * meta_has_3D)
                loss['smpl_joint_cam'] = cfg.TRAIN.joint_loss_weight * self.loss['smpl_joint_cam'](pred_joint_cam, tar_smpl_joint_cam, meta_has_param[:, :, None])
                loss['joint_proj'] = cfg.TRAIN.proj_loss_weight * self.loss['joint_proj'](pred_joint_proj, tar_joint_img[..., :2], meta_joint_valid)
                loss['pose_param'] = cfg.TRAIN.pose_loss_weight * self.loss['pose_param'](pred_smpl_pose, tar_pose, meta_has_param)
                loss['shape_param'] = cfg.TRAIN.shape_loss_weight * self.loss['shape_param'](pred_smpl_shape, tar_shape, meta_has_param)
                loss['prior'] = cfg.TRAIN.prior_loss_weight * self.loss['prior'](pred_smpl_pose[:, 3:], pred_smpl_shape)

                loss['joint_img'] = cfg.TRAIN.joint_img_loss_weight * self.loss['joint_img'](pred_joint_img, tar_joint_img, meta_joint_valid, meta_has_3D)
                return loss

            else:
                return pred_mesh_cam, pred_joint_cam, pred_joint_proj, pred_smpl_pose, pred_smpl_shape

        else:
            logger.info('Invalid Model Type!')
            assert 0


    def forward_contrastive(self, inp_img, joints=None, joints_valid=None):
        batch_size = inp_img.shape[0]
        img_feat = self.backbone(inp_img)

        joint_feat = self.sampling_joint_feature(img_feat, joints, joints_valid)
        joint_feat = joint_feat.reshape(-1, joint_feat.shape[-1])

        joint_embedding = self.head(joint_feat)
        joint_embedding = F.normalize(joint_embedding, dim=1)
        joint_embedding = joint_embedding.reshape(batch_size, -1, joint_embedding.shape[-1])
        return joint_embedding


    def forward_2d_joint(self, inp_img):
        batch_size = inp_img.shape[0]
        img_feat = self.backbone(inp_img)
        
        joint_heatmap = self.head(img_feat)
        return joint_heatmap


    def forward_body(self, inp_img):
        batch_size = inp_img.shape[0]
        img_feat = self.backbone(inp_img)

        smpl_pose, smpl_shape, cam_trans, joint_img = self.head(img_feat)

        smpl_pose = rot6d_to_axis_angle(smpl_pose.reshape(-1,6)).reshape(batch_size,-1)
        cam_trans = self.get_camera_trans(cam_trans)
        joint_proj, joint_cam, mesh_cam = self.get_coord(smpl_pose[:,:3], smpl_pose[:,3:], smpl_shape, cam_trans)
        
        return mesh_cam, joint_cam, joint_proj, smpl_pose, smpl_shape, joint_img


    def forward_hand(self, inp_img):
        pass
    
    
    def sampling_joint_feature(self, img_feat, joints, joints_valid):
        batch_size, joint_num = joints.shape[:2]

        img_feat_joints = []
        for j in range(joint_num):
            x = joints[:, j, 0] / (cfg.MODEL.img_feat_shape[1]-1) * 2 - 1
            y = joints[:, j, 1] / (cfg.MODEL.img_feat_shape[0]-1) * 2 - 1
            grid = torch.stack((x, y), 1)[:, None, None, :]
            img_feat_j = F.grid_sample(img_feat, grid, align_corners=True)[:, :, 0, 0]  # (batch_size, channel_dim)
            img_feat_joints.append(img_feat_j)

        img_feat_joints = torch.stack(img_feat_joints)  # (joint_num, batch_size, channel_dim)
        img_feat_joints = img_feat_joints.permute(1, 0, 2)  # (batch_size, joint_num, channel_dim)
        return img_feat_joints
    
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


def get_model(is_train, loss):
    if cfg.MODEL.backbone == 'resnet50':
        backbone = PoseResNet(50, do_upsampling=cfg.MODEL.use_upsampling_layer)
        if cfg.MODEL.use_upsampling_layer: 
            cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//4, cfg.MODEL.input_img_shape[1]//4)
            backbone_out_dim = 256
        else: 
            cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//32, cfg.MODEL.input_img_shape[1]//32)
            backbone_out_dim = 2048
    elif cfg.MODEL.backbone == 'hrnetw32':
        backbone = PoseHighResolutionNet(do_upsampling=cfg.MODEL.use_upsampling_layer)
        cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//4, cfg.MODEL.input_img_shape[1]//4)
        if cfg.MODEL.use_upsampling_layer: backbone_out_dim = 480
        else: backbone_out_dim = 32
    
    if cfg.MODEL.type == 'contrastive':
        head = Projector(backbone_out_dim,cfg.MODEL.projector_hidden_dim,cfg.MODEL.projector_out_dim)
    elif cfg.MODEL.type == '2d_joint':
        head = HeatmapPredictor(backbone_out_dim,cfg.MODEL.predictor_hidden_dim,smpl.joint_num)
    elif cfg.MODEL.type == 'body':
        head = BodyPredictor(backbone_out_dim,cfg.MODEL.predictor_hidden_dim, cfg.MODEL.predictor_pose_feat_dim, cfg.MODEL.predictor_shape_feat_dim, img_feat_shape=cfg.MODEL.img_feat_shape, pos_enc=True)
    elif cfg.MODEL.type == 'hand':
        pass
    else:
        assert 0
    
    
    if is_train:
        if cfg.TRAIN.pretrained_model_type is 'posecontrast' or cfg.TRAIN.transfer_backbone:
            backbone.init_weights('')
        else:
            message = f'==> Pretrained type: {cfg.TRAIN.pretrained_model_type.upper()}'
            logger.info(message)
            backbone.init_weights(cfg.TRAIN.pretrained_model_type)
            
        head.apply(init_weights)
            
    model = Model(backbone, head, loss)
    return model


def transfer_backbone(model, checkpoint):    
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if 'backbone' in k:
            name = k.replace('backbone.', '')
            new_state_dict[name] = v
            
    model.backbone.load_state_dict(new_state_dict)