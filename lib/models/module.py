import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models import make_linear_layers, make_conv_layers, make_deconv_layers, LocallyConnected2d, KeypointAttention
from human_models import smpl, coco

class PAREHead(nn.Module):
    def __init__(self, in_dim, pose_feat_dim, shape_feat_dim):
        super().__init__()            
        self.keypoint_deconv_layers = make_conv_layers([in_dim, pose_feat_dim, pose_feat_dim], kernel=3, padding=1, use_bn=True, bnrelu_final=True)
        self.smpl_deconv_layers = make_conv_layers([in_dim, pose_feat_dim, pose_feat_dim], kernel=3, padding=1, use_bn=True, bnrelu_final=True)

        self.keypoint_final_layer = make_conv_layers([pose_feat_dim, smpl.joint_num], kernel=1, padding=0, use_bn=False, bnrelu_final=False)
        
        self.smpl_final_layer = make_conv_layers([pose_feat_dim, shape_feat_dim], kernel=1, padding=0, use_bn=False)
        self.shape_mlp = make_linear_layers([shape_feat_dim*smpl.joint_num,smpl.shape_param_dim], relu_final=False)
        self.cam_mlp = make_linear_layers([shape_feat_dim*smpl.joint_num,3], relu_final=False)
        self.pose_mlp = LocallyConnected2d(in_channels=pose_feat_dim, out_channels=6,  output_size=[smpl.joint_num, 1], kernel_size=1, stride=1)
        
        self.keypoint_attention = KeypointAttention(use_conv=False,
                in_channels=(pose_feat_dim, shape_feat_dim),
                out_channels=(pose_feat_dim, shape_feat_dim),
                act='softmax',
                use_scale=False)
        

    def forward(self, features):
        part_feats = self.keypoint_deconv_layers(features)
        part_attention = self.keypoint_final_layer(part_feats)

        smpl_feats = self.smpl_deconv_layers(features)
        cam_shape_feats = self.smpl_final_layer(smpl_feats)

        point_local_feat = self.keypoint_attention(smpl_feats, part_attention)
        cam_shape_feats = self.keypoint_attention(cam_shape_feats, part_attention)

        pose_feats = point_local_feat
        shape_feats = cam_shape_feats
        pose_feats = pose_feats.unsqueeze(-1)
        shape_feats = torch.flatten(shape_feats, start_dim=1)

        pred_pose = self.pose_mlp(pose_feats)
        pred_cam = self.cam_mlp(shape_feats)
        pred_shape = self.shape_mlp(shape_feats)

        pred_pose = pred_pose.squeeze(-1).transpose(2, 1)
        return pred_pose, pred_shape, pred_cam

class HMRHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, smpl_mean_params='', n_iter=3):
        super().__init__()            
        self.fc1 = nn.Linear(in_dim+157, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hidden_dim, 144)
        self.decshape = nn.Linear(hidden_dim, 10)
        self.deccam = nn.Linear(hidden_dim, 3)
        self.n_iter = n_iter
        
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        
    def forward(self, features):
        batch_size = features.shape[0]
        xf = features.mean((2,3))

        init_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(self.n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        return pred_pose, pred_shape, pred_cam


class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim,bias=False)
        )   

    def forward(self, joint_feat):
        joint_feat = self.projection_head(joint_feat)

        return joint_feat
    

class HeatmapPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.head = make_conv_layers([in_dim, hidden_dim, out_dim], kernel=1, padding=0, use_bn=False)

    def forward(self, x):
        x = self.head(x)
        return x