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