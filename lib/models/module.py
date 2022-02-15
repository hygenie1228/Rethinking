import torch
import torch.nn as nn
from torch.nn import functional as F

from models import make_linear_layers, make_conv_layers, make_deconv_layers, LocallyConnected2d

from human_models import smpl, coco

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

class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, img_feat_shape, pos_enc=False):
        super().__init__()
        self.img_feat_shape = img_feat_shape
        self.pos_enc = pos_enc
        
        if pos_enc:
            in_dim += 2
            
        self.conv = make_conv_layers([in_dim, hidden_dim, hidden_dim], kernel=1, padding=0, use_bn=False)
        self.root_pose_out = make_linear_layers([hidden_dim,hidden_dim,6], relu_final=False, use_bn=True)
        self.pose_out = make_linear_layers([hidden_dim,hidden_dim,(smpl.joint_num-1)*6], relu_final=False, use_bn=True)
        self.shape_out = make_linear_layers([hidden_dim,hidden_dim,smpl.shape_param_dim], relu_final=False, use_bn=True)
        self.cam_out = make_linear_layers([hidden_dim,hidden_dim,3], relu_final=False, use_bn=True)
        
        if self.pos_enc:
            pos_h = torch.arange(self.img_feat_shape[0])[:, None]
            pos_h = torch.repeat_interleave(pos_h, self.img_feat_shape[1], dim=1)
            pos_w = torch.arange(self.img_feat_shape[1])[None, :]
            pos_w = torch.repeat_interleave(pos_w, self.img_feat_shape[0], dim=0)
            positional_encoding = torch.stack([pos_h, pos_w])
            self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        if self.pos_enc:
            positional_encoding = torch.repeat_interleave(self.positional_encoding[None,...], x.shape[0], dim=0)
            x = torch.cat([x, positional_encoding], dim=1)
        
        x = self.conv(x)
        x = x.mean((2,3))

        root_pose = self.root_pose_out(x)
        pose = self.pose_out(x)
        shape = self.shape_out(x)
        cam_trans = self.cam_out(x)

        pose = torch.cat([root_pose,pose], dim=-1)
        return pose, shape, cam_trans


class BodyPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, shape_feat_dim, img_feat_shape):
        super().__init__()
        self.img_feat_shape = img_feat_shape
            
        self.conv = make_conv_layers([in_dim, hidden_dim_1, hidden_dim_1], kernel=3, padding=1, use_bn=True, bnrelu_final=True)
        
        self.atten_conv = make_conv_layers([hidden_dim_1, hidden_dim_2, hidden_dim_2], kernel=3, padding=1, use_bn=True, bnrelu_final=True)
        self.smpl_conv = make_conv_layers([hidden_dim_1, hidden_dim_2, hidden_dim_2], kernel=3, padding=1, use_bn=True, bnrelu_final=True)
        self.shape_cam_conv = make_conv_layers([hidden_dim_2, shape_feat_dim], kernel=1, padding=0, use_bn=False)

        self.atten_final_conv = make_conv_layers([hidden_dim_2, smpl.joint_num], kernel=1, padding=0, use_bn=False, bnrelu_final=False)
        self.pose_out = LocallyConnected2d(in_channels=hidden_dim_2, out_channels=6,  output_size=[smpl.joint_num, 1], kernel_size=1, stride=1)
        self.shape_out = make_linear_layers([shape_feat_dim*smpl.joint_num,smpl.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([shape_feat_dim*smpl.joint_num,3], relu_final=False)

    def forward(self, x):
        x = self.conv(x)
        atten_map = self.atten_conv(x)
        atten_map = self.atten_final_conv(atten_map)        
        
        pose_feat = self.smpl_conv(x)
        shape_cam_feat = self.shape_cam_conv(pose_feat)
        
        # adapt attention
        pose_feat = self.attention_feature(pose_feat, atten_map)
        shape_cam_feat = self.attention_feature(shape_cam_feat, atten_map)

        # pose layer
        pose = self.pose_out(pose_feat.unsqueeze(-1))
        pose = pose.squeeze(-1).transpose(2, 1)

        # shape & cam layer
        shape_cam_feat = torch.flatten(shape_cam_feat, start_dim=1)
        shape = self.shape_out(shape_cam_feat)
        cam_trans = self.cam_out(shape_cam_feat)

        return pose, shape, cam_trans

    def attention_feature(self, features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape
        normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)
        features = features.reshape(batch_size, -1, height*width)

        features = torch.matmul(normalized_heatmap, features.transpose(2,1))
        features = features.transpose(2,1)
        return features