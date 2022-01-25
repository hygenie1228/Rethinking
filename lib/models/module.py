import torch
import torch.nn as nn
from torch.nn import functional as F

#from models import ResNetBackbone
from core.config import cfg
from models import make_linear_layers, make_conv_layers, make_deconv_layers, LocallyConnected2d

from utils.human_models import smpl

class ResNetFPN(nn.Module):
    def __init__(self, resnet_type, n_feats):
        super().__init__()      
        
        if resnet_type == 50:
            layers = [256, 512, 1024, 2048]

        n_feats = n_feats//4
        self.resnet = ResNetBackbone(resnet_type)
        self.fpn_lateral2 = nn.Conv2d(layers[0], n_feats, 1, 1)
        self.fpn_lateral3 = nn.Conv2d(layers[1], n_feats, 1, 1)
        self.fpn_lateral4 = nn.Conv2d(layers[2], n_feats, 1, 1)
        self.fpn_lateral5 = nn.Conv2d(layers[3], n_feats, 1, 1)

        self.fpn_output2 = nn.Sequential(*[nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(n_feats, n_feats, 3, 1, padding=1), \
                                          nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(n_feats, n_feats, 3, 1, padding=1), \
                                          nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(n_feats, n_feats, 3, 1, padding=1)])
        self.fpn_output3 = nn.Sequential(*[nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(n_feats, n_feats, 3, 1, padding=1), \
                                          nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(n_feats, n_feats, 3, 1, padding=1)])
        self.fpn_output4 = nn.Sequential(*[nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(n_feats, n_feats, 3, 1, padding=1)])
        self.fpn_output5 = nn.Sequential(*[nn.Conv2d(n_feats, n_feats, 3, 1, padding=1)])


    def forward(self, x):
        x = self.resnet.feature_extract(x)

        features = []
        feature = self.fpn_lateral5(x[3])
        features.append(feature)
        
        top_down_features = F.interpolate(feature, scale_factor=2, mode="nearest")
        lateral_features = self.fpn_lateral4(x[2])
        feature = lateral_features + top_down_features
        features.append(feature)

        top_down_features = F.interpolate(feature, scale_factor=2, mode="nearest")
        lateral_features = self.fpn_lateral3(x[1])
        feature = lateral_features + top_down_features
        features.append(feature)

        top_down_features = F.interpolate(feature, scale_factor=2, mode="nearest")
        lateral_features = self.fpn_lateral2(x[0])
        feature = lateral_features + top_down_features
        features.append(feature)

        x1 = self.fpn_output2(features[0])
        x2 = self.fpn_output3(features[1])
        x3 = self.fpn_output4(features[2])
        x4 = self.fpn_output5(features[3])

        x = torch.cat([x1, x2, x3, x4], 1)
        return x

class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.projection_head = make_linear_layers([in_dim, hidden_dim, out_dim], relu_final=False)

    def forward(self, x):
        x = self.projection_head(x)
        return x
    
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
    def __init__(self, in_dim, hidden_dim, pose_feat_dim, shape_feat_dim, img_feat_shape, pos_enc=False):
        super().__init__()
        self.img_feat_shape = img_feat_shape
        self.pos_enc = pos_enc
        
        if pos_enc:
            in_dim += 2
            
        self.conv = make_conv_layers([in_dim, hidden_dim, hidden_dim], kernel=3, padding=1, use_bn=True, bnrelu_final=True)

        self.joint_conv = make_conv_layers([hidden_dim, smpl.joint_num * cfg.MODEL.depth_size], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.atten_conv = make_conv_layers([hidden_dim, hidden_dim, smpl.joint_num], kernel=3, padding=1, use_bn=True, bnrelu_final=True)
        self.smpl_conv = make_conv_layers([hidden_dim, hidden_dim, pose_feat_dim], kernel=3, padding=1, use_bn=True, bnrelu_final=True)
        self.shape_cam_conv = make_conv_layers([pose_feat_dim, shape_feat_dim], kernel=1, padding=0, use_bn=False)

        self.pose_out = LocallyConnected2d(in_channels=pose_feat_dim, out_channels=6,  output_size=[smpl.joint_num, 1], kernel_size=1, stride=1)
        self.shape_out = make_linear_layers([shape_feat_dim*smpl.joint_num,smpl.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([shape_feat_dim*smpl.joint_num,3], relu_final=False)

        if self.pos_enc:
            pos_h = torch.arange(self.img_feat_shape[0])[:, None]
            pos_h = torch.repeat_interleave(pos_h, self.img_feat_shape[1], dim=1)
            pos_w = torch.arange(self.img_feat_shape[1])[None, :]
            pos_w = torch.repeat_interleave(pos_w, self.img_feat_shape[0], dim=0)
            positional_encoding = torch.stack([pos_h, pos_w])
            self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        batch_size = x.shape[0]

        if self.pos_enc:
            positional_encoding = torch.repeat_interleave(self.positional_encoding[None,...], x.shape[0], dim=0)
            x = torch.cat([x, positional_encoding], dim=1)

        x = self.conv(x)

        # get 2.5D joint locations
        h, w = x.shape[2:]
        joint_heatmap = self.joint_conv(x).view(-1, smpl.joint_num, cfg.MODEL.depth_size, h, w)
        joint_img = self.soft_argmax_3d(joint_heatmap)
        joint_img[..., 0] = joint_img[..., 0] / w * cfg.MODEL.input_img_shape[1]
        joint_img[..., 1] = joint_img[..., 1] / h * cfg.MODEL.input_img_shape[0]

        atten_map = self.atten_conv(x)
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

        return pose, shape, cam_trans, joint_img

    def attention_feature(self, features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape
        normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)
        features = features.reshape(batch_size, -1, height*width)

        features = torch.matmul(normalized_heatmap, features.transpose(2,1))
        features = features.transpose(2,1)
        return features

    def soft_argmax_3d(self, heatmap3d):
        d, h, w = heatmap3d.shape[-3:]
        heatmap3d = heatmap3d.reshape((-1, smpl.joint_num, d * h * w))
        heatmap3d = F.softmax(heatmap3d, 2)
        heatmap3d = heatmap3d.reshape((-1, smpl.joint_num, d, h, w))

        accu_x = heatmap3d.sum(dim=(2, 3))
        accu_y = heatmap3d.sum(dim=(2, 4))
        accu_z = heatmap3d.sum(dim=(3, 4))

        accu_x = accu_x * torch.arange(w).float().cuda()[None, None, :]
        accu_y = accu_y * torch.arange(h).float().cuda()[None, None, :]
        accu_z = accu_z * torch.arange(d).float().cuda()[None, None, :]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
        return coord_out