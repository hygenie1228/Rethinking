import sys
import os.path as osp
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter

from core.config import cfg
from core.logger import logger

from multiple_datasets import MultipleDatasets
from models.model import get_model
from core.loss import get_loss
from coord_utils import get_max_preds, flip_back
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters
from eval_utils import eval_mpjpe, eval_pa_mpjpe, calc_dists, dist_acc
from vis_utils import save_plot
from human_models import smpl, coco

from renderer import render_mesh

for dataset in cfg.DATASET.train_list+cfg.DATASET.test_list:
    exec(f'from {dataset}.dataset import {dataset}')

def get_dataloader(dataset_names, is_train):
    if len(dataset_names) == 0: return None, None

    dataset_split = 'TRAIN' if is_train else 'TEST'  
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    logger.info(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        if cfg.MODEL.normalize_imagenet:
            transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.ToTensor()
        dataset = eval(f'{name}')(transform, dataset_split.lower())
        logger.info(f"# of {dataset_split} {name} data: {len(dataset)}")
        dataset_list.append(dataset)
    
    def worker_init_fn(worder_id):
        np.random.seed(np.random.get_state()[1][0] + worder_id)

    if not is_train:
        for dataset in dataset_list:
            dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=True,
                                drop_last=False,
                                worker_init_fn=worker_init_fn)
            dataloader_list.append(dataloader)
        
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, partition=cfg.DATASET.train_partition, make_same_len=cfg.DATASET.make_same_len)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=batch_per_dataset * len(dataset_names), shuffle=cfg[dataset_split].shuffle,
                                     num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
        return dataset_list, batch_generator


def prepare_network(args, load_dir='', is_train=True):    
    model, checkpoint = None, None
    
    model = get_model(is_train)
    logger.info(f"==> Constructing Model...")
    logger.info(f"# of model parameters: {count_parameters(model)}")
    logger.info(model)
    
    if load_dir and (not is_train or args.resume_training):
        logger.info(f"==> Loading checkpoint: {load_dir}")
        checkpoint = load_checkpoint(load_dir=load_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    return model, checkpoint


def train_setup(model, checkpoint):    
    criterion, optimizer, lr_scheduler = None, None, None
    if cfg.MODEL.type == 'contrastive':
        loss_history = {'total_loss': [], 'contrast_loss': []}
        error_history = {}
    elif cfg.MODEL.type == '2d_joint':
        loss_history = {'total_loss': [], 'hm_loss': []}
        error_history = {'pck': []}
    elif cfg.MODEL.type == 'body':
        loss_history = {'total_loss': [], 'joint_loss': [], 'smpl_joint_loss': [], 'proj_loss': [], 'pose_param_loss': [], 'shape_param_loss': [], 'prior_loss': []}
        error_history = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
    
    criterion = get_loss()
    optimizer = get_optimizer(model=model)
    lr_scheduler = get_scheduler(optimizer=optimizer)
    
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        curr_lr = 0.0

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']

        lr_state = checkpoint['scheduler_state_dict']
        lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
        lr_scheduler.load_state_dict(lr_state)

        loss_history = checkpoint['train_log']
        error_history = checkpoint['test_log']
        cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
        logger.info("===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}"
                    .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return criterion, optimizer, lr_scheduler, loss_history, error_history
    

class Trainer:
    def __init__(self, args, load_dir):
        self.model, checkpoint = prepare_network(args, load_dir, True)
        self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history = train_setup(self.model, checkpoint)
        dataset_list, self.batch_generator = get_dataloader(cfg.DATASET.train_list, is_train=True)
        
        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model) 
        self.print_freq = cfg.TRAIN.print_freq
        
        if cfg.MODEL.type == 'contrastive':
            self.train = self.train_contrastive
            self.contrast_loss_weight = cfg.TRAIN.contrast_loss_weight
        elif cfg.MODEL.type == '2d_joint':
            self.train = self.train_2d_joint
            self.hm_loss_weight = cfg.TRAIN.hm_loss_weight
        elif cfg.MODEL.type == 'body':
            self.train = self.train_body
            self.joint_loss_weight = cfg.TRAIN.joint_loss_weight
            self.proj_loss_weight = cfg.TRAIN.proj_loss_weight
            self.pose_loss_weight = cfg.TRAIN.pose_loss_weight
            self.shape_loss_weight = cfg.TRAIN.shape_loss_weight
            self.prior_loss_weight = cfg.TRAIN.prior_loss_weight
        elif cfg.MODEL.type == 'hand':
            self.train = self.train_hand

    def train_contrastive(self, epoch):
        self.model.train()
        lr = self.lr_scheduler.get_lr()[0]

        running_loss = 0.0
        running_contrast_loss = 0.0
        
        batch_generator = tqdm(self.batch_generator)
        for i, batch in enumerate(batch_generator):
            inp_img_1, inp_img_2 = batch['img'][0].cuda(), batch['img'][1].cuda()
            meta_hm_1, meta_hm_2 = batch['hm'][0].cuda(), batch['hm'][1].cuda()
            meta_joint_valid_1, meta_joint_valid_2 = batch['joint_valid'][0].cuda(), batch['joint_valid'][1].cuda()
            
            inp_img = torch.cat([inp_img_1, inp_img_2])
            meta_hm = torch.cat([meta_hm_1, meta_hm_2])
            meta_joint_valid = torch.cat([meta_joint_valid_1, meta_joint_valid_2])
            
            joint_feat = self.model(inp_img, meta_hm, meta_joint_valid)

            batch_size = inp_img_1.shape[0]
            joint_feat = torch.stack([joint_feat[:batch_size],joint_feat[batch_size:]])
            joint_feat = joint_feat.permute(1,2,0,3).contiguous()
            meta_joint_valid = meta_joint_valid_1 * meta_joint_valid_2

            # joint_feat: [bs, joint_num, n_views, feat_dim]
            # joint_valid: [bs, joint_num]
            loss1 = self.contrast_loss_weight*self.loss['joint_cont'](joint_feat, meta_joint_valid)
            loss = loss1
            
            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            loss, loss1 = loss.detach(), loss1.detach()
            running_loss += float(loss.item())
            running_contrast_loss += float(loss1.item())

            if i % self.print_freq == 0:
                batch_generator.set_description(f'Epoch{epoch} ({i}/{len(batch_generator)}), lr {lr:.1E} => '
                                                f'joint_cont loss: {loss1:.4f}')
            
            # visualize
            if cfg.TRAIN.vis and i % (len(batch_generator)//6) == 0:
                import cv2
                from vis_utils import vis_keypoints_with_skeleton, vis_heatmaps
                inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                img = inv_normalize(inp_img_1[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                img = np.ascontiguousarray(img, dtype=np.uint8)
                
                joint_img, _ = get_max_preds(meta_hm.cpu().detach().numpy())
                joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]
                joint_valid = meta_joint_valid.cpu().detach().numpy()

                img = inv_normalize(inp_img_1[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                img = np.ascontiguousarray(img, dtype=np.uint8)
                tmp_hm = meta_hm_1[0].cpu().numpy()
                tmp_img = vis_heatmaps(img[None,...], tmp_hm[None,...])
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_hm_1.png'), tmp_img)

                tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([joint_img[0],joint_valid[0,:, None]],1), coco.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_1.png'), tmp_img)

                img = inv_normalize(inp_img_2[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                img = np.ascontiguousarray(img, dtype=np.uint8)
                tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([joint_img[batch_size],joint_valid[0,:, None]],1), coco.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_2.png'), tmp_img)
                
                tmp_hm = meta_hm_2[0].cpu().numpy()
                tmp_img = vis_heatmaps(img[None,...], tmp_hm[None,...])
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_hm_2.png'), tmp_img)

            
        self.loss_history['total_loss'].append(running_loss / len(batch_generator))
        self.loss_history['contrast_loss'].append(running_contrast_loss / len(batch_generator))  
            
        logger.info(f'Epoch{epoch} Loss: {self.loss_history["total_loss"][-1]:.4f}')
        
    def train_2d_joint(self, epoch):
        self.model.train()
        lr = self.lr_scheduler.get_lr()[0]

        running_loss = 0.0
        running_hm_loss = 0.0
        
        batch_generator = tqdm(self.batch_generator)
        for i, batch in enumerate(batch_generator):
            inp_img = batch['img'].cuda()
            tar_heatmap = batch['hm'].cuda()
            meta_hm_valid = batch['hm_valid'].cuda()
            
            pred_heatmap = self.model(inp_img)

            loss1 = self.hm_loss_weight * self.loss['hm'](pred_heatmap, tar_heatmap, meta_hm_valid)
            loss = loss1
            
            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            loss, loss1 = loss.detach(), loss1.detach()
            running_loss += float(loss.item())
            running_hm_loss += float(loss1.item())
            
            if i % self.print_freq == 0:
                batch_generator.set_description(f'Epoch{epoch} ({i}/{len(batch_generator)}), lr {lr} => '
                                                f'hm loss: {loss1:.4f}')
            
            # visualize 
            if cfg.TRAIN.vis and i % (len(batch_generator)//6) == 0:
                import cv2
                from vis_utils import vis_keypoints_with_skeleton, vis_heatmaps
                
                inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                img = inv_normalize(inp_img[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                img = np.ascontiguousarray(img, dtype=np.uint8)
                
                pred_heatmap = pred_heatmap.cpu().detach().numpy()
                tar_heatmap = tar_heatmap.cpu().detach().numpy()
                meta_hm_valid = meta_hm_valid.cpu().numpy()

                pred_joint_img, pred_joint_valid = get_max_preds(pred_heatmap)
                pred_joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                pred_joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]

                tar_joint_img, _ = get_max_preds(tar_heatmap)
                tar_joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                tar_joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]
                
                tmp_img = vis_heatmaps(img[None,...], pred_heatmap[0, None,...])
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_hm_pred.png'), tmp_img)

                tmp_img = vis_heatmaps(img[None,...], tar_heatmap[0, None,...])
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_hm_gt.png'), tmp_img)

                tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([pred_joint_img[0],pred_joint_valid[0,:]],1), coco.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_pred.png'), tmp_img)

                tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([tar_joint_img[0],meta_hm_valid[0,:, None]],1), coco.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_gt.png'), tmp_img)
    

        self.loss_history['total_loss'].append(running_loss / len(batch_generator)) 
        self.loss_history['hm_loss'].append(running_hm_loss / len(batch_generator))     
        logger.info(f'Epoch{epoch} Loss: {self.loss_history["total_loss"][-1]:.4f}')

        
    def train_body(self, epoch):
        self.model.train()
        lr = self.lr_scheduler.get_lr()[0]

        running_loss = 0.0
        running_joint_loss = 0.0
        running_smpl_joint_loss = 0.0
        running_proj_loss = 0.0
        running_pose_param_loss = 0.0
        running_shape_param_loss = 0.0
        running_prior_loss = 0.0
        
        batch_generator = tqdm(self.batch_generator)
        for i, batch in enumerate(batch_generator):
            inp_img = batch['img'].cuda()
            tar_joint_img, tar_joint_cam, tar_smpl_joint_cam = batch['joint_img'].cuda(), batch['joint_cam'].cuda(), batch['smpl_joint_cam'].cuda()
            tar_pose, tar_shape = batch['pose'].cuda(), batch['shape'].cuda()
            meta_joint_valid, meta_has_3D, meta_has_param = batch['joint_valid'].cuda(), batch['has_3D'].cuda(), batch['has_param'].cuda()
            
            pred_mesh_cam, pred_joint_cam, pred_joint_proj, pred_smpl_pose, pred_smpl_shape, pred_joint_img = self.model(inp_img)

            loss1 = self.joint_loss_weight * self.loss['joint_cam'](pred_joint_cam, tar_joint_cam, meta_joint_valid * meta_has_3D)
            loss2 = self.joint_loss_weight * self.loss['smpl_joint_cam'](pred_joint_cam, tar_smpl_joint_cam, meta_has_param[:,:,None])
            loss3 = self.proj_loss_weight * self.loss['joint_proj'](pred_joint_proj, tar_joint_img, meta_joint_valid)
            loss4 = self.pose_loss_weight * self.loss['pose_param'](pred_smpl_pose, tar_pose, meta_has_param)
            loss5 = self.shape_loss_weight * self.loss['shape_param'](pred_smpl_shape, tar_shape, meta_has_param)
            if cfg.MODEL.regressor == 'pose2pose':
                loss6 = self.proj_loss_weight * self.loss['joint_proj'](pred_joint_img, tar_joint_img, meta_joint_valid)
            else:
                loss6 = torch.tensor(0).cuda()
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            
            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            loss, loss1, loss2, loss3, loss4, loss5, loss6 = loss.detach(), loss1.detach(), loss2.detach(), loss3.detach(), loss4.detach(), loss5.detach(), loss6.detach()
            running_loss += float(loss.item())
            running_joint_loss += float(loss1.item())
            running_smpl_joint_loss += float(loss2.item())
            running_proj_loss += float(loss3.item())
            running_pose_param_loss += float(loss4.item())
            running_shape_param_loss += float(loss5.item())
            running_prior_loss += float(loss6.item())
            
            if i % self.print_freq == 0:
                batch_generator.set_description(f'Epoch{epoch} ({i}/{len(batch_generator)}) => '
                                                f'joint: {loss1:.4f} smpl_joint: {loss2:.4f} proj: {loss3:.4f} pose: {loss4:.4f}, shape: {loss5:.4f}, prior: {loss6:.4f}')
            
            # visualize 
            if cfg.TRAIN.vis and i % (len(batch_generator)//2) == 0:
                import cv2
                from vis_utils import vis_keypoints_with_skeleton, vis_3d_pose, save_obj
                inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                img = inv_normalize(inp_img[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                img = np.ascontiguousarray(img, dtype=np.uint8)
                
                pred_joint_proj, pred_joint_cam = pred_joint_proj[0].detach().cpu().numpy(), pred_joint_cam[0].detach().cpu().numpy()
                tar_joint_img, tar_joint_cam = tar_joint_img[0].cpu().numpy(), tar_joint_cam[0].cpu().numpy()
                meta_joint_valid, meta_has_3D = meta_joint_valid[0].cpu().numpy(), meta_has_3D[0].cpu().numpy()
            
                tmp_img = vis_keypoints_with_skeleton(img, pred_joint_proj, smpl.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_pred.png'), tmp_img)
                vis_3d_pose(pred_joint_cam*1000, smpl.skeleton, 'smpl', osp.join(cfg.vis_dir, f'train_{i}_joint_cam_pred.png'))
                
                tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([tar_joint_img,meta_joint_valid[:,None]],1), smpl.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_gt.png'), tmp_img)
                
                save_obj(pred_mesh_cam[0].detach().cpu().numpy(), smpl.face, osp.join(cfg.vis_dir, f'train_{i}_mesh_cam_pred.obj'))

                if meta_has_3D > 0:
                    vis_3d_pose(tar_joint_cam*1000, smpl.skeleton, 'smpl', osp.join(cfg.vis_dir, f'train_{i}_joint_cam_gt.png'), kps_3d_vis=meta_joint_valid)

                if meta_has_param[0] > 0:
                    tar_pose, tar_shape = tar_pose[0].cpu(), tar_shape[0].cpu()
                    tar_smpl_joint_cam = tar_smpl_joint_cam[0].cpu().numpy()

                    root_pose, body_pose = tar_pose[:3].reshape(1,-1), tar_pose[3:].reshape(1,-1)
                    smpl_shape = tar_shape.reshape(1,-1)

                    output = smpl.layer['neutral'](betas=smpl_shape, body_pose=body_pose, global_orient=root_pose)
                    gt_mesh_cam = output.vertices[0].numpy()
                    joint_cam = np.dot(smpl.h36m_joint_regressor, gt_mesh_cam)
                    gt_mesh_cam = gt_mesh_cam - joint_cam[smpl.h36m_root_joint_idx]
                    
                    vis_3d_pose(tar_smpl_joint_cam*1000, smpl.skeleton, 'smpl', osp.join(cfg.vis_dir, f'train_{i}_smpl_joint_cam_gt.png'))
                    save_obj(gt_mesh_cam * 1000, smpl.face, osp.join(cfg.vis_dir, f'train_{i}_mesh_cam_gt.obj'))
                    
                    

        self.loss_history['total_loss'].append(running_loss / len(batch_generator)) 
        self.loss_history['joint_loss'].append(running_joint_loss / len(batch_generator))     
        self.loss_history['smpl_joint_loss'].append(running_smpl_joint_loss / len(batch_generator))     
        self.loss_history['proj_loss'].append(running_proj_loss / len(batch_generator)) 
        self.loss_history['pose_param_loss'].append(running_pose_param_loss / len(batch_generator)) 
        self.loss_history['shape_param_loss'].append(running_shape_param_loss / len(batch_generator)) 
        self.loss_history['prior_loss'].append(running_prior_loss / len(batch_generator)) 
        
        logger.info(f'Epoch{epoch} Loss: {self.loss_history["total_loss"][-1]:.4f}')


class Tester:
    def __init__(self, args, load_dir=''):
        if load_dir != '':
            self.model, _ = prepare_network(args, load_dir, False)
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)

        dataset_list, self.val_loader = get_dataloader(cfg.DATASET.test_list, is_train=False)
        if dataset_list is not None:
            self.val_dataset = dataset_list[0]
            self.val_loader = self.val_loader[0]
            self.dataset_length = len(self.val_dataset)
            
            if self.val_dataset.joint_set['name'] == '3DPW' or self.val_dataset.joint_set['name'] == 'AGORA':
                self.eval_mpvpe = True
            else:
                self.eval_mpvpe = False
        
        self.J_regressor = torch.from_numpy(smpl.h36m_joint_regressor).float().cuda()

        self.print_freq = cfg.TRAIN.print_freq
        self.vis_freq = cfg.TEST.vis_freq
        
        if cfg.MODEL.type == '2d_joint':
            self.test = self.test_2d_joint
            self.pck = 9999
        elif cfg.MODEL.type == 'body':
            self.test = self.test_body
            self.mpjpe = 9999
            self.pa_mpjpe = 9999
            self.mpvpe = 9999
        elif cfg.MODEL.type == 'hand':
            self.test = self.test_hand
            
    def test_2d_joint(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        
        pck = []
        pck_cnt = []
        
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                inp_img = batch['img'].cuda()
                tar_hm = batch['hm'].cuda()
                batch_size = inp_img.shape[0]

                # feed-forward
                output = self.model(inp_img)
                
                input_flipped = inp_img.flip(3)
                output_flipped = self.model(input_flipped)

                output_flipped = flip_back(output_flipped.cpu().numpy(), self.val_dataset.joint_set['flip_pairs'])
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
                output = (output + output_flipped) * 0.5

                acc, avg_acc, cnt, pred = self.eval_2d_accuracy(output, tar_hm)
                pck.append(avg_acc*cnt)
                pck_cnt.append(cnt)
                pck_i = avg_acc

                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => PCK: {pck_i:.3f}')
                
                if cfg.TEST.vis and (i % self.vis_freq == 0):
                    import cv2
                    from vis_utils import vis_keypoints_with_skeleton
                    pred_heatmap = output.cpu().numpy()
                    pred_joint_img, pred_joint_valid = get_max_preds(pred_heatmap)
                    pred_joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                    pred_joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]

                    tar_heatmap = tar_hm.cpu().numpy()
                    tar_joint_img, tar_joint_valid = get_max_preds(tar_heatmap)
                    tar_joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                    tar_joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]

                    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                    img = inv_normalize(inp_img[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                    img = np.ascontiguousarray(img, dtype=np.uint8)
                    
                    tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([pred_joint_img[0],pred_joint_valid[0,:]],1), coco.skeleton)
                    cv2.imwrite(osp.join(cfg.vis_dir, f'test_{i}_joint_img_pred.png'), tmp_img)

                    tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([tar_joint_img[0],tar_joint_valid[0,:]],1), coco.skeleton)
                    cv2.imwrite(osp.join(cfg.vis_dir, f'test_{i}_joint_img_gt.png'), tmp_img)

            self.pck = sum(pck) / sum(pck_cnt)
            logger.info(f'>> {eval_prefix} PCK: {self.pck:.3f}')


    def test_body(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        
        mpjpe, pa_mpjpe, mpvpe = [], [], []
        error_x, error_y, error_z = [], [], []

        error_list = []

        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                inp_img = batch['img'].cuda()
                batch_size = inp_img.shape[0]
                
                # feed-forward
                pred_mesh_cam, pred_joint_cam, pred_joint_proj, pred_smpl_pose, pred_smpl_shape, _ = self.model(inp_img)
                # meter to milimeter
                pred_mesh_cam, pred_joint_cam = pred_mesh_cam * 1000, pred_joint_cam * 1000

                # eval post processing
                pred_joint_cam = torch.matmul(self.J_regressor[None, :, :], pred_mesh_cam)
                pred_joint_cam = pred_joint_cam.cpu().numpy()
                tar_joint_cam = batch['joint_cam'].cpu().numpy()
                pred_mesh_cam = pred_mesh_cam.cpu().numpy()
                tar_mesh_cam = batch['mesh_cam'].cpu().numpy()
                
                mpjpe_i, pa_mpjpe_i = self.eval_3d_joint(pred_joint_cam, tar_joint_cam)
                #error_x_i, error_y_i, error_z_i = self.eval_xyz_joint(pred_joint_cam, tar_joint_cam)
                #error_x.extend(error_x_i); error_y.extend(error_y_i); error_z.extend(error_z_i)

                error_list.append(mpjpe_i[0])
                mpjpe.extend(mpjpe_i); pa_mpjpe.extend(pa_mpjpe_i)
                mpjpe_i, pa_mpjpe_i = sum(mpjpe_i)/batch_size, sum(pa_mpjpe_i)/batch_size
    

                if self.eval_mpvpe:
                    mpvpe_i = self.eval_mesh(pred_mesh_cam, tar_mesh_cam, pred_joint_cam, tar_joint_cam)
                    mpvpe.extend(mpvpe_i)
                    mpvpe_i = sum(mpvpe_i)/batch_size
                                
                if i % self.print_freq == 0:
                    if self.eval_mpvpe:
                        loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => MPJPE: {mpjpe_i:.2f}, PA-MPJPE: {pa_mpjpe_i:.2f} MPVPE: {mpvpe_i:.2f}')
                    else:
                        loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => MPJPE: {mpjpe_i:.2f}, PA-MPJPE: {pa_mpjpe_i:.2f}')
                    
                if cfg.TEST.vis:
                    import cv2
                    from vis_utils import vis_3d_pose, save_obj
                    
                    if True:
                        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                        img = inv_normalize(inp_img[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                        img = np.ascontiguousarray(img, dtype=np.uint8)
                        cv2.imwrite(osp.join(cfg.vis_dir, f'test_{i}_img.png'), img)
                        
                        #vis_3d_pose(pred_joint_cam[0], smpl.h36m_skeleton, 'human36', osp.join(cfg.vis_dir, f'test_{i}_joint_cam_pred.png'))
                        #vis_3d_pose(tar_joint_cam[0], smpl.h36m_skeleton, 'human36', osp.join(cfg.vis_dir, f'test_{i}_joint_cam_gt.png'))
                        
                        save_obj(pred_mesh_cam[0], smpl.face, osp.join(cfg.vis_dir, f'test_{i}_mesh_cam_pred.obj'))
                        #save_obj(tar_mesh_cam[0], smpl.face, osp.join(cfg.vis_dir, f'test_{i}_mesh_cam_gt.obj'))

                        import copy
                        gt_bbox = batch['bbox'][0].cpu().numpy()
                        focal = copy.copy(cfg.CAMERA['focal'])
                        princpt = copy.copy(cfg.CAMERA['princpt'])
                        
                        cam_param = {'focal': focal, 'princpt': princpt}
                        
                        pred_rendered_img = render_mesh(img, pred_mesh_cam[0]/1000, smpl.face, cam_param)
                        cv2.imwrite(osp.join(cfg.vis_dir, f'test_{i}_render.png'), pred_rendered_img)

            with open('4.txt', 'w') as f:
                for d in error_list:
                    f.write(f'{d:.2f}\n')

                       
            self.mpjpe = sum(mpjpe) / self.dataset_length
            self.pa_mpjpe = sum(pa_mpjpe) / self.dataset_length
            self.mpvpe = sum(mpvpe) / self.dataset_length
            #self.mpjpe_x = sum(error_x) / self.dataset_length
            #self.mpjpe_y = sum(error_y) / self.dataset_length
            #self.mpjpe_z = sum(error_z) / self.dataset_length
            
            if self.eval_mpvpe:
                logger.info(f'>> {eval_prefix} MPJPE: {self.mpjpe:.2f}, PA-MPJPE: {self.pa_mpjpe:.2f} MPVPE: {self.mpvpe:.2f}')
            else:
                logger.info(f'>> {eval_prefix} MPJPE: {self.mpjpe:.2f}, PA-MPJPE: {self.pa_mpjpe:.2f}')
            
            #logger.info(f'>> {eval_prefix} MPJPE_X: {self.mpjpe_x:.2f}, MPJPE_Y: {self.mpjpe_y:.2f}, MPJPE_Z: {self.mpjpe_z:.2f}')


    def test_hand(self, epoch, current_model=None):
        pass


    def save_history(self, loss_history, error_history, epoch):
        if cfg.MODEL.type == 'contrastive':
            save_plot(loss_history['contrast_loss'], epoch, title='Contrast Loss')
        
        elif cfg.MODEL.type == 'body':
            error_history['mpjpe'].append(self.mpjpe)
            error_history['pa_mpjpe'].append(self.pa_mpjpe)
            error_history['mpvpe'].append(self.mpvpe)

            save_plot(error_history['mpjpe'], epoch, title='MPJPE', show_min=True)
            save_plot(error_history['pa_mpjpe'], epoch, title='PA-MPJPE', show_min=True)
            save_plot(error_history['mpvpe'], epoch, title='MPVPE', show_min=True)
            
            save_plot(loss_history['joint_loss'], epoch, title='Joint Loss')
            save_plot(loss_history['smpl_joint_loss'], epoch, title='SMPL Joint Loss')
            save_plot(loss_history['proj_loss'], epoch, title='Joint Proj Loss')
            save_plot(loss_history['pose_param_loss'], epoch, title='Pose Param Loss')
            save_plot(loss_history['shape_param_loss'], epoch, title='Shape Param Loss')
            save_plot(loss_history['prior_loss'], epoch, title='Prior Loss')
        
        save_plot(loss_history['total_loss'], epoch, title='Total Loss')
        
    
    '''def eval_2d_joint(self, pred, target, target_valid):
        pred, target = pred.copy(), target.copy()
        batch_size = pred.shape[0]
        
        pck = []
        for j in range(batch_size):
            pred_i, target_i, target_val_i = pred[j], target[j], target_valid[j]
            pred_i, target_i = pred_i[target_val_i>0], target_i[target_val_i>0]
            pck.append(eval_2d_joint_accuracy(pred_i[None,...], target_i[None,...], cfg.MODEL.input_img_shape)[1])
            
        return pck'''

    def eval_2d_accuracy(self, pred, target):
        '''
        Calculate accuracy according to PCK,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        '''
        output = pred[:, coco.eval_joints].cpu().numpy()
        target = target[:, coco.eval_joints].cpu().numpy()
        hm_type = 'gaussian'
        thr = 0.5

        idx = list(range(output.shape[1]))
        norm = 1.0
        if hm_type == 'gaussian':
            pred, _ = get_max_preds(output)
            target, _ = get_max_preds(target)
            h = output.shape[2]
            w = output.shape[3]
            norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        
        dists = calc_dists(pred, target, norm)

        acc = np.zeros((len(idx) + 1))
        avg_acc = 0
        cnt = 0

        for i in range(len(idx)):
            acc[i + 1] = dist_acc(dists[idx[i]])
            if acc[i + 1] >= 0:
                avg_acc = avg_acc + acc[i + 1]
                cnt += 1

        avg_acc = avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
            acc[0] = avg_acc
            
        return acc, avg_acc, cnt, pred



    def eval_3d_joint(self, pred, target):
        pred, target = pred.copy(), target.copy()
        batch_size = pred.shape[0]
        
        pred, target = pred - pred[:, None, smpl.h36m_root_joint_idx, :], target - target[:, None, smpl.h36m_root_joint_idx, :]
        pred, target = pred[:, smpl.h36m_eval_joints, :], target[:, smpl.h36m_eval_joints, :]
        
        mpjpe, pa_mpjpe = [], []
        for j in range(batch_size):
            mpjpe.append(eval_mpjpe(pred[j], target[j]))
            pa_mpjpe.append(eval_pa_mpjpe(pred[j], target[j]))
        
        return mpjpe, pa_mpjpe

    def eval_xyz_joint(self, pred, target):
        pred, target = pred.copy(), target.copy()
        batch_size = pred.shape[0]
        
        pred, target = pred - pred[:, None, smpl.h36m_root_joint_idx, :], target - target[:, None, smpl.h36m_root_joint_idx, :]
        pred, target = pred[:, smpl.h36m_eval_joints, :], target[:, smpl.h36m_eval_joints, :]
        
        error_x, error_y, error_z = [], [], []
        for j in range(batch_size):
            err_x = np.mean(np.sqrt(( pred[j][:,0] - target[j][:,0]) ** 2))
            err_y = np.mean(np.sqrt(( pred[j][:,1] - target[j][:,1]) ** 2))
            err_z = np.mean(np.sqrt(( pred[j][:,2] - target[j][:,2]) ** 2))
            error_x.append(err_x); error_y.append(err_y); error_z.append(err_z)
        
        return error_x, error_y, error_z 
    
    
    def eval_mesh(self, pred, target, pred_joint_cam, gt_joint_cam):
        pred, target = pred.copy(), target.copy()
        batch_size = pred.shape[0]
        
        pred, target = pred - pred_joint_cam[:, None, smpl.h36m_root_joint_idx, :], target - gt_joint_cam[:, None, smpl.h36m_root_joint_idx, :]
     
        mpvpe = []
        for j in range(batch_size):
            mpvpe.append(eval_mpjpe(pred[j], target[j]))
        
        return mpvpe