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

import MSCOCO.dataset, MPII.dataset, PW3D.dataset, Human36M.dataset, MPI_INF_3DHP.dataset
from models import get_model, transfer_backbone
from multiple_datasets import MultipleDatasets
from core.loss import get_loss
from utils.coord_utils import heatmap_to_coords
from utils.funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters
from utils.eval_utils import eval_mpjpe, eval_pa_mpjpe, eval_2d_joint_accuracy
from utils.vis_utils import save_plot
from utils.human_models import smpl


def get_dataloader(dataset_names, is_train):
    if len(dataset_names) == 0: return None, None

    dataset_split = 'TRAIN' if is_train else 'TEST'  
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    logger.info(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        dataset = eval(f'{name}.dataset')(transform, dataset_split.lower())
        logger.info(f"# of {dataset_split} {name} data: {len(dataset)}")
        dataset_list.append(dataset)

    if not is_train:
        for dataset in dataset_list:
            dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=True,
                                drop_last=False)
            dataloader_list.append(dataloader)
        
        return dataset_list, dataloader_list
    else:
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        trainset_loader = MultipleDatasets(dataset_list, partition=cfg.DATASET.train_partition, make_same_len=cfg.DATASET.make_same_len)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=batch_per_dataset * len(dataset_names), shuffle=cfg[dataset_split].shuffle,
                                     num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
        return dataset_list, batch_generator


def prepare_network(args, load_dir='', is_train=True):
    model, checkpoint = None, None

    criterion = get_loss()
    model = get_model(is_train, criterion)

    logger.info(f"==> Constructing Model...")
    logger.info(f"# of model parameters: {count_parameters(model)}")
    logger.info(model)
    
    if load_dir and (not is_train or args.resume_training):
        logger.info(f"==> Loading checkpoint: {load_dir}")
        checkpoint = load_checkpoint(load_dir=load_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif load_dir and cfg.TRAIN.transfer_backbone and is_train:
        logger.info(f"==> Transfer from checkpoint: {load_dir}")
        checkpoint = load_checkpoint(load_dir=load_dir)
        transfer_backbone(model, checkpoint['model_state_dict'])
        checkpoint = None
        
    return model, checkpoint


def train_setup(model, checkpoint):    
    criterion, optimizer, lr_scheduler = None, None, None
    if cfg.MODEL.type == 'contrastive':
        loss_history = {'total_loss': [], 'inter_joint_loss': [], 'intra_joint_loss': []}
        error_history = {'contrastive_loss': []}
    elif cfg.MODEL.type == '2d_joint':
        loss_history = {'total_loss': [], 'hm_loss': []}
        error_history = {'pck': []}
    elif cfg.MODEL.type == 'body':
        loss_history = {'total_loss': [], 'joint_loss': [], 'smpl_joint_loss': [], 'proj_loss': [], 'pose_param_loss': [], 'shape_param_loss': [], 'prior_loss': []}
        error_history = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
    
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

    return optimizer, lr_scheduler, loss_history, error_history
    

class Trainer:
    def __init__(self, args, load_dir):
        self.model, checkpoint = prepare_network(args, load_dir, True)
        self.optimizer, self.lr_scheduler, self.loss_history, self.error_history = train_setup(self.model, checkpoint)
        dataset_list, self.batch_generator = get_dataloader(cfg.DATASET.train_list, is_train=True)

        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model) 
        self.print_freq = cfg.TRAIN.print_freq
        
        if cfg.MODEL.type == 'contrastive':
            self.train = self.train_contrastive
            self.inter_joint_loss_weight = cfg.TRAIN.inter_joint_loss_weight
            self.intra_joint_loss_weight = cfg.TRAIN.intra_joint_loss_weight
        elif cfg.MODEL.type == '2d_joint':
            self.train = self.train_2d_joint
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

        running_loss = 0.0
        running_inter_joint_loss = 0.0
        running_intra_joint_loss = 0.0
        
        batch_generator = tqdm(self.batch_generator)
        for i, batch in enumerate(batch_generator):
            loss = self.model(batch)
            loss = {k: loss[k].mean() for k in loss}
            tot_loss = sum(loss[k] for k in loss)
            
            # update weights
            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()

            # log
            running_loss += float(tot_loss.detach().item())
            running_inter_joint_loss += float(loss['inter_joint'].detach().item())
            running_intra_joint_loss += float(loss['intra_joint'].detach().item())

            if i % self.print_freq == 0:
                loss['tot_loss'] = tot_loss
                loss_message = ' '.join(['%s: %.4f' % (k, v.detach()) for k, v in loss.items()])
                batch_generator.set_description(f'Epoch{epoch} ({i}/{len(batch_generator)}) => {loss_message}')

        self.loss_history['total_loss'].append(running_loss / len(batch_generator))
        self.loss_history['inter_joint_loss'].append(running_inter_joint_loss / len(batch_generator))
        self.loss_history['intra_joint_loss'].append(running_intra_joint_loss / len(batch_generator))        
            
        logger.info(f'Epoch{epoch} Loss: {self.loss_history["total_loss"][-1]:.4f}')
        
    def train_2d_joint(self, epoch):
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]

        running_loss = 0.0
        running_hm_loss = 0.0
        
        batch_generator = tqdm(self.batch_generator)
        for i, batch in enumerate(batch_generator):
            inp_img = batch['img'].cuda()
            tar_heatmap = batch['hm'].cuda()
            meta_hm_valid = batch['hm_valid'].cuda()
            
            pred_heatmap = self.model(inp_img)

            loss1 = self.loss['hm'](pred_heatmap, tar_heatmap, meta_hm_valid)
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
            if cfg.TRAIN.vis and i % (len(batch_generator)//10) == 0:
                import cv2
                from utils.vis_utils import vis_keypoints_with_skeleton, vis_heatmaps
                
                inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                img = inv_normalize(inp_img[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                img = np.ascontiguousarray(img, dtype=np.uint8)
                
                pred_heatmap = pred_heatmap.cpu().detach().numpy()
                tar_heatmap = tar_heatmap.cpu().detach().numpy()
                meta_hm_valid = meta_hm_valid.cpu().numpy()

                pred_joint_img, pred_joint_valid = heatmap_to_coords(pred_heatmap)
                pred_joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                pred_joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]

                tar_joint_img, _ = heatmap_to_coords(tar_heatmap)
                tar_joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                tar_joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]
                
                tmp_img = vis_heatmaps(img[None,...], pred_heatmap[0, None,...])
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_hm_pred.png'), tmp_img)

                tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([pred_joint_img[0],pred_joint_valid[0,:, None]],1), smpl.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_pred.png'), tmp_img)

                tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([tar_joint_img[0],meta_hm_valid[0,:, None]],1), smpl.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_gt.png'), tmp_img)
    

        self.loss_history['total_loss'].append(running_loss / len(batch_generator)) 
        self.loss_history['hm_loss'].append(running_hm_loss / len(batch_generator))     
        logger.info(f'Epoch{epoch} Loss: {self.loss_history["total_loss"][-1]:.4f}')

        
    def train_body(self, epoch):
        self.model.train()

        running_loss = 0.0
        running_joint_loss = 0.0
        running_smpl_joint_loss = 0.0
        running_proj_loss = 0.0
        running_pose_param_loss = 0.0
        running_shape_param_loss = 0.0
        running_prior_loss = 0.0
        
        batch_generator = tqdm(self.batch_generator)
        for i, batch in enumerate(batch_generator):
            loss = self.model(batch)
            loss = {k: loss[k].mean() for k in loss}
            tot_loss = sum(loss[k] for k in loss)

            # update weights
            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()

            # log
            running_loss += float(tot_loss.detach().item())
            running_joint_loss += float(loss['joint_cam'].detach().item())
            running_smpl_joint_loss += float(loss['smpl_joint_cam'].detach().item())
            running_proj_loss += float(loss['joint_proj'].item())
            running_pose_param_loss += float(loss['pose_param'].item())
            running_shape_param_loss += float(loss['shape_param'].item())
            running_prior_loss += float(loss['prior'].item())
            
            if i % self.print_freq == 0:
                loss['tot_loss'] = tot_loss
                loss_message = ' '.join(['%s: %.4f' % (k, v.detach()) for k,v in loss.items()])
                batch_generator.set_description(f'Epoch{epoch} ({i}/{len(batch_generator)}) => {loss_message}')

        self.loss_history['total_loss'].append(running_loss / len(batch_generator)) 
        self.loss_history['joint_loss'].append(running_joint_loss / len(batch_generator))     
        self.loss_history['smpl_joint_loss'].append(running_smpl_joint_loss / len(batch_generator))     
        self.loss_history['proj_loss'].append(running_proj_loss / len(batch_generator)) 
        self.loss_history['pose_param_loss'].append(running_pose_param_loss / len(batch_generator)) 
        self.loss_history['shape_param_loss'].append(running_shape_param_loss / len(batch_generator)) 
        self.loss_history['prior_loss'].append(running_prior_loss / len(batch_generator)) 
        
        logger.info(f'Epoch{epoch} Loss: {self.loss_history["total_loss"][-1]:.4f}')

    def train_hand(self, epoch):
        pass
    

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
            
            if self.val_dataset.joint_set['name'] == '3DPW':
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
        
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                inp_img = batch['img'].cuda()
                batch_size = inp_img.shape[0]

                # feed-forward
                pred_heatmap = self.model(inp_img)
                pred_heatmap = pred_heatmap.cpu().numpy()
                pred_joint_img, pred_joint_valid = heatmap_to_coords(pred_heatmap)
                pred_joint_img[:,:,0] *= cfg.MODEL.input_img_shape[1] / cfg.MODEL.img_feat_shape[1]
                pred_joint_img[:,:,1] *= cfg.MODEL.input_img_shape[0] / cfg.MODEL.img_feat_shape[0]
                tar_joint_img = batch['joint_img'].cpu().numpy()
                meta_joint_valid = batch['joint_valid'].cpu().numpy()

                pck_i = self.eval_2d_joint(pred_joint_img, tar_joint_img, meta_joint_valid)
                
                pck.extend(pck_i)
                pck_i = sum(pck_i)/batch_size
                
                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => PCK: {pck_i:.3f}')
                
                if cfg.TEST.vis:
                    import cv2
                    from utils.vis_utils import vis_keypoints_with_skeleton
                    
                    if i % self.vis_freq == 0:
                        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                        img = inv_normalize(inp_img[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                        img = np.ascontiguousarray(img, dtype=np.uint8)
                        cv2.imwrite(osp.join(cfg.vis_dir, f'test_{i}_img.png'), img)
                        
                        tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([pred_joint_img[0],pred_joint_valid[0,:, None]],1), smpl.skeleton)
                        cv2.imwrite(osp.join(cfg.vis_dir, f'test_{i}_joint_img_pred.png'), tmp_img)

                        tmp_img = vis_keypoints_with_skeleton(img, np.concatenate([tar_joint_img[0],meta_joint_valid[0,:, None]],1), smpl.skeleton)
                        cv2.imwrite(osp.join(cfg.vis_dir, f'train_{i}_joint_img_gt.png'), tmp_img)

            self.pck = sum(pck) / self.dataset_length
            logger.info(f'>> {eval_prefix} PCK: {self.pck:.3f}')


    def test_body(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        
        mpjpe, pa_mpjpe, mpvpe = [], [], []
        
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                inp_img = batch['img'].cuda()
                batch_size = inp_img.shape[0]

                # feed-forward
                pred_mesh_cam, pred_joint_cam, pred_joint_proj, pred_smpl_pose, pred_smpl_shape = self.model(inp_img)
                # meter to milimeter
                pred_mesh_cam, pred_joint_cam = pred_mesh_cam * 1000, pred_joint_cam * 1000

                # eval post processing
                pred_joint_cam = torch.matmul(self.J_regressor[None, :, :], pred_mesh_cam)
                pred_joint_cam = pred_joint_cam.cpu().numpy()
                tar_joint_cam = batch['joint_cam'].cpu().numpy()
                pred_mesh_cam = pred_mesh_cam.cpu().numpy()
                tar_mesh_cam = batch['mesh_cam'].cpu().numpy()
                
                mpjpe_i, pa_mpjpe_i = self.eval_3d_joint(pred_joint_cam, tar_joint_cam)
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
                    from utils.vis_utils import vis_3d_pose, save_obj
                    
                    if i % self.vis_freq == 0:
                        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                        img = inv_normalize(inp_img[0]).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                        img = np.ascontiguousarray(img, dtype=np.uint8)
                        cv2.imwrite(osp.join(cfg.vis_dir, f'test_{i}_img.png'), img)
                        
                        vis_3d_pose(pred_joint_cam[0], smpl.h36m_skeleton, 'human36', osp.join(cfg.vis_dir, f'test_{i}_joint_cam_pred.png'))
                        vis_3d_pose(tar_joint_cam[0], smpl.h36m_skeleton, 'human36', osp.join(cfg.vis_dir, f'test_{i}_joint_cam_gt.png'))
                        
                        save_obj(pred_mesh_cam[0], smpl.face, osp.join(cfg.vis_dir, f'test_{i}_mesh_cam_pred.obj'))
                        save_obj(tar_mesh_cam[0], smpl.face, osp.join(cfg.vis_dir, f'test_{i}_mesh_cam_gt.obj'))
                       
            self.mpjpe = sum(mpjpe) / self.dataset_length
            self.pa_mpjpe = sum(pa_mpjpe) / self.dataset_length
            self.mpvpe = sum(mpvpe) / self.dataset_length
            
            if self.eval_mpvpe:
                logger.info(f'>> {eval_prefix} MPJPE: {self.mpjpe:.2f}, PA-MPJPE: {self.pa_mpjpe:.2f} MPVPE: {self.mpvpe:.2f}')
            else:
                logger.info(f'>> {eval_prefix} MPJPE: {self.mpjpe:.2f}, PA-MPJPE: {self.pa_mpjpe:.2f}')


    def test_hand(self, epoch, current_model=None):
        pass


    def save_history(self, loss_history, error_history, epoch):
        if cfg.MODEL.type == 'contrastive':
            save_plot(loss_history['inter_joint_loss'], epoch, title='Inter Joint Loss')
            save_plot(loss_history['inter_joint_loss'], epoch, title='Intra Joint Loss')
        
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
        
    
    def eval_2d_joint(self, pred, target, target_valid):
        pred, target = pred.copy(), target.copy()
        batch_size = pred.shape[0]
        
        pck = []
        for j in range(batch_size):
            pred_i, target_i, target_val_i = pred[j], target[j], target_valid[j]
            pred_i, target_i = pred_i[target_val_i>0], target_i[target_val_i>0]
            pck.append(eval_2d_joint_accuracy(pred_i[None,...], target_i[None,...], cfg.MODEL.input_img_shape)[1])
            
        return pck


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
    
    
    def eval_mesh(self, pred, target, pred_joint_cam, gt_joint_cam):
        pred, target = pred.copy(), target.copy()
        batch_size = pred.shape[0]
        
        pred, target = pred - pred_joint_cam[:, None, smpl.h36m_root_joint_idx, :], target - gt_joint_cam[:, None, smpl.h36m_root_joint_idx, :]
        pred, target = pred[:, smpl.h36m_eval_joints, :], target[:, smpl.h36m_eval_joints, :]
        
        mpvpe = []
        for j in range(batch_size):
            mpvpe.append(eval_mpjpe(pred[j], target[j]))
        
        return mpvpe