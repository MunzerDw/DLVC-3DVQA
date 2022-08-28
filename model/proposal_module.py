"""
Modified from https://github.com/zlccccc/3DVG-Transformer/blob/main/models/proposal_module.py
and from
https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)   # DETR

from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from easydict import EasyDict
from model.detr.detr3d import DETR3D
from model.detr.transformer3D import decode_scores_boxes
from utils.box_util import get_3d_box_batch

from data.scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()

# Calculate sem. class score and objectness score from proposal predictions
def decode_scores_classes(output_dict, end_points, num_class):
    pred_logits = output_dict['pred_logits']
    assert pred_logits.shape[-1] == 2+num_class, 'pred_logits.shape wrong'
    objectness_scores = pred_logits[:,:,0:2]  # TODO CHANGE IT; JUST SOFTMAXd
    end_points['objectness_scores'] = objectness_scores
    sem_cls_scores = pred_logits[:,:,2:2+num_class] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points

# Calculate features and bounding boxes of predicted proposals
def decode_dataset_config(data_dict, dataset_config):
    if dataset_config is not None:
        # print('decode_dataset_config', flush=True)
        pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
                                             pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
                                          pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
        batch_size = pred_center.shape[0]
        pred_obbs, pred_bboxes = [], []
        for i in range(batch_size):
            pred_obb_batch = dataset_config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i],
                                                            pred_heading_residual[i],
                                                            pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            pred_obbs.append(torch.from_numpy(pred_obb_batch))
            pred_bboxes.append(torch.from_numpy(pred_bbox_batch))
            # print(pred_obb_batch.shape, pred_bbox_batch.shape)
        data_dict['pred_obbs'] = torch.stack(pred_obbs, dim=0).cuda()
        data_dict['pred_bboxes'] = torch.stack(pred_bboxes, dim=0).cuda()
        data_dict['bbox_corner'] = torch.stack(pred_bboxes, dim=0).cuda()
        data_dict["bbox_feature"] = data_dict["aggregated_vote_features"]
        data_dict["bbox_mask"] = data_dict['objectness_scores'].argmax(-1)
        data_dict['bbox_sems'] = data_dict['sem_cls_scores'].argmax(-1) # # B x num_proposal
    return data_dict

# Call decoding functions
def decode_scores(output_dict, end_points,  num_class, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias=False, quality_channel=False, dataset_config=None):
    end_points = decode_scores_classes(output_dict, end_points, num_class)
    end_points = decode_scores_boxes(output_dict, end_points, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias, quality_channel)
    end_points = decode_dataset_config(end_points, dataset_config)
    return end_points

## Extended Proposal Module, computes refined object proposals w.r.t. spatial proximity
class ProposalModuleSpatial(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, config_transformer=None, quality_channel=False, dataset_config=None):
        super().__init__()
        if config_transformer is None:
            raise NotImplementedError('You should input a config')
            config_transformer = {
                'mask': 'near_5',
                'weighted_input': True,
                'transformer_type': 'deformable',
                'deformable_type': 'interpolation',
                'position_embedding': 'none',
                'input_dim': 0,
                'enc_layers': 0,
                'dec_layers': 4,
                'dim_feedforward': 2048,
                'hidden_dim': 288,
                'dropout': 0.1,
                'nheads': 8,
                'pre_norm': False
            }
        config_transformer = EasyDict(config_transformer)
        # print(config_transformer, '<< config transformer ')

        transformer_type = config_transformer.get('transformer_type', 'enc_dec')
        position_type = config_transformer.get('position_type', 'vote')
        self.transformer_type = transformer_type
        self.position_type = position_type
        self.center_with_bias = 'dec' not in transformer_type

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.quality_channel = quality_channel
        self.dataset_config = dataset_config

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        # JUST FOR

        if self.position_type == 'seed_attention':
            self.seed_feature_trans = torch.nn.Sequential(
                torch.nn.Conv1d(256, 128, 1),
                torch.nn.BatchNorm1d(128),
                torch.nn.PReLU(128)
            )

        self.detr = DETR3D(config_transformer, input_channels=128, class_output_shape=2+num_class, bbox_output_shape=3+num_heading_bin*2+num_size_cluster*4+int(quality_channel))

    def forward(self, xyz, features, end_points):  # initial_xyz and xyz(voted): just for encoding
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        seed_xyz, seed_features = end_points['seed_xyz'], features
        xyz, features, fps_inds = self.vote_aggregation(xyz, features) #  batch, votenet_mlp_size (128), 256
        sample_inds = fps_inds

        end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ----------  TODO PROPOSAL GENERATION AND CHANGE LOSS GENERATION
        # print(features.mean(), features.std(), ' << first,votenet forward features mean and std', flush=True) # TODO CHECK IT
        features = F.relu(self.bn1(self.conv1(features)))
        features = F.relu(self.bn2(self.conv2(features)))

        # print(features.mean(), features.std(), ' << votenet forward features mean and std', flush=True) # TODO CHECK IT

        # _xyz = torch.gather(initial_xyz, 1, sample_inds.long().unsqueeze(-1).repeat(1,1,3))
        # print(initial_xyz.shape, xyz.shape, sample_inds.shape, _xyz.shape, '<< sample xyz shape', flush=True)
        features = features.permute(0, 2, 1)
        # print(xyz.shape, features.shape, '<< detr input feature dim')
        if self.position_type == 'vote':
            output_dict = self.detr(xyz, features, end_points)
            end_points['detr_features'] = output_dict['detr_features']
        elif self.position_type == 'seed_attention':
            decode_vars = {
                'num_class': self.num_class, 
                'num_heading_bin': self.num_heading_bin,
                'num_size_cluster': self.num_size_cluster, 
                'mean_size_arr': self.mean_size_arr,
                'aggregated_vote_xyz': xyz
            }
            seed_features = self.seed_feature_trans(seed_features)
            seed_features = seed_features.permute(0, 2, 1).contiguous()
            output_dict = self.detr(xyz, features, end_points, seed_xyz=seed_xyz, seed_features=seed_features, decode_vars=decode_vars)
        else:
            raise NotImplementedError(self.position_type)
        # output_dict = self.detr(xyz, features, end_points)
        end_points = decode_scores(output_dict, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr,
                                   self.center_with_bias, quality_channel=self.quality_channel, dataset_config=self.dataset_config)

        return end_points


class ProposalModuleStandard(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, proposal_size=128, radius=0.3, nsample=16):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.votenet_hidden_size = proposal_size

        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.num_proposal,
            radius=radius, # 0.3 (scanrefer, votenet), 5 (scan2cap)
            nsample=nsample, # 16 (scanrefer, votenet), 512 (scan2cap)
            mlp=[self.seed_feat_dim, proposal_size, proposal_size, proposal_size],
            use_xyz=True,
            normalize_xyz=True
        )
            
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)

        self.proposal = nn.Sequential(
            nn.Conv1d(proposal_size,proposal_size,1, bias=False),
            nn.BatchNorm1d(proposal_size),
            nn.ReLU(),
            nn.Conv1d(proposal_size,proposal_size,1, bias=False),
            nn.BatchNorm1d(proposal_size),
            nn.ReLU(),
            nn.Conv1d(proposal_size,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        )

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # Farthest point sampling (FPS) on votes
        # feturea: batch, 256, 1024
        xyz, features, fps_inds = self.vote_aggregation(xyz, features) #  batch, votenet_mlp_size (128), 256
        
        sample_inds = fps_inds
        data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features)
        
        # net: batch, ???, num_proposals (32, 97, 256)
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_pred_box(self, data_dict):
        # predicted bbox
        pred_center = data_dict["center"].detach().cpu().numpy() # (B,K,3)
        pred_heading_class = torch.argmax(data_dict["heading_scores"], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(data_dict["size_scores"], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

        batch_size, num_proposals, _ = pred_center.shape
        pred_bboxes = []
        for i in range(batch_size):
            # convert the bbox parameters to bbox corners
            pred_obb_batch = DC.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                        pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            pred_bboxes.append(torch.from_numpy(pred_bbox_batch).cuda().unsqueeze(0))
        pred_bboxes = torch.cat(pred_bboxes, dim=0) # batch_size, num_proposals, 8, 3
        return pred_bboxes

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        #net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, num_proposal, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2]

        base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)

        heading_scores = net_transposed[:,:,5:5+num_heading_bin]
        heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
        
        size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        
        sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        data_dict['heading_scores'] = heading_scores # B x num_proposal x num_heading_bin
        data_dict['heading_residuals_normalized'] = heading_residuals_normalized # B x num_proposal x num_heading_bin (should be -1 to 1)
        data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # B x num_proposal x num_heading_bin
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores # B x num_proposal x 10

        # processed box info
        data_dict["bbox_corner"] = self.decode_pred_box(data_dict) # batch_size, num_proposals, 8, 3 (bounding box corner coordinates)
        data_dict["bbox_feature"] = data_dict["aggregated_vote_features"]
        data_dict["bbox_mask"] = objectness_scores.argmax(-1)
        data_dict['bbox_sems'] = sem_cls_scores.argmax(-1) # # B x num_proposal
        #data_dict['sem_cls'] = sem_cls_scores.argmax(-1)

        return data_dict