import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import os

from model.mcan_module import MCAN_ED, AttFlat, LayerNorm
from model.voting_module import VotingModule
from model.proposal_module import ProposalModuleSpatial, ProposalModuleStandard
from model.backbone_module import Pointnet2Backbone
from model.lang_module import LangModule
from model.answer_module import AnswerModule
from model.answer_transformer import AnswerTransformer
from model.graph_module import GraphModule

from lib.eval_helper import get_eval
from lib.loss_helper import get_loss_vn

from data.scannet.model_util_scannet import ScannetDatasetConfig
from data.config import CONF

# Scannet config
DATASET_CONFIG = ScannetDatasetConfig()

class Ours(pl.LightningModule):
    def __init__(self, hparams, answer_vocab):
        super().__init__()

        # update hyperparameters
        self.hparams.update(hparams)

        self.use_answer_transformer = hparams['use_answer_transformer']

        ##########################
        ##                      ##
        ##    Question Branch   ##
        ##                      ##
        ##########################
        
        self.answer_vocab = answer_vocab
        self.num_object_class = 18
        self.bi_hidden_size = hparams['hidden_size'] * (1 + hparams['lang_use_bidir'])

        # language Bi-LSTM
        self.lang_net = LangModule(num_object_class=self.num_object_class, 
            use_lang_classifier=True, 
            use_bidir=hparams['lang_use_bidir'], 
            num_layers=1,
            hidden_size=hparams['hidden_size'],
            emb_size=hparams['emb_size']
        )  

        # language MLP
        self.lang_feat_linear = nn.Sequential(
            nn.Linear(hparams['hidden_size'] * (1 + hparams['lang_use_bidir']), hparams['hidden_size']),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.LayerNorm(hparams['hidden_size'])
        )

        #######################
        ##                   ##
        ##    Scene Branch   ##
        ##                   ##
        #######################

        """
        How proposal creation works in ScanQA:
            1. create num_proposal-proposals with PointNet++ "Hierarchical Point Set Feature Learning" Method with proposal_size features each
            2. For each proposal, compute bounding box from feature vector (and forget about it until the end)
            3. use the pontnet++ feature vectors for fusion module
            4. For each vector, predict a score of how likely the object is related to the question
            5. The feature vector with the highest score is the output
            6. when predicting the bounding boxes, use the bounding box that was computed in 2.) belonging to the highest score
            7. For loss, use the combination from all scores and all bounding boxes.
        """

        self.num_class = 18
        self.num_heading_bin = 1
        self.num_size_cluster = 18
        self.mean_size_arr = np.load(os.path.join(CONF.PATH.SCANNET, 'meta_data/scannet_reference_means.npz'))['arr_0']
        self.num_proposal = hparams['num_proposal']
        self.num_decoder_layers = 6
        self.self_position_embedding = 'loc_learned'
        self.cross_position_embedding = 'xyz_learned'
        self.nhead = 8
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.activation = 'relu'
        self.bn_momentum = 0.1
        self.input_dim = hparams['input_dim']

        # Pointnet++ backbone, input feature dim = 3 for xyz, and 6 if colors are used
        self.backbone_net = Pointnet2Backbone(
            input_feature_dim=self.input_dim, 
            width=1, 
            depth=2, 
            seed_feat_dim=288
        )

        # VoteNet params
        vote_factor=1
        sampling="vote_fps"
        proposal_size=128      
        vote_radius=0.3 
        vote_nsample=16
        seed_feat_dim = 288
        self.use_object_mask = True
        
        # Hough voting
        self.voting_net = VotingModule(vote_factor, seed_feat_dim)

        # Vote aggregation and object proposal
        if not hparams["use_standard_proposal"]:
            config_transformer = {
                'mask': 'no_mask',
                'weighted_input': True,
                'transformer_type': 'myAdd_20;deformable',
                'deformable_type': 'myAdd',
                'position_embedding': 'none',
                'input_dim': 0,
                'enc_layers': 0,
                'dec_layers': 2,
                'dim_feedforward': 2048,
                'hidden_dim': 288,
                'dropout': 0.1,
                'nheads': 8,
                'pre_norm': False
            }
            self.proposal_net = ProposalModuleSpatial(
                self.num_class, 
                self.num_heading_bin, 
                self.num_size_cluster, 
                self.mean_size_arr, 
                self.num_proposal,
                sampling, 
                seed_feat_dim=seed_feat_dim, 
                config_transformer=config_transformer, 
                dataset_config=DATASET_CONFIG
            )
        else:
            self.proposal_net = ProposalModuleStandard(
                self.num_class, 
                self.num_heading_bin, 
                self.num_size_cluster, 
                self.mean_size_arr, 
                self.num_proposal, 
                sampling, 
                seed_feat_dim=seed_feat_dim, 
                proposal_size=proposal_size,
                radius=vote_radius, nsample=vote_nsample
            )   

        # object proposals MLP
        self.object_feat_linear = nn.Sequential(
            nn.Linear(proposal_size, hparams['hidden_size']),
            nn.GELU()
        )

        #################
        ##             ##
        ##    Fusion   ##
        ##             ##
        #################

        self.answer_pdrop=0.3
        self.mcan_num_layers=2
        self.mcan_num_heads=8
        self.mcan_pdrop=0.1
        self.mcan_flat_mlp_size=512
        self.mcan_flat_glimpses=1
        self.mcan_flat_out_size=hparams['mcan_flat_out_size']

        # fusion backbone (MCAN)
        self.fusion_backbone = MCAN_ED(
            self.bi_hidden_size, 
            num_heads=self.mcan_num_heads, 
            num_layers=self.mcan_num_layers, 
            pdrop=self.mcan_pdrop
        )
        self.fusion_norm = LayerNorm(self.mcan_flat_out_size)
        self.attflat_visual = AttFlat(
            self.bi_hidden_size, 
            self.mcan_flat_mlp_size, 
            self.mcan_flat_glimpses, 
            self.mcan_flat_out_size, 
            0.1
        )
        self.attflat_lang = AttFlat(
            self.bi_hidden_size, 
            self.mcan_flat_mlp_size, 
            self.mcan_flat_glimpses, 
            self.mcan_flat_out_size, 
            0.1
        )

        ################
        ##            ##
        ##    Graph   ##
        ##            ##
        ################
        
        self.graph_module = GraphModule(hparams['hidden_size'], hparams['hidden_size'], k=6, layers=2)

        ##################
        ##              ##
        ##    Outputs   ##
        ##              ##
        ##################

        # answer module
        if self.use_answer_transformer:
            self.answer_module = AnswerTransformer(answer_vocab=answer_vocab)
        else:
            self.answer_module = AnswerModule(
                emb_size=hparams['emb_size'], 
                hidden_size=hparams['hidden_size'],
                bi_hidden_size=self.bi_hidden_size,
                mcan_flat_out_size=hparams['mcan_flat_out_size'],
                vocab_size=len(answer_vocab),
                num_layers=1,
                num_proposal=hparams['num_proposal'],
                answer_vocab=answer_vocab
            )

        # object localization MLP
        self.object_cls = nn.Sequential(
            nn.Linear(self.bi_hidden_size, self.bi_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.bi_hidden_size, 1)
        )

        # object classification MLP
        self.lang_cls = nn.Sequential(
            nn.Linear(self.mcan_flat_out_size, self.bi_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.bi_hidden_size, self.num_object_class)
        )
        
        ## Init
        self.init_weights()
        self.init_bn_momentum()
    
    # initialize weights 
    def init_weights(self):
        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    # initialize batch normalization momentum
    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    # log hyperparameters on start
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, data_dict):
        ##########################
        ##                      ##
        ##    Question Branch   ##
        ##                      ##
        ##########################
        
        # --------- QUESTION ENCODING ---------
        data_dict = self.lang_net(data_dict) 
        
        #######################
        ##                   ##
        ##    Scene Branch   ##
        ##                   ##
        #######################
        
        # --------- POINTNET++ ---------
        data_dict = self.backbone_net(data_dict)
        
        # --------- HOUGH VOTING ---------
        # query points
        xyz = data_dict['fp2_xyz']
        features = data_dict['fp2_features']
        data_dict['seed_inds'] = data_dict['fp2_inds']
        data_dict['seed_xyz'] = xyz
        data_dict['seed_features'] = features
        xyz, features = self.voting_net(xyz, features) # batch_size, vote_feature_dim, num_seed * vote_factor, (16, 256, 1024)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features 
        
        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal_net(xyz, features, data_dict)
        
        ################
        ##            ##
        ##   Fusion   ##
        ##            ##
        ################

        # unpack outputs from question encoding branch
        lang_feat = data_dict["lang_out"].detach().clone() # word embeddings after LSTM (batch_size, num_words(max_question_length), hidden_size * num_dir)
        # we detach lang_feat from the computational graph so we can use the 
        # unprocessed question sequence data_dict["lang_out"] in the attention part of our answer module
        lang_mask = data_dict["lang_mask"] # word attetion (batch, num_words)
        
        # unpack outputs from detection branch
        object_feat = data_dict['aggregated_vote_features'] # batch_size, num_proposal, proposal_size (128)
        
        if self.use_object_mask:
            object_mask = ~data_dict["bbox_mask"].bool().detach() #  # batch, num_proposals
        else:
            object_mask = None            

        if lang_mask.dim() == 2:
            lang_mask = lang_mask.unsqueeze(1).unsqueeze(2)
        if object_mask.dim() == 2:
            object_mask = object_mask.unsqueeze(1).unsqueeze(2)        

        # Pre-process Lanauge & Image Feature
        lang_feat = self.lang_feat_linear(lang_feat) # batch_size, num_words, hidden_size
        object_feat = self.object_feat_linear(object_feat) # batch_size, num_proposal, hidden_size
        
        # MCAN
        lang_feat, object_feat = self.fusion_backbone(
            lang_feat,
            object_feat,
            lang_mask,
            object_mask,
        ) 
        # object_feat: batch_size, num_proposal, hidden_size
        # lang_feat: batch_size, num_words, hidden_size
        
        # --------- GRAPH ---------
        object_feat = self.graph_module(object_feat, data_dict)
        data_dict["object_proposals"] = object_feat
        
        lang_feat_comb = self.attflat_lang(
            lang_feat,
            lang_mask
        )

        object_feat_comb = self.attflat_visual(
            object_feat,
            object_mask
        )

        # get final fused feature vector
        fuse_feat = self.fusion_norm(lang_feat_comb + object_feat_comb)
        
        data_dict['contexts'] = fuse_feat
        
        ################
        ##            ##
        ##   Ouputs   ##
        ##            ##
        ################
    
        # --------- ANSWER ---------
        if self.use_answer_transformer:
            lang_mask = lang_mask.squeeze(1).squeeze(1)
            data_dict["src"] = lang_feat
            data_dict['src_mask'] = lang_mask
        data_dict = self.answer_module(data_dict)

        # --------- OBJECT CLASSIFICATION ---------
        data_dict["lang_scores"] = self.lang_cls(fuse_feat) # batch_size, num_object_classes
        
        # --------- OBJECT LOCALIZATION ---------
        object_conf_feat = object_feat * data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)
        data_dict["cluster_ref"] = self.object_cls(object_conf_feat).squeeze(-1) # batch_size, num_proposals
        
        return data_dict

    def training_step(self, batch, batch_idx):
        # forward pass
        data_dict = self.forward(batch)
        
        # loss
        loss, data_dict, losses = get_loss_vn(data_dict, DATASET_CONFIG, vocab = self.answer_vocab)
        
        # logging
        self.log("train/loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        for key, value in losses.items():
          self.log('train/' + key, value, prog_bar=False, logger=True, on_epoch=True, on_step=False)

        self.log("train_batch/loss", loss, prog_bar=True, logger=True, on_epoch=False, on_step=True)
        for key, value in losses.items():
          self.log('train_batch/' + key, value, prog_bar=False, logger=True, on_epoch=False, on_step=True)

        self.log('all/loss', {'train': loss}, prog_bar=False, logger=True, on_epoch=True, on_step=False)
        self.log('all/answer_loss', {'train': losses['answer_loss']}, prog_bar=False, logger=True, on_epoch=True, on_step=False)
        
        return {**{'loss': loss}, **losses}
        

    def validation_step(self, batch, batch_idx):
        # forward pass
        data_dict = self.forward(batch)

        # loss
        loss, data_dict, losses = get_loss_vn(data_dict, DATASET_CONFIG)

        # accuracies
        data_dict, accuracies = get_eval(
            data_dict=data_dict,
            config=DATASET_CONFIG,
            answer_vocab=self.answer_vocab,
            use_reference=True, 
            use_lang_classifier=True
        )

        # logging
        self.log("val_batch/loss", loss, prog_bar=True, logger=True, on_epoch=False, on_step=True)
        for key, value in losses.items():
            self.log('val_batch/' + key, value, prog_bar=False, logger=True, on_epoch=False, on_step=True)

        self.log("val/loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        for key, value in losses.items():
            self.log('val/' + key, value, prog_bar=False, logger=True, on_epoch=True, on_step=False)
        for key, value in accuracies.items():
            self.log('acc/' + key, value, prog_bar=False, logger=True, on_epoch=True, on_step=False)

        self.log('all/loss', {'val': loss}, prog_bar=False, logger=True, on_epoch=True, on_step=False)
        self.log('all/answer_loss', {'val': losses['answer_loss']}, prog_bar=False, logger=True, on_epoch=True, on_step=False)

        return {**{'loss': loss}, **losses }

    def configure_optimizers(self):
        # optimizer
        optim = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams["learning_rate"], 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=self.hparams['weight_decay'], 
            amsgrad=False
        )

        # learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optim, factor=0.5, patience=2, threshold=0.008)

        return {"optimizer": optim, "lr_scheduler": lr_scheduler, "monitor": "val/answer_loss"}

    # change the mode of the answer module to training/validating
    def train_answer(self):
        self.answer_module.set_testing(False)

    # change the mode of the answer module to testing (predicting)
    def eval_answer(self):
        self.answer_module.set_testing(True)
