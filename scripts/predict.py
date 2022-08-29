import sys
import os
import json
import pickle
import yaml
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import nltk

sys.path.append(os.path.join(os.getcwd()))

from data.config import CONF
from data.dataset import ScannetQADataset
from data.scannet.model_util_scannet import ScannetDatasetConfig

from lib.ap_helper import parse_predictions

from utils.box_util import get_3d_box, get_3d_box_batch

from model.ours import Ours

nltk.download('omw-1.4')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Scannet config
DC = ScannetDatasetConfig()

# config
POST_DICT = {
    "remove_empty_box": True, 
    "use_3d_nms": True, 
    "nms_iou": 0.25,
    "use_old_type_nms": False, 
    "cls_nms": True, 
    "per_class_proposal": True,
    "conf_thresh": 0.05,
    "dataset_config": DC
}

# prediction
def predict(dataset, model, device=None):
    # set model to evaluation mode
    model.eval()
    
    # set answer module to testing mode
    model.eval_answer()
    
    # get dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # get real scene IDs
    scene_number_to_id = {v: k for k, v in dataset.scene_id_to_number.items()}

    scene_ids = []
    question_ids = []
    answer_top10s = []
    questions = []
    bboxes = []
    
    # for json and pickle files
    preds = []
    preds_pickle = {}

    batch_iter = iter(dataloader)
    for batch in tqdm(batch_iter):
        # put batch on device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(DEVICE)
                
        with torch.no_grad():
            out = model(batch)
            answer_top10s.extend(out['pred_answers'])

        # get scene IDs and question IDs of the samples in the batch
        scene_ids_2 = [scene_number_to_id[item] for item in out['scene_id'].cpu().tolist()]
        question_ids_2 = ['val-' + scene_ids_2[i].split("_")[0] + "-" + str(item) for i, item in enumerate(out['question_id'].cpu().tolist())]
        
        # store batch results
        scene_ids += scene_ids_2
        question_ids += question_ids_2
        questions += out['question']
        
        # bbox prediction
        objectness_preds_batch = torch.argmax(out['objectness_scores'], 2).long()
        if POST_DICT:
            _ = parse_predictions(out, POST_DICT)
            nms_masks = torch.LongTensor(out['pred_mask']).cuda()
            # construct valid mask
            pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        else:
            # construct valid mask
            pred_masks = (objectness_preds_batch == 1).float()

        pred_ref = torch.argmax(out['cluster_ref'] * pred_masks, 1) # (B,)
        pred_center = out['center'] # (B,K,3)
        pred_heading_class = torch.argmax(out['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(out['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        pred_size_class = torch.argmax(out['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(out['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3
        
        # ground truth bbox
        gt_center = out['ref_center_label'].cpu().numpy() # (B,3)
        gt_heading_class = out['ref_heading_class_label'].cpu().numpy() # B
        gt_heading_residual = out['ref_heading_residual_label'].cpu().numpy() # B
        gt_size_class = out['ref_size_class_label'].cpu().numpy() # B
        gt_size_residual = out['ref_size_residual_label'].cpu().numpy() # B,3
        
        # convert gt bbox parameters to bbox corners
        gt_obb_batch = DC.param2obb_batch(
            gt_center[:, 0:3], 
            gt_heading_class, 
            gt_heading_residual,
            gt_size_class, 
            gt_size_residual
        )
        gt_bbox_batch = get_3d_box_batch(
            gt_obb_batch[:, 3:6], 
            gt_obb_batch[:, 6], 
            gt_obb_batch[:, 0:3]
        )
        
        # save bounding boxes
        for i in range(pred_ref.shape[0]):
            # compute the iou
            pred_ref_idx = pred_ref[i]
            pred_obb = DC.param2obb(
                pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(), 
                pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
                pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
            )
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
            bboxes.append(pred_bbox)

    # format saved data in correct format
    for i, scene_id in enumerate(scene_ids):
        # json file
        dict = {
            "scene_id": scene_id,
            "question_id": question_ids[i],
            "answer_top10": answer_top10s[i],
            "question": questions[i],
            "bbox": bboxes[i].tolist(),
        }
        preds.append(dict)
        
        # pickle file
        if dict['scene_id'] not in preds_pickle:
            preds_pickle[dict['scene_id']] = {}
        if dict['question_id'] not in preds_pickle[dict['scene_id']]:
            preds_pickle[dict['scene_id']][dict['question_id']] = {}
        preds_pickle[dict['scene_id']][dict['question_id']]['pred_answers_at10'] = dict['answer_top10']
        preds_pickle[dict['scene_id']][dict['question_id']]['question'] = dict['question']
        
    return preds, preds_pickle

def parse_arguments():
    test_types = ['test_w_obj', 'test_wo_obj']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('version', type=int)
    parser.add_argument('--test_type', choices=test_types, type=str, required=True)

    args = parser.parse_args()
    return vars(args)
    
def main():
    # read arguments
    params = parse_arguments()
    experiment = params["experiment"]
    version = "version_" + str(params["version"])
    test_type = params["test_type"]
    
    # hyperparameters path
    params_path = os.path.join(CONF.PATH.OUTPUT, experiment, version, "hparams.yaml")
    with open(params_path, 'r') as yaml_in:
        params = yaml.safe_load(yaml_in)
        print(params)

    #####################
    ##                 ##
    ##      Data       ##
    ##                 ##
    #####################

    SCANQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANQA, "ScanQA_v1.0_train.json"))) 
    SCANQA_TEST = json.load(open(os.path.join(CONF.PATH.SCANQA, "ScanQA_v1.0_" + test_type + ".json")))
    scanqa = {
        'train': SCANQA_TRAIN,
    }
    test_dataset = ScannetQADataset(
      scanqa=SCANQA_TEST, 
      scanqa_both=scanqa,
      split='test', 
      augment=False,
      use_color = params["use_color"],
      use_height=params["use_height"],
      use_normal = params["use_normal"],
      use_multiview = params["use_multiview"]
    )
    print(f'loaded data ({len(test_dataset)})')

    ######################
    ##                  ##
    ##      Model       ##
    ##                  ##
    ######################

    # checkpoint path
    checkpoint = os.listdir(os.path.join(
      CONF.PATH.OUTPUT, 
      experiment, 
      version,
      "checkpoints"
    ))[0]
    cp_path = os.path.join(CONF.PATH.OUTPUT, experiment, version, "checkpoints", checkpoint)
        
    # load model checkpoint
    model = Ours.load_from_checkpoint(cp_path, hparams=params, answer_vocab=test_dataset.vocab)
    model.to(DEVICE)
    print('loaded model')

    # get predictions
    print('evaluating...')
    preds, preds_pickle = predict(test_dataset, model, device=DEVICE)

    # save predictions to files
    pred_path = os.path.join(CONF.PATH.OUTPUT, experiment, version, "pred." + test_type + ".pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(preds_pickle, f)
    json.dump(preds, open(pred_path[:-4]+'.json','w'), indent=4)
    print('saved results')

if __name__ == '__main__':
    try:
        main()
        sys.stdout.flush()
    except KeyboardInterrupt:
        print('Interrupted...')
        try:
            sys.exit(0)
        except:
            os._exit(0)