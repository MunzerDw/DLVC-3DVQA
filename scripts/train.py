import sys
import os
import json
import argparse
import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import nltk

sys.path.append(os.path.join(os.getcwd()))

from model.ours import Ours
from data.config import CONF
from data.dataset import ScannetQADataset

nltk.download('punkt')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--resume', help='resume the training of latest version of the experiment', action='store_true')
    parser.add_argument('--use_color', help='include color features in the point cloud', action='store_true')
    parser.add_argument('--use_normal', help='include normal features in the point cloud', action='store_true')
    parser.add_argument('--use_height', help='include height features in the point cloud', action='store_true')
    parser.add_argument('--use_multiview', help='include multiview features in the point cloud', action='store_true')
    parser.add_argument('--use_standard_proposal', help='use proposal matching module from 3DVG', action='store_true')
    parser.add_argument('--use_answer_transformer', help='use transformer for the answer module', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=int, default=0.00008)
    parser.add_argument('--lang_use_bidir', help='use bi-directional LSTM in the Seq2Seq question encoder', action='store_true')
    parser.add_argument('--weight_decay', type=int, default=0.0004)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--mcan_flat_out_size', type=int, default=512)
    parser.add_argument('--emb_size', type=int, default=300)
    parser.add_argument('--num_proposal', type=int, default=128)

    args = parser.parse_args()
    return vars(args)

def main():
  ################################
  ##                            ##
  ##      Hyperparameters       ##
  ##                            ##
  ################################

  # read arguments
  params = parse_arguments()

  if params["resume"]:
    # get latest version
    versions = [x[1] for x in os.walk(
        os.path.join(
          "logs", 
          params["experiment"], 
        )
    )][0]
    max_version = max([int(version[-1]) for version in versions if "version" in version])
    version = "version_" + str(max_version)
        
    # get best checkpoint file name
    checkpoint = os.listdir(os.path.join(
      "./logs", 
      params["experiment"],
      version,
      "checkpoints"
    ))[0]
        
    # get best model path
    checkpoint_path = os.path.join(
      "logs", 
      params["experiment"], 
      version, 
      "checkpoints",
      checkpoint
    )
    
    # hyperparameters path
    params_path = os.path.join("logs", params["experiment"], version, "hparams.yaml")
    with open(params_path, 'r') as yaml_in:
        params = yaml.safe_load(yaml_in)
        print(params)
    
  else:
    checkpoint_path = None
    print(params)

  #####################
  ##                 ##
  ##      Data       ##
  ##                 ##
  #####################

  # all ScanQA data
  SCANQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANQA, "ScanQA_v1.0_train.json"))) 
  SCANQA_VAL = json.load(open(os.path.join(CONF.PATH.SCANQA, "ScanQA_v1.0_val.json")))

  # Datasets
  scanqa = {
      'train': SCANQA_TRAIN,
      'val': SCANQA_VAL
  }
  train_dataset = ScannetQADataset(
    scanqa=scanqa['train'], 
    scanqa_both=scanqa,
    split='train', 
    augment=True,
    use_color = params["use_color"],
    use_height=params["use_height"],
    use_normal = params["use_normal"],
    use_multiview = params["use_multiview"]
  )
  val_dataset = ScannetQADataset(
    scanqa=scanqa['val'], 
    scanqa_both=scanqa,
    split='val', 
    augment=False,
    use_color = params["use_color"],
    use_height=params["use_height"],
    use_normal = params["use_normal"],
    use_multiview = params["use_multiview"]
  )

  # Dataloaders
  batch_size = 8
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

  print("Number of training samples:", len(train_dataset))
  print("Number of validation samples:", len(val_dataset))

  #########################
  ##                     ##
  ##      Training       ##
  ##                     ##
  #########################

  # function to get the model
  def get_model(params):
      # model
      model = Ours(params, answer_vocab=train_dataset.vocab)
      
      model.train()
      return model

  epochs = 40

  # callbacks
  checkpoint_callback = ModelCheckpoint(dirpath=None, save_top_k=2, monitor="val/answer_loss")
  lr_monitor = LearningRateMonitor(logging_interval="epoch")

  # model
  model = get_model(params)

  # logger
  logger = TensorBoardLogger("./logs", name=params['experiment'])

  # checkpointing
  checkpoint_callback = ModelCheckpoint(dirpath=None, save_top_k=1, monitor="val/answer_loss")

  # trainer
  trainer = pl.Trainer(
    max_epochs=epochs, 
    log_every_n_steps=1, 
    accelerator='gpu', 
    devices=1, 
    logger=logger, 
    enable_checkpointing=True, 
    callbacks=[checkpoint_callback, lr_monitor]
  )

  # training
  trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader, 
    ckpt_path=checkpoint_path
  )

if __name__ == '__main__':
    try:
        main()
        sys.stdout.flush()
    except KeyboardInterrupt:
        print('Interrupted...')
        print('Best model saved under logs/<experiment name>/checkpoints/')
        print("You can resume an experiment by setting the '--resume' flag")
        try:
            sys.exit(0)
        except:
            os._exit(0)