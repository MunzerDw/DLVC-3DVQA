# 3D Visual Question Answering

<p align="center"><img width="350" src="./docs/model.png"></p>

This is the official repository of our report [**3D Visual Question Answering**](./docs/3D_Visual_Question_Answering.pdf) by Leonard Schenk and Munzer Dwedari for the course Deep Learning in Visual Computing.
## Abstract
In this work, we introduce a new Seq2Seq architecture for the task of 3D Visual-Question-Answering (3D-VQA) on the ScanQA [^scanqa] benchmark. We especially distinguish ourselves from the baseline model ScanQA by not choosing the answer among the collection of answer candidates but by creating a language model to predict the answer word by word. Moreover, we employ attention mechanisms, which provide additional explainability, on both modalities of the input in the answer module. We also enhance the fusion of both modalities with an additional graph module. Our model outperforms the current baseline on 5 out of 7 benchmark scores [^scores]. Apart from that we shed light on a problem where models neglect the scene information during the answer prediction.

[^scanqa]: AZUMA, Daichi, et al. ScanQA: 3D Question Answering for Spatial Scene Understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022. S. 19129-19139.
[^scores]: https://eval.ai/web/challenges/challenge-page/1715/overview
## Installation

Please refer to [installation guide](docs/installation.md).

## Dataset

Please refer to [data preparation](docs/dataset.md) for preparing the ScanNet v2 and ScanQA datasets.
## Usage

### Training
- Start training:

  ```shell
  python scripts/train.py <experiment> --use_color --use_normal --use_height --use_multiview
  ```

  For more training options, please run `scripts/train.py -h`.

### Evaluation
- Evaluate a trained model in the validation dataset:

  ```shell
  python scripts/evaluate.py <experiment> <version>
  ```
  \<experiment> corresponds to the exerpiment under logs/ and \<version> to the experiment version to load the model from.

### Inference
- Prediction with the test dataset:

  ```shell
  python scripts/predict.py <experiment> <version> --test_type test_w_obj (or test_wo_obj)
  ```
  
### Scoring
- Scoring with the val dataset:

  ```shell
  python scripts/score.py <experiment> <version>
  ```
  
- Scoring with the test datasets:
  Please upload your inference results (pred.test_w_obj.json or pred.test_wo_obj.json) to the [ScanQA benchmark](https://eval.ai/web/challenges/challenge-page/1715/overview), which is hosted on [EvalAI](https://eval.ai/). 

## Logging

You can use tensorboard to check losses and accuracies by visiting <b>localhost:6006</b> after running:
```shell
tensorboard --logdir logs
```

## Acknowledgements
We would like to thank [ATR-DBI/ScanQA](https://github.com/ATR-DBI/ScanQA) for the dataset, the benchmark and its fusion codebase, [zlccccc/3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer) for the spatially refined object proposals, [facebookresearch/votenet](https://github.com/facebookresearch/votenet) for the 3D object detection and [daveredrum/ScanRefer](https://github.com/daveredrum/ScanRefer) for the 3D localization codebase.

