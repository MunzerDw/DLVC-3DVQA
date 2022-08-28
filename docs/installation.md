## Installation

This code is based on [ScanRefer](https://github.com/daveredrum/ScanRefer). Please also refer to the ScanRefer setup.


- Install PyTorch:
    ```shell
    conda install pytorch==1.9.0 torchvision==0.10.0 torchtext==0.10.0 cudatoolkit=10.2 -c pytorch
    ```
- Install pytorch-geometric:
    ```shell
    conda install pyg -c pyg
    ```
- Install the necessary packages with `requirements.txt`:
    ```shell
    pip install -r requirements.txt
    ```

- Compile the CUDA modules for the PointNet++ backbone:
    ```shell
    cd lib/pointnet2
    python setup.py install
    ```

Note that this code has been tested with Python 3.8, pytorch 1.9.0, and CUDA 10.2 on Ubuntu 20.04.