# LSDFormer: Lightweight SAR Ship Detection Enhanced with Efficient MultAttention and Structural Reparameterization
This is the official repo archiving the source code of LSDFormer

## How to
LSDFormer is implemented based on [Ultralytics YOLO](https://docs.ultralytics.com/zh/). We only made some modifications on it, so there is no need to download the complete code again. However, it should be noted that:

Please install the ultralytics package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8.  
    `pip install ultralytics`        

model profiles path:

    .\profiles  
This path contains the model configuration files for YOLO11 and LSDFormer. Additionally, a template for dataset configuration files is also provided.  

Training and testing script paths:

    .\train.py   
    .\test.py
We still provide training and testing file templates, and all parameters can be customized.

## Note
Before starting the model training, please first prepare the dataset and configure the dataset configuration file.