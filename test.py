import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('') #模型训练权重文件
    model.val(data='', #数据集配置文件
              split='val',
              imgsz=640,
              batch=256,
              rect=False,
              save_json=False,
              project='runs/val',
              name='exp',
              )