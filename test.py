import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('') #Path of the model training weight file
    model.val(data='', #Dataset configuration file path
              split='val',
              imgsz=640,
              batch=256,
              rect=False,
              save_json=False,
              project='runs/val',
              name='exp',
              )
