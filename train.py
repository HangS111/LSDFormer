import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'') #模型配置文件

    model.train(data=r'', #数据集配置文件
                task='detect',
                imgsz=640,
                epochs=300,
                batch=8,
                workers=20,
                device='0',
                amp=False,
                optimizer='SGD',
                resume=True,
                project='run',
                name='1',
                seed=3407
                )