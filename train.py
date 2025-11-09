import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'') #Path of the model configuration file

    model.train(data=r'', #Dataset configuration file path
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
