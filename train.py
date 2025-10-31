import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\EdgeDownload\ultralytics-main\profiles\LSDFormer.yaml')

    model.train(data=r'D:\EdgeDownload\ultralytics-main\profiles\dataset_profiles\HRSID.yaml',
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