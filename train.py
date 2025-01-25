import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.train(data="",
                      cache=False,
                      imgsz=640,
                      epochs=150,
                      batch=32,
                      close_mosaic=0,
                      workers=4,
                      # device='0',
                      optimizer='SGD',  # using SGD
                      project='runs/train',
                      name='exp',)