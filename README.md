### Python
```python
from ultralytics import YOLO

model_path = r"yolov8n_CRA_BDA_SMA.yaml"
model = YOLO(model_path)
data = r"dataset_example/my_dataset.yaml"
project = "Craniocerebral_Multi-Abnormality_Detection"
name = "yolov8n_CRA_BDA_SMA"
path = project + "/" + name
if __name__ == '__main__':
    # #s
    results = model.train(data=data, project=project, name=name, imgsz=512, batch=32,
                          epochs=200, patience=100,
                          pretrained=False, save_period=5,device=[0,1], resume=True)
