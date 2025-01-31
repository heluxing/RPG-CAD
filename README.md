We provide a sample database. Please organize your data according to this sample, paying special attention to modifying the root directory and categories in "dataset_example/my_dataset.yaml". Then, run the following code to start training.
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
```
If you want to test a trained model, use the following code for testing.
### Python
```python
from ultralytics import YOLO
model = YOLO("your_best.pt")
data = 'dataset_example/my_dataset.yaml'
if __name__ == '__main__':
    model.val(data=data, split="test",name="test",batch=1, device=0,plots=False)
```

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

Thanks for the great implementations! 
