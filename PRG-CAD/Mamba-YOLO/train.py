from ultralytics import YOLO,RTDETR

model_path = "Mamba-YOLO-T-ours.yaml"  # 3,238,564 parameter
model = YOLO(model_path)
data = 'dataset.yaml'
project = " CraniocerebralMulti-AbnormalityDetection"
name = "mamba-yolo-T_ours"
path = project + "/" + name
if __name__ == '__main__':
    results = model.train(data=data, project=project, name=name, imgsz=512, batch=32,
                          epochs=200, patience=100,
                          pretrained=False, save_period=1, device=[0, 1], resume=True,mosaic=0)  # #optimizer="Adam",
