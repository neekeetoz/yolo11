from ultralytics import YOLO

# Load a model
model = YOLO("model/yolo11m.pt")  # load an official model

# Train the model on your custom dataset
model.train(data="model/dataset_custom.yaml", epochs=100, imgsz=640, batch=8, workers=0, device=0)