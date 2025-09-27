from ultralytics import YOLO
from model import parallelTestModule

# Load a model
model = YOLO("model/yolo11m_custom100.pt")

if __name__ == '__main__':
    extractor = parallelTestModule.ParallelExtractor()
    extractor.runInParallel(numProcesses=2, numThreads=4)
    # Validate the model
    metrics = model.val(data="model/dataset_custom.yaml")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    print(f"list of mAP50-95 for each category: {metrics.box.maps}")