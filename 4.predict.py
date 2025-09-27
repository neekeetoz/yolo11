from ultralytics import YOLO

model = YOLO('model/yolo11m_custom100.pt') # Укажите путь к вашему файлу best.pt

results = model.predict('C:/Users/Nikita/PycharmProjects/yolo11m/data/val/images/autosave21_01_2013_10_09_44_0.jpg',save=True)

names = [results[0].names[cls.item()] for cls in results[0].boxes.cls.int()]  # class name of each box
confs = results[0].boxes.conf  # confidence score of each box

for i in range(len(names)):
    print(f"{names[i]}: {confs[i]}")