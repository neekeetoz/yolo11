import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score, roc_curve

# Имена классов
class_names = ["blue_border", "blue_rect", "danger", "main_road", "mandatory", "prohibitory"]

# Папки с изображениями и разметкой
image_dir = 'C:/Users/Nikita/PycharmProjects/yolo11m/data/val/images'
label_dir = 'C:/Users/Nikita/PycharmProjects/yolo11m/data/val/labels'

# Инициализация модели
model = YOLO("model/yolo11m_custom100.pt")

# Поддерживаемые расширения изображений
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def load_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    labels.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height]
                    })
    return labels

# Список изображений
image_files = [f for f in os.listdir(image_dir)
               if os.path.splitext(f)[1].lower() in valid_extensions]

all_y_true = []
all_pred_scores = []
all_pred_classes = []

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    label_name = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(label_dir, label_name)

    labels = load_labels(label_path)
    if not labels:
        print(f"Внимание: файл разметки не найден или пуст для изображения {img_file}")
        continue

    results = model.predict(img_path)
    boxes = results[0].boxes
    pred_classes = boxes.cls.cpu().numpy().astype(int)
    pred_confs = boxes.conf.cpu().numpy()

    true_classes = np.array([lbl['class_id'] for lbl in labels])

    for c in true_classes:
        all_y_true.append(1)
        mask_pred = (pred_classes == c)
        score = pred_confs[mask_pred].max() if np.any(mask_pred) else 0
        all_pred_scores.append(score)
        all_pred_classes.append(c)

    for c, conf in zip(pred_classes, pred_confs):
        if c not in true_classes:
            all_y_true.append(0)
            all_pred_scores.append(conf)
            all_pred_classes.append(c)

all_y_true = np.array(all_y_true)
all_pred_scores = np.array(all_pred_scores)
all_pred_classes = np.array(all_pred_classes)

unique_classes = np.unique(all_pred_classes)

plt.figure(figsize=(8, 6))
for cls in unique_classes:
    y_true_cls = (all_pred_classes == cls).astype(int) * all_y_true + (all_pred_classes != cls).astype(int) * 0
    y_scores_cls = (all_pred_classes == cls).astype(float) * all_pred_scores

    if np.sum(y_true_cls == 1) > 0 and np.sum(y_true_cls == 0) > 0:
        roc_auc = roc_auc_score(y_true_cls, y_scores_cls)
        fpr, tpr, _ = roc_curve(y_true_cls, y_scores_cls)
        class_name = class_names[cls] if cls < len(class_names) else f'Класс {cls}'
        print(f"Класс '{class_name}': ROC AUC = {roc_auc:.3f}")
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    else:
        class_name = class_names[cls] if cls < len(class_names) else f'Класс {cls}'
        print(f"Класс '{class_name}': недостаточно данных для ROC AUC")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve по классам')
plt.legend()
plt.grid(True)
plt.show()