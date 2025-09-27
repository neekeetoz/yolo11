# YOLO11m

Модель yolo11m обучена на датасете RTSD-D3 в течение 100 эпох.

Предварительно перед обучением модели производилась нормализация разметки дорожных знаков под формат обучения модели yolo11.

Модель yolo11m_custom100.pt обучалась на предварительно обученной модели yolo11m.pt, которая содержит в себе баланс между точность обнаружения дорожного знака и скоростью распознавания на видео.

<img width="1016" height="456" alt="image" src="https://github.com/user-attachments/assets/436d7244-fccb-4929-8323-c69ad3f92419" />


В данном исследовании производится сравнение обученной модели yolo11m_custom100.pt с моделью из статьи ниже: [Ссылка](https://cyberleninka.ru/article/n/rossiyskaya-baza-izobrazheniy-avtodorozhnyh-znakov/viewer)

Пример детектирования дорожных знаков продемонстрирован ниже на изображении:

![autosave21_01_2013_10_09_44_0](https://github.com/user-attachments/assets/54ded67a-00b9-49f5-820c-965849b83c4c)

____

# Результаты валидации

Class | Images | Instances | P | R | mAP50 | mAP50-95
----------|-------|------|------|------|------|------
all | 3022 | 4826 | 0.863 | 0.907 | 0.927 | 0.601
blue_border | 431 |  474 | 0.877 | 0.845 | 0.92 | 0.597
blue_rect | 1393 |  2085 | 0.85 | 0.913 | 0.919 | 0.642
danger | 594 | 651  | 0.893 | 0.937 | 0.95 | 0.595
main_road | 422 | 431  | 0.862 | 0.959 | 0.933 | 0.657
mandatory | 408 | 501  | 0.855 | 0.886 | 0.92 | 0.555
prohibitory | 620 | 684  | 0.841 | 0.901 | 0.919 | 0.561

Roc-auc на валидационной выборке:
<img width="865" height="688" alt="image" src="https://github.com/user-attachments/assets/10881d8e-38a9-4bc3-8dd3-0dbcde1efd70" />

Сравнение с результатами обучения модели из статьи на выборке RTSD-D3:

Class | Results from the article | yolo11m_custom100.pt
----------|-------|------
blue_border | 0.83 | 0.92
blue_rect | 0.76 | 0.97
danger | 0.86 | 0.97
main_road | 0.9 | 0.97
mandatory | 0.8 | 0.95
prohibitory | 0.72 | 0.95

Ссылка на результаты обучения и валидации модели: [Ссылка](https://github.com/neekeetoz/yolo11/tree/main/runs/detect/train)

Ссылка на папку с моделью, исходными данными и результатами детектирования по видео: [Ссылка](https://disk.yandex.ru/d/KPygXiOIxa1qjQ)


