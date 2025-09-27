import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_and_label(path_class_tuples):
    dfs = []
    for path, class_id in path_class_tuples:
        df = pd.read_csv(path)
        df['num_class'] = class_id
        dfs.append(df)
    return pd.concat(dfs).sort_values(by='sign_class')

def process_and_save(df, save_dir):
    for filename in df['filename'].unique():
        data_tmp = df[df['filename'] == filename].copy()
        data_tmp['x_center'] = data_tmp['x_from'] + data_tmp['width'] / 2
        data_tmp['y_center'] = data_tmp['y_from'] + data_tmp['height'] / 2

        if filename.startswith(('autosave13_04_2013', 'autosave16_04_2013')):
            w, h = 1920, 1080
        else:
            w, h = 1280, 720

        data_tmp['x_percent'] = data_tmp['x_center'] / w
        data_tmp['y_percent'] = data_tmp['y_center'] / h
        data_tmp['width_percent'] = data_tmp['width'] / w
        data_tmp['height_percent'] = data_tmp['height'] / h

        data_tmp[['num_class','x_percent','y_percent','width_percent','height_percent']].to_csv(
            f'data/{save_dir}/labels/{filename[:-4]}.txt',
            sep=' ', header=False, index=False, encoding='utf-8'
        )

train_files = [
    (r"C:/Users/Nikita/PycharmProjects/yolo11m/rtsd-d3-gt/blue_border/train_gt.csv", 0),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/blue_rect/train_gt.csv", 1),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/danger/train_gt.csv", 2),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/main_road/train_gt.csv", 3),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/mandatory/train_gt.csv", 4),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/prohibitory/train_gt.csv", 5),
]

val_files = [
    (r"C:/Users/Nikita/PycharmProjects/yolo11m/rtsd-d3-gt/blue_border/test_gt.csv", 0),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/blue_rect/test_gt.csv", 1),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/danger/test_gt.csv", 2),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/main_road/test_gt.csv", 3),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/mandatory/test_gt.csv", 4),
    (r"C:/Users/Nikita/Downloads/rtsd-public/detection/rtsd-d3-gt/prohibitory/test_gt.csv", 5),
]

data_train = load_and_label(train_files)
data_val = load_and_label(val_files)

process_and_save(data_train, 'train')
process_and_save(data_val, 'val')