import json
import os
import random
from json import JSONEncoder
import cv2
import numpy as np
from tqdm import tqdm
from wand.image import Image


def wave_horizontal_image(image_file, to_image_file, wave_count=2):

    # Открываем изображение
    with Image(filename=image_file) as img:
        # Настраиваем фон на черный
        img.background_color = 'black'

        # Применяем волновой эффект
        amplitude = img.height / 14
        wave_length = img.width / wave_count
        img.wave(amplitude=amplitude, wave_length=wave_length)

        # Сохраняем результат
        img.save(filename=to_image_file)


def wave_vertical_image(image_file, to_image_file, wave_count=2):

    # Открываем изображение
    with Image(filename=image_file) as img:
        # Настраиваем фон на черный
        width = img.width
        height = img.height

        img.background_color = 'black'

        # Поворачиваем изображение на -90 градусов
        img.rotate(-90)

        # Применяем волновой эффект
        amplitude = img.width / 10
        wave_length = img.height / wave_count
        img.wave(amplitude=amplitude, wave_length=wave_length)

        # Поворачиваем изображение обратно на +90 градусов
        img.rotate(90)

        img.crop(width=width, height=height, gravity='center')

        # Сохраняем результат
        img.save(filename=to_image_file)


def rotate_image(image, angle):
    height, width = image.shape[:2]

    # Вычисли центр изображения
    center = (width // 2, height // 2)
    # Определи матрицу преобразования для поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # # Вычисли новые размеры изображения
    # new_width = int(width * np.abs(np.cos(np.radians(angle))) + height * np.abs(np.sin(np.radians(angle))))
    # new_height = int(height * np.abs(np.cos(np.radians(angle))) + width * np.abs(np.sin(np.radians(angle))))

    # Примени поворот к изображению
    border_mode = cv2.BORDER_CONSTANT
    border_value = (0, 0, 0)  # Черный цвет в формате BGR
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)
    return rotated_image



class CustomEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def add_aug_images(label_json_source_file: str, label_json_dest_file):
    with open(label_json_source_file) as json_file:
        data = json.load(json_file)

    # Создаем словарь для отслеживания количества повторений значений
    value_counts = {}

    # Создаем новый словарь с ограниченным количеством повторений значений
    new_dict = {}
    for key, value in tqdm(data.items()):
        image = cv2.imread(key)
        file_name_without_extension, file_extension = os.path.splitext(key)
        new_dict[key] = value
        for angle in [-10, -5, 5, 10]:
            new_image = rotate_image(image, angle)
            new_file_name = f"{file_name_without_extension}_r{angle}{file_extension}"
            new_dict[new_file_name] = value
            cv2.imwrite(new_file_name, new_image)

        new_file_name = f"{file_name_without_extension}_wh{file_extension}"
        wave_horizontal_image(key, new_file_name)
        new_dict[new_file_name] = value

        new_file_name = f"{file_name_without_extension}_wv{file_extension}"
        wave_vertical_image(key, new_file_name)
        new_dict[new_file_name] = value

    with open(label_json_dest_file, 'w') as file:
        json.dump(new_dict, file, indent=2, cls=CustomEncoder)



def prepare_data(label_file: str, train_file: str, val_file: str, split_coef: float = 0.75):
    with open(label_file) as f:
        train_data = json.load(f)

    train_data = [(k, v) for k, v in train_data.items()]
    random.shuffle(train_data)

    print('train len', len(train_data))

    train_len = int(len(train_data) * split_coef)

    train_data_splitted = train_data[:train_len]
    val_data_splitted = train_data[train_len:]

    print('train len after split', len(train_data_splitted))
    print('val len after split', len(val_data_splitted))

    with open(train_file, 'w') as f:
        json.dump(dict(train_data_splitted), f)

    with open(val_file, 'w') as f:
        json.dump(dict(val_data_splitted), f)



if __name__ == '__main__':
    label = "./hcr-archive-metric-book/data_mb/word_images/label.json"
    train = "./hcr-archive-metric-book/data_mb/word_images/label_train.json"
    train_aug = "./hcr-archive-metric-book/data_mb/word_images/label_train_with_rotated.json"
    val = "./hcr-archive-metric-book/data_mb/word_images/label_val.json"
    # 1 - разбиваем оригинальный файл label на два train и val
    prepare_data(label, train, val, split_coef=0.70)
    # 2 - для train увеличиваем кол-во картинок за счет поворота и дублей
    add_aug_images(train, train_aug)

    # wave_horizontal_image("./test/tools/ImageMagick-7.0.5/test3.jpg",
    #            "./test/tools/ImageMagick-7.0.5/test-wave-h.jpg",
    #                       wave_count=2)
    #
    # wave_vertical_image("./test/tools/ImageMagick-7.0.5/test3.jpg",
    #            "./test/tools/ImageMagick-7.0.5/test-wave-v.jpg",
    #                     wave_count=2)
