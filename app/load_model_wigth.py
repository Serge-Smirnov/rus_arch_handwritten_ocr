import requests
import os
import glob


def __clear_directory__(dir_path: str):
    files = glob.glob(f'{dir_path}/*')
    for f in files:
        # конфиг файлы надо тоже положить в bin-repo и загружать, так более целостно будет чтобы не напутать какой файл от какой модели
        if f.lower().endswith(".json") or f.lower().endswith(".md") or f.lower().endswith(".yml") or f.lower().endswith(".yaml"):
            continue
        os.remove(f)


def __prepare_directory__(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        __clear_directory__(dir_path)    


def __save_binary_file__(full_file_name: str, binary_data: bytes):
    with open(full_file_name, 'wb') as file:
        file.write(binary_data)


def load_segmentation_weights(weigth_url: str, directory: str = './app/segmentation_weight'):
    __prepare_directory__(directory)
    response = requests.get(weigth_url, allow_redirects=True)
    __save_binary_file__(f'{directory}/model_weight.pth', response.content)


def load_translation_weights(model_weight_url: str, resnet_weight_url: str, directory:str = './app/translation_weight'):
    __prepare_directory__(directory)
    response = requests.get(model_weight_url, allow_redirects=True)
    __save_binary_file__(f'{directory}/model_weight.ckpt', response.content)
    response = requests.get(resnet_weight_url, allow_redirects=True)
    __save_binary_file__(f'{directory}/resnet34.pth', response.content)


if __name__ == '__main__':
    pass
