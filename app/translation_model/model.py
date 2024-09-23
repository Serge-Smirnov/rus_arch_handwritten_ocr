import cv2

import numpy as np
import torch
import torch.nn as nn
import torchvision

OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in pytorch."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        try:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
        return image


def __get_char_map__(alphabet):
    """Make from string alphabet character2int dict.
    Add BLANK char fro CTC loss and OOV char for out of vocabulary symbols."""
    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map


def __get_transforms__(height, width):
    transforms = torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


def __predict__(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds


class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = __get_transforms__(height, width)

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor


class Tokenizer:
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    def __init__(self, alphabet):
        self.char_map = __get_char_map__(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                        char_enc != self.char_map[OOV_TOKEN]
                        and char_enc != self.char_map[CTC_BLANK]
                        # idx > 0 to avoid selecting [-1] item
                        and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


def get_resnet34_backbone(resnet_model_path: str):
    m = torchvision.models.resnet34(pretrained=False)
    custom_weights = torch.load(resnet_model_path)
    m.load_state_dict(custom_weights)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu,
              m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(
            self, resnet_model_path: str, number_class_symbols, time_feature_count=96, lstm_hidden_size=256,
            lstm_num_layers=3,
    ):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(resnet_model_path=resnet_model_path)
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count,
                             lstm_hidden_size, lstm_num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


class HcrPredictor:
    def __init__(self, model_path, resnet_model_path, config, device='cpu'):
        self.tokenizer = Tokenizer(config['alphabet'])
        self.device = torch.device(device)
        # load model
        self.model = CRNN(resnet_model_path=resnet_model_path, number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config['image']['height'],
            width=config['image']['width'],
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images = self.transforms(images)
        pred = __predict__(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred


class Translator():
    def __init__(self, model_path, resnet_model_path, config, device='cpu'):
        self.predictor = HcrPredictor(model_path=model_path, resnet_model_path=resnet_model_path, config=config, device=device)

    def translate(self, image) -> str:
        return self.predictor(image)


def get_translator_instance(model_path: str, resnet_model_path: str, config: dict, device: str = 'cpu') -> Translator:
    return Translator(model_path, resnet_model_path, config, device)


if __name__ == '__main__':
    config_json = {
        "alphabet": " ,-.0123456789IVXbiАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёѣѲѳ№",
        "save_dir": "./hcr-archive-metric-book/data_mb/saved_models/",
        "log_dir": "./hcr-archive-metric-book/data_mb/saved_validations/",
        "num_epochs": 200,
        "image": {
            "width": 256,
            "height": 32
        },
        "train": {
            "root_path": "./hcr-archive-metric-book/data_mb/word_images/",
            "json_path": "./hcr-archive-metric-book/data_mb/word_images/label_train_with_rotated.json",
            "batch_size": 32
        },
        "val": {
            "root_path": "./hcr-archive-metric-book/data_mb/word_images/",
            "json_path": "./hcr-archive-metric-book/data_mb/word_images/label_val.json",
            "batch_size": 128
        }
    }

    if torch.cuda.is_available():
        print('cuda')
    else:
        print('cpu')

    predictor = HcrPredictor(
        model_path='./hcr-archive-metric-book/app/translation_weight/model_weight.ckpt',
        config=config_json)
    print(predictor)

    img = cv2.imread(
        f'./hcr-archive-metric-book/app/test/cut_images/good/Ивановичъ__Ивановичъ__00015_Image00015.jpg_39.jpg')
    pred = predictor(img)
    print('Перевод: ', predictor(img))
