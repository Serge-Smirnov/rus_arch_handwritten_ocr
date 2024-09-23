import json
import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.animation as animation
class MetricsPlotter:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()

        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()

        self.showing = False

    def add_point(self, epoch, train_loss, train_acc, val_loss, val_accuracy):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)

        if not self.showing:
            self.show()

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(self.epochs, self.train_losses, 'b', label='Train Loss')
        self.ax1.plot(self.epochs, self.val_losses, 'g', label='Validation Loss')
        self.ax2.plot(self.epochs, self.train_accuracies, 'b', label='Train Accuracy')
        self.ax2.plot(self.epochs, self.val_accuracies, 'r', label='Validation Accuracy')

        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()

        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()

        plt.pause(0.1)

    def show(self):
        plt.show(block=False)
        self.showing = True


def prepare_data(label_file: str, split_coef: float = 0.75, total_images_limit=None):
    with open(label_file) as f:
        train_data = json.load(f)

    train_data = [(k, v) for k, v in train_data.items()]

    if total_images_limit:
        train_data = train_data[:total_images_limit]

    print('train len', len(train_data))

    train_len = int(len(train_data) * split_coef)

    train_data_splitted = train_data[:train_len]
    val_data_splitted = train_data[train_len:]

    print('train len after split', len(train_data_splitted))
    print('val len after split', len(val_data_splitted))

    with open('data_mb/final_data/train/train_labels_splitted.json', 'w') as f:
        json.dump(dict(train_data_splitted), f)

    with open('data_mb/final_data/train/val_labels_splitted.json', 'w') as f:
        json.dump(dict(val_data_splitted), f)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# функция которая помогает объединять картинки и таргет-текст в батч
def collate_fn(batch):
    images, texts, enc_texts, img_paths = zip(*batch)
    images = torch.stack(images, 0)
    text_lens = torch.LongTensor([len(text) for text in texts])
    enc_pad_texts = pad_sequence(enc_texts, batch_first=True, padding_value=0)
    return images, texts, enc_pad_texts, text_lens, img_paths


def get_data_loader(
        transforms, json_path, root_path, tokenizer, batch_size, drop_last
):
    dataset = OCRDataset(json_path, root_path, tokenizer, transforms)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1, # todo 8,
    )
    return data_loader


class OCRDataset(Dataset):
    def __init__(self, json_path, root_path, tokenizer, transform=None):
        super().__init__()
        self.transform = transform
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.data_len = len(data)

        self.img_paths = []
        self.texts = []
        for img_name, text in data.items():
            self.img_paths.append(os.path.join(root_path, img_name))
            self.texts.append(text)
        self.enc_texts = tokenizer.encode(self.texts)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        text = self.texts[idx]
        enc_text = torch.LongTensor(self.enc_texts[idx])
        image = cv2.imread(img_path)

        # # зачитываем картинку в сером цвете (только один канал)
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # # востанавливаем из двумерного массива в трехмерный массив, с которым дальше будет работать нейронка
        # image = image[:, :,  np.newaxis]

        if self.transform is not None:
            image = self.transform(image)
        return image, text, enc_text, img_path


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


def get_char_map(alphabet):
    """Make from string alphabet character2int dict.
    Add BLANK char fro CTC loss and OOV char for out of vocabulary symbols."""
    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map


class Tokenizer:
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    def __init__(self, alphabet):
        self.char_map = get_char_map(alphabet)
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


def get_accuracy_train(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score_train = np.mean(scores)
    return avg_score_train


def get_accuracy_val(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score_val = np.mean(scores)
    return avg_score_val


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
            image = cv2.resize(image, (self.width, self.height),
                               interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
        return image


def get_train_transforms(height, width):
    transforms_list = [
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ]

    # Добавляем аугментацию данных
    augmentations = [
        transforms.RandomRotation(degrees=10),
    ]

    # Соединяем существующие трансформации с аугментациями
    transforms_list.extend(augmentations)

    # Создаем и возвращаем набор трансформаций
    return transforms.Compose(transforms_list)


def get_val_transforms(height, width):
    transforms = torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


def get_resnet34_backbone(pretrained=True):
    m = torchvision.models.resnet34(pretrained=True)
    # m = torchvision.models.resnet152(pretrained=pretrained)
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
            self, number_class_symbols, time_feature_count=96, lstm_hidden_size=256,
            lstm_num_layers=3,
    ):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden_size, lstm_num_layers)
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


def val_loop(data_loader, model, tokenizer, criterion, device, log_dir: str, enable_log=False):
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()

    model.eval()  # Переводим модель в режим evaluation (не обучение)

    good_dir = f'{log_dir}/good/'
    bad_dir = f'{log_dir}/bad/'
    if enable_log:
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)

    with torch.no_grad():  # Отключаем вычисление градиентов для эффективности
        for images, texts, enc_pad_texts, text_lens, img_paths in data_loader:
            images = images.to(device)
            enc_pad_texts = enc_pad_texts.to(device)

            # Предсказываем значения текста
            output = model(images)
            output_lenghts = torch.full(
                size=(output.size(1),),
                fill_value=output.size(0),
                dtype=torch.long
            )

            # Рассчитываем потери между предсказаниями и истинными значениями
            loss = criterion(output, enc_pad_texts, output_lenghts, text_lens)

            # Рассчитываем точность предсказания
            pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
            text_preds = tokenizer.decode(pred)
            acc = get_accuracy_val(texts, text_preds)
            if enable_log:
                for text, text_pred, path in zip(texts, text_preds, img_paths):
                    if text == text_pred:
                        shutil.copy(path, f'{good_dir}/{text_pred}__{text}__{os.path.basename(path)}')
                    else:
                        shutil.copy(path, f'{bad_dir}/{text_pred}__{text}__{os.path.basename(path)}')

            # Обновляем средние потери и точность
            loss_avg.update(loss.item(), len(texts))
            acc_avg.update(acc, len(texts))

    print(f'Validation, Loss: {loss_avg.avg:.4f}, Accuracy: {acc_avg.avg:.4f}')
    return loss_avg.avg, acc_avg.avg


def train_loop(data_loader, model, criterion, optimizer, epoch, tokenizer):
    loss_avg = AverageMeter()
    accuracy_avg = AverageMeter()  # Для подсчёта средней точности
    model.train()
    for images, texts, enc_pad_texts, text_lens, img_paths in data_loader:
        model.zero_grad()
        images = images.to(DEVICE)
        batch_size = len(texts)
        output = model(images)
        
        y_pred = predict_test(images, model, tokenizer)
        y_true = texts
        
        accuracy = get_accuracy_train(y_true, y_pred)
        accuracy_avg.update(accuracy, batch_size)  # Обновление средней точности

        output_lenghts = torch.full(
            size=(output.size(1),),
            fill_value=output.size(0),
            dtype=torch.long
        )
        loss = criterion(output, enc_pad_texts, output_lenghts, text_lens)
        loss_avg.update(loss.item(), batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print(f'\nEpoch {epoch}, Loss: {loss_avg.avg:.5f}, LR: {lr:.7f}, Train, acc: {accuracy_avg.avg:.4f}')
    return loss_avg.avg, accuracy_avg.avg


def predict_test(images, model, tokenizer):
    model.train()
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds


def get_loaders(tokenizer, config):
    train_transforms = get_train_transforms(
        height=config['image']['height'],
        width=config['image']['width']
    )
    train_loader = get_data_loader(
        json_path=config['train']['json_path'],
        root_path=config['train']['root_path'],
        transforms=train_transforms,
        tokenizer=tokenizer,
        batch_size=config['train']['batch_size'],
        drop_last=True
    )
    val_transforms = get_val_transforms(
        height=config['image']['height'],
        width=config['image']['width']
    )
    val_loader = get_data_loader(
        transforms=val_transforms,
        json_path=config['val']['json_path'],
        root_path=config['val']['root_path'],
        tokenizer=tokenizer,
        batch_size=config['val']['batch_size'],
        drop_last=False
    )
    return train_loader, val_loader


def train(config):
    plotter = MetricsPlotter()
    tokenizer = Tokenizer(config['alphabet'])
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    train_loader, val_loader = get_loaders(tokenizer, config)
    print("train: " + str(train_loader.dataset.data_len))
    print("val: " + str(val_loader.dataset.data_len))

    model = CRNN(number_class_symbols=tokenizer.get_num_chars())
    # load from saved model:
    # model.load_state_dict(torch.load("./hcr-archive-metric-book/stage_2/data/experiments/test/model-76-0.7030.ckpt"))
    model.to(DEVICE)

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,
                                  weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5, patience=15)
    best_acc = -np.inf
    val_loop(val_loader, model, tokenizer, criterion, DEVICE, config['log_dir'], False)
    i = 0
    for epoch in tqdm(range(config['num_epochs'])):
        train_loss_avg, train_acc_avg = train_loop(train_loader, model, criterion, optimizer, epoch, tokenizer)

        # директорая в которую будем сохранять результат валидации
        val_dir = f'{config["log_dir"]}/{i}/'

        val_loss_avg, acc_avg = val_loop(val_loader, model, tokenizer, criterion, DEVICE, val_dir, i % 3 == 0)
        scheduler.step(acc_avg)
        plotter.add_point(epoch, train_loss_avg, train_acc_avg, val_loss_avg, acc_avg)
        plotter.show()
        if acc_avg > 0.7 and (acc_avg - 0.005) > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(config['save_dir'], f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')
        i = i + 1

mb_config_json = {
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

if __name__ == '__main__':
    # prepare_data("./hcr-archive-metric-book/stage_2/data_mb/final_data/train/label.json", split_coef=0.8)
    train(mb_config_json)
