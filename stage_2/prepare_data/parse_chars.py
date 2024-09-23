import json
from collections import Counter
import shutil
import os
import matplotlib.pyplot as plt

def mb():
    with open("metric_book_1-10.json") as json_file:
        data = json.load(json_file)

    annotations = data["annotations"]
    names = []
    for annotation in annotations:
        metadata = annotation["metadata"]
        if metadata:
            name = metadata["name"]
            if name:
                names.append(name)

    # chars = set()
    # for name in names:
    #     for char in name:
    #         chars.add(char)

    # print("".join(sorted(chars)))

    word_count = Counter(names)

    # Сортируем слова по убыванию количества
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # draw_plot(sorted_word_count)
    # Выводим отсортированный список и их количество
    for word, count in sorted_word_count:
        print(f"{word}: {count}")

def print_all_symbols(label_file):
    with open(label_file) as json_file:
        data = json.load(json_file)

    names = list(data.values())
    simbols = set()
    for name in names:
        for symbol in name:
            simbols.add(symbol)

    print("".join(sorted(simbols)))


def by_lebels():
    with open("./hcr-archive-metric-book/data_mb/word_images/label_normalized.json") as json_file:
        data = json.load(json_file)

    names = list(data.values())

    word_count = Counter(names)

    # Сортируем слова по убыванию количества
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # draw_plot(sorted_word_count)
    # Выводим отсортированный список и их количество
    for word, count in sorted_word_count:
        files = get_keys_by_value(data, word)
        for file in files:
            dest = f'./hcr-archive-metric-book/data_mb/word_images/by_word/{word}/'
            os.makedirs(dest, exist_ok=True)
            shutil.copy(file, f'{dest}/{os.path.basename(file)}')
        print(f"{word}: {count}")


def draw_plot(sorted_word_count):
    # Псевдографика
    print("Распределение слов:")
    for word, count in sorted_word_count:
        bar = "#" * count
        print(f"{word}: {bar}")

    # # Разделяем слова и их количество на два отдельных списка
    # words, counts = zip(*sorted_word_count)
    #
    # # Создаем столбчатую диаграмму
    # plt.figure(figsize=(10, 6))
    # plt.bar(words, counts, color='skyblue')
    # plt.xlabel('Слово')
    # plt.ylabel('Количество')
    # plt.title('Распределение слов')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    #
    # # Отображаем график
    # plt.show()


def get_keys_by_value(d, value):
    return [key for key, val in d.items() if val == value]


if __name__ == '__main__':
    print_all_symbols("./hcr-archive-metric-book/data_mb/word_images/label.json")
    # by_lebels()
