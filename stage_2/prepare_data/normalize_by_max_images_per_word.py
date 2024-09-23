import json
from json import JSONEncoder


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def normalize(label_json_source_file: str, label_json_dest_file, max_repeat_words=100):
    with open(label_json_source_file) as json_file:
        data = json.load(json_file)

    # Создаем словарь для отслеживания количества повторений значений
    value_counts = {}

    # Создаем новый словарь с ограниченным количеством повторений значений
    new_dict = {}
    for key, value in data.items():
        if value not in value_counts or value_counts[value] < max_repeat_words:
            new_dict[key] = value
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1

    with open(label_json_dest_file, 'w') as file:
        json.dump(new_dict, file, indent=2, cls=CustomEncoder)


if __name__ == '__main__':
    normalize("./hcr-archive-metric-book/data_mb/word_images/label.json",
              "./hcr-archive-metric-book/data_mb/word_images/label_normalized.json",
              100)
