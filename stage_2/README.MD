Перевод текста

Для подготовки dataset - вырезанных слов из картинки, можно использовать тот же coco формат, в метаданные добавлять перевод каждого слова.


Профилировка запуска скрипта:

python.exe -m memory_profiler .\stage_2\prepare_data\prepare_cut_images_words.py
Filename: .\stage_2\prepare_data\prepare_cut_images_words.py

```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    77     28.4 MiB     28.4 MiB           1   @profile
    78                                         def parseCocoJson(full_file_name: str) -> []:
    79     28.4 MiB      0.0 MiB           1       result = []
    80
    81     36.2 MiB      0.0 MiB           2       with open(full_file_name) as json_file:  
    82     35.6 MiB      7.2 MiB           1           data = json.load(json_file)
    83
    84     35.6 MiB      0.0 MiB           1           annotations = data['annotations']    
    85
    86     36.2 MiB      0.0 MiB          13           for image in data['images']:
    87     36.2 MiB      0.0 MiB          12               image_id = image['id']
    88
    89     36.2 MiB      0.0 MiB          12               words = []
    90                                                     # nested loop filter
    91     36.2 MiB      0.0 MiB       23784               for item in annotations:
    92     36.2 MiB      0.0 MiB       23772                   if item['image_id'] == image_id:
    93     36.2 MiB      0.0 MiB        1981                       original_bbox = item['bbox']
    94     36.2 MiB      0.6 MiB        3962                       words.append(Word(bbox=Bbox(x=original_bbox[0], y=original_bbox[1], width=original_bbox[2], height=original_bbox[3]),
    95     36.2 MiB      0.0 MiB        1981                                         segmentation=item['segmentation'][0]))
    96
    97     36.2 MiB      0.0 MiB          24               cocoJson = CocoJson(path=image['path'],
    98     36.2 MiB      0.0 MiB          12                                   file_name=image['file_name'],
    99     36.2 MiB      0.0 MiB          12                                   width=image['width'],
   100     36.2 MiB      0.0 MiB          12                                   height=image['height'],
   101     36.2 MiB      0.0 MiB          12                                   words=words)
   102     36.2 MiB      0.0 MiB          12               result.append(cocoJson)
   103
   104     36.2 MiB      0.0 MiB           1       return result


Filename: .\stage_2\prepare_data\prepare_cut_images_words.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
   107     28.3 MiB     28.3 MiB           1   @profile
   108                                         def prepare_images_for_dataset(json_full_file_name: str, dataset_dir_name: str):
   109
   110     36.2 MiB      7.9 MiB           1       metadata = parseCocoJson(json_full_file_name)
   111
   112     36.3 MiB      0.0 MiB           2       with open('./stage_2/prepare_data/stage_2_annotations.json', 'w') as file:  
   113     36.3 MiB      0.1 MiB           2           json.dump({"annotations": metadata}, file,
   114     36.2 MiB      0.0 MiB           1                     indent=2, cls=CustomEncoder)
   ```

Обработка 12-ти образов
Подготовка
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 1199.77it/s]

Вырезание и сохранения слов для каждого из 12-ти образов
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [06:28<00:00, 32.35s/it]

Кол-во слов ~1 981 

**Внимание!!!**

В примере базового pipline для обучения второго этапа было **66 599** картинок со словами!