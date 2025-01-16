import os
import shutil
import random
import yaml


def move_files_with_labels(img_source_dir, label_source_dir, dest_dirs, percentages):
    """
    Переміщує файли з img_source_dir та відповідні файли з label_source_dir в одну з підпапок img і label в dest_dirs.

    :param img_source_dir: Шлях до папки з зображеннями для переміщення.
    :param label_source_dir: Шлях до папки з файлами міток для переміщення.
    :param dest_dirs: Список папок призначення (train, test, valid).
    :param percentages: Відсотки заповнення для кожної папки.
    """
    if not os.path.exists(img_source_dir):
        pass
        return

    if not os.path.exists(label_source_dir):
        pass
        return

    if len(dest_dirs) != len(percentages):
        pass
        return

    img_files = os.listdir(img_source_dir)
    random.shuffle(img_files)  # Перемішуємо файли для випадкового розподілу

    # Розраховуємо кількість файлів для кожної папки
    total_files = len(img_files)
    file_counts = [int(total_files * (p / 100)) for p in percentages]

    # Якщо є залишок файлів, додаємо їх до останньої папки
    remaining_files = total_files - sum(file_counts)
    if remaining_files > 0:
        file_counts[-1] += remaining_files

    start = 0
    for dest_dir, count in zip(dest_dirs, file_counts):
        img_dest_dir = os.path.join(dest_dir, "images")
        label_dest_dir = os.path.join(dest_dir, "labels")

        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(label_dest_dir, exist_ok=True)

        end = start + count
        for file_name in img_files[start:end]:
            img_source_path = os.path.join(img_source_dir, file_name)
            label_source_path = os.path.join(label_source_dir, os.path.splitext(file_name)[
                0] + ".txt")  # Припускаємо, що розширення файлів міток - .txt

            if os.path.isfile(img_source_path):
                img_dest_path = os.path.join(img_dest_dir, file_name)
                shutil.move(img_source_path, img_dest_path)

            if os.path.isfile(label_source_path):
                label_dest_path = os.path.join(label_dest_dir, os.path.basename(label_source_path))
                shutil.move(label_source_path, label_dest_path)

        start = end


def divide_into_sets():
    # Папки для зображень та міток
    img_source = "wheel/sets/analyzed/images"
    label_source = "wheel/sets/analyzed/labels"
    destinations = ["wheel/divided/train", "wheel/divided/test", "wheel/divided/valid"]
    percentages = [80, 15, 5]  # Відсотки для папок призначення

    # Створюємо папки призначення, якщо вони не існують
    for folder in destinations:
        os.makedirs(folder, exist_ok=True)

    # Переміщуємо файли
    move_files_with_labels(img_source, label_source, destinations, percentages)

    # train:../ train / images
    # val:../ valid / images
    # test:../ test / images
    #
    # nc: 8
    # names = ['1', '10', '100', '2', '200', '300', '5', '8']
    data = dict(
        train=destinations[0],
        val=destinations[2],
        test=destinations[1],

        nc=8,
        names=['1', '10', '100', '2', '200', '300', '5', '8']
    )

    with open('wheel/divided/data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=None, sort_keys=False)
