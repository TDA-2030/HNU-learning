import json
import os
import sys
import random
import warnings
import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil
"""
Download labelme with follow command: 
wget -c https://github.com/wkentaro/labelme/releases/download/v5.0.5/labelme-Linux
"""


def main():
    parser = argparse.ArgumentParser(
        usage=
        f'python {os.path.basename(__file__)} -n <label names in .txt> <dataset dir1> <dataset dir2> ...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'datas',
        default='./dataset',
        nargs='+',
        metavar="DIR",
        help='input dataset paths with annotated files',
        type=str,
    )
    parser.add_argument(
        '-n'
        '--name-file',
        default='./names.txt',
        nargs='?',
        dest='name_file',
        metavar="FILE",
        help='input the object name file ',
        type=str,
    )
    parser.add_argument('-r',
                        '--ratio',
                        type=float,
                        help='Validation ratio',
                        default=0.1)
    parser.add_argument(
        '--combine',
        default=False,
        dest='combine',
        action="store_true",
        help='combine all input dataset',
    )
    parser.add_argument(
        '-v'
        '--visualize',
        default=False,
        dest='visualize',
        action="store_true",
        help='generate visualize image',
    )
    parser.add_argument(
        '--add-flags',
        default=False,
        dest='add_flags',
        action="store_true",
        help='add classification flags from label of bounding box',
    )
    parser.add_argument(
        '--pick',
        default=False,
        dest='pick_images',
        action="store_true",
        help='pick the annotated images and labels to a new folder',
    )

    args = parser.parse_args()

    for dir in args.datas:
        json_to_txt(dir, args.name_file, args.add_flags, args.pick_images)
        dir = Path(dir)
        if args.combine:
            out_dir = dir.parent / Path('dataset_combine')
        else:
            out_dir = dir.parent / Path(dir.name + '_out')
        data_to_yolo_dataset(str(dir), out_dir, args.ratio, args.combine)

    if args.visualize:
        for d in args.datas:
            d = Path(d).resolve()
            out_dir = str(d.parent / (d.name + "_visualize"))
            visualize_annotation(d, out_dir, args.name_file)


def _convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def _read_name_file(name_path: Path) -> tuple:
    names = []
    with open(name_path, "r") as name_file:
        for name in name_file:
            names.append(name.replace("\n", "").strip())
    return names


def json_to_txt(path: str, name_file: str, add_flags: bool, pick_images: bool):
    """

    """
    print(f"convert dirs: {path}")
    names = _read_name_file(name_file)

    path: Path = Path(path).resolve()
    if pick_images:
        picked_dir = path.parent / (path.name + '_picked')
        if os.path.exists(picked_dir):
            shutil.rmtree(picked_dir)
        os.mkdir(picked_dir)
    for f in (os.listdir(path)):
        if (f.split('.')[-1] in ["json"]):
            # pick the annotated images
            if pick_images:
                img_name = Path(f.rstrip(".json") + ".jpg")
                shutil.copy(path / f, picked_dir / f)
                shutil.copy(path / img_name, picked_dir / img_name)

            # read json
            txt_name = f.rstrip(".json") + ".txt"
            txt_outpath = os.path.join(path, txt_name)
            txt_outfile = open(txt_outpath, "w")
            json_file_path = os.path.join(path, f)

            json_file = open(json_file_path)
            js = json.load(json_file)
            json_file.close()
            del json_file

            label_list = []
            for item in js["shapes"]:
                label = item["label"]
                label_list.append(label)
                for i, name in enumerate(names):
                    if label == name:
                        cls = str(i)

                point = item["points"]
                x1 = point[0][0]
                y1 = point[0][1]
                x2 = point[1][0]
                y2 = point[1][1]

                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)

                height = js["imageHeight"]
                width = js["imageWidth"]
                b = (xmin, xmax, ymin, ymax)
                bb = _convert((width, height), b)

                txt_outfile.write(cls + " " + " ".join([str(a)
                                                        for a in bb]) + '\n')
            # add flags from labels of shape
            if add_flags:
                flags = {}
                for n in names:
                    flags[n] = True if n in label_list else False
                js["flags"] = flags
                jsonString = json.dumps(js, indent=2, ensure_ascii=True)
                new_json_file = open(json_file_path, "w")
                new_json_file.write(jsonString)
                new_json_file.close()
    txt_outfile.close()


def show_progress(i: int, prefix: str = "progress:"):
    i = int(i)
    print("{} {}%: ".format(prefix, i), "▋" * (i // 2), end="\r")
    sys.stdout.flush()


def annotate_img(args):
    input_folder, output_folder, file, names = args['input_folder'], args[
        'output_folder'], args['file'], args['names']
    txt_file = open(os.path.join(input_folder, file), 'r')
    lines = txt_file.readlines()
    txt_file.close()
    img = cv2.imread(os.path.join(input_folder, file[:-3] + 'jpg'))
    img_hei, img_wid = img.shape[:2]
    pid=os.getpid()
    try:
        for line in lines:
            sample = line.split(' ')
            if len(sample) < 1:
                break

            data = np.array(list(map(float, sample)), dtype=float)
            class_ID = data[0]
            x_center = data[1] * img_wid
            y_center = data[2] * img_hei
            wid = data[3] * img_wid
            hei = data[4] * img_hei
            x1, x2 = x_center - wid / 2, x_center + wid // 2
            y1, y2 = y_center - hei // 2, y_center + hei // 2

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                        2)
            cv2.putText(img,
                        f"{str(int(class_ID))}:{names[int(class_ID)]}",
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255),
                        thickness=3,
                        lineType=cv2.LINE_AA)
            cv2.putText(img,
                        f"{str(int(class_ID))}:{names[int(class_ID)]}",
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_folder, file[:-3] + 'jpg'), img)

    except KeyboardInterrupt:
        print('process %d KeyboardInterrupt ...'%pid)


def visualize_annotation(input_folder, output_folder, name_file):
    print(f"Visualize dir {input_folder}")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    names = _read_name_file(name_file)

    args = []
    for file in os.listdir(input_folder):
        if file.split(".")[-1] in "txt":
            arg = {}
            arg['output_folder'] = output_folder
            arg['input_folder'] = input_folder
            arg['names'] = names
            arg['file'] = file
            args.append(arg)

    index = 0
    from multiprocessing import Pool
    try:
        with Pool(processes=None) as p:
            for _ in p.imap(annotate_img, args):
                show_progress(index * 100 / len(args), "visualize progress:")
                index += 1
    except KeyboardInterrupt:
        print('main process catch keyboardinterupterror')
    except Exception as e:
        print(e)


def data_to_yolo_dataset(data_dir: str, out_dir: str, val_ratio: float,
                         auto_rename: bool) -> None:
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    images_dir = Path("images")
    labels_dir = Path("labels")
    try:
        (out_dir / images_dir / "train").mkdir(parents=True)
        (out_dir / images_dir / "valid").mkdir(parents=True)
        (out_dir / labels_dir / "train").mkdir(parents=True)
        (out_dir / labels_dir / "valid").mkdir(parents=True)
    except FileExistsError as e:
        warnings.warn(f"The out directory already exist {e}")

    for i, l in enumerate(data_dir.glob('*.txt')):
        train_or_valid = 'train' if random.random() > val_ratio else 'valid'
        l_path = str(out_dir / labels_dir / train_or_valid / l.name)
        img = l.parent / Path(l.stem + '.jpg')
        img_path = str(out_dir / images_dir / train_or_valid / img.name)

        if auto_rename and os.path.exists(l_path):
            l_path = l_path[:-4] + "_1" + l_path[-4:]
            img_path = img_path[:-4] + "_1" + img_path[-4:]
        shutil.copy(str(l), l_path)
        shutil.copy(str(img), img_path)


if __name__ == '__main__':
    main()
