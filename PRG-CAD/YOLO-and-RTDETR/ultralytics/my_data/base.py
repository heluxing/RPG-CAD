# Ultralytics YOLO 🚀, AGPL-3.0 license
import copy
import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

# $$$修改
from ultralytics.my_data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


def _draw_box(img4, labels):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # 创建一个matplotlib的图像对象
    fig, ax = plt.subplots(1)
    ax.imshow(img4)
    # 绘制边界框
    for box in labels["instances"].bboxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')  # 关闭坐标轴
    plt.show()

class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
            self,
            img_path,
            imgsz=640,
            cache=False,
            augment=True,
            hyp=DEFAULT_CFG,
            prefix="",
            rect=False,
            batch_size=16,
            stride=32,
            pad=0.5,
            single_cls=False,
            classes=None,
            fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
        # self.label_files = None

        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        # self.rect = rect
        self.rect = False
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        # self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.ims, self.im_hw0, self.im_hw =[[None] *(len(f)) for f in self.im_files],[[None] *(len(f)) for f in self.im_files],[[None] *(len(f)) for f in self.im_files]
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.npy_files=[]
        for i in range(len(self.im_files)):
            self.npy_files.append([Path(f).with_suffix(".npy") for f in self.im_files[i]])
        # self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if (self.cache == "ram" and self.check_cache_ram()) or self.cache == "disk":
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
        # self.near_img_index = self.get_near_img_index()  # 一个字典，得到每一个标签的前后几张图片的标签在self.label_files中的索引
        # self.sequences=

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    # 遍历root_dir下的所有子目录
                    for subdir, dirs, files in os.walk(p):
                        # 跳过根目录，因为我们只对子目录中的文件感兴趣
                        if subdir != str(p):
                            # 创建一个子列表来存储当前子目录中的文件路径
                            one_sequence = [os.path.join(subdir, file) for file in files]
                            # 将这个子列表添加到主列表中
                            f.append(one_sequence)


                            # f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            im_files = f
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        success = mem_required < mem.available  # to cache or not to cache, that is the question
        if not success:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
        return success

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index,图片数量/batch_size[0,0,..,1,1,...]表示有batchsize个图片为第一批，接着第二批。。。
        nb = bi[-1] + 1  # number of batches 一共要走这么多批

        # s = np.array([x.pop("shape") for x in self.labels])  # hw
        s = np.array([x[0].pop("shape") for x in self.labels])  # hw本来是每张图片的尺寸，这里一个序列给一个就行了
        ar = s[:, 0] / s[:, 1]  # aspect ratio 长宽比，全是1的ndarray
        # irect = ar.argsort()  # 这里应该是要按图片的长宽比排序，然后按这个顺序重新调整self.im_files， self.labels
        # self.im_files = [self.im_files[i] for i in irect]
        # self.labels = [self.labels[i] for i in irect]
        # ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb # 全是1，所以后面的注释掉了
        # for i in range(nb):
        #     ari = ar[bi == i]
        #     mini, maxi = ari.min(), ari.max()
        #     if maxi < 1:
        #         shapes[i] = [maxi, 1]
        #     elif mini > 1:
        #         shapes[i] = [1, 1 / mini]
        # [nb一个epoch的总批次,2] self.pad=0.5  self.stride=32
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image


    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        label_sequence=self.get_image_and_label(index)
        transformed_labels = self.transforms(label_sequence)


        import torch
        img_tensor_list = [label['img'] for label in transformed_labels]
        img_combined_tensor = torch.stack(img_tensor_list, dim=0)  # dim=0是因为我们想要将tensor堆叠在channel维度之前【5,3,512,512】
        cls_combined_tensor = [label['cls'].tolist() for label in transformed_labels]  # 这里就得到长度为5的list
        bboxes_combined_tensor = [label['bboxes'].tolist() for label in transformed_labels]
        copied_label = copy.deepcopy(transformed_labels[0])
        # 做预测时才打开它，训练时用不到它的
        # copied_label['im_file'] = [img['im_file'] for img in transformed_labels]
        copied_label['img'] = img_combined_tensor
        copied_label['cls'] = cls_combined_tensor
        copied_label['bboxes'] = bboxes_combined_tensor

        # 绘制边界框
        # before_boxes=self.draw_bbox(near_label_group[0]['img'][0], near_label_group[0]['bboxes'])
        # img_with_bboxes = self.draw_bbox(transformed_labels[0]['img'][0], transformed_labels[0]['bboxes'])
        # import matplotlib.pyplot as plt
        # before_boxes = self.draw_bbox( copied_label['img'][0].numpy(),  copied_label['bboxes'][0])
        # plt.imshow(before_boxes)
        # plt.show()

        # for box in copied_label["bboxes"]:
        #     if len(box)>0:
        #         with open('/home/HELUXING/Brain_project/YOLOv11/cls8_yolo/example——duich.txt', 'a', encoding='utf-8') as file:
        #             for boxi in box:
        #                 file.write(', '.join(map(str, boxi[:2])) + '\n')



        # if aa['bboxes'].shape[0]>0:
        #     with open('/home/HELUXING/Brain_project/YOLOv11/cls8_yolo/example.txt', 'a', encoding='utf-8') as file:
        #         for sublist in aa['bboxes'][:,0:2].tolist():
        #             file.write(', '.join(map(repr, sublist)) + '\n')
        #
        # return aa
        return copied_label
        # return self.transforms(self.get_image_and_label(index))

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

    def draw_bbox(self, img, bboxes, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制边界框
        :param img: 原始图像
        :param bboxes: 边界框列表，每个边界框是一个形如[center_x, center_y, width, height]的列表
        :param color: 边界框的颜色，默认为绿色
        :param thickness: 边界框的线条粗细
        :return: 绘制了边界框的图像
        """
        import cv2
        for bbox in bboxes:
            center_x, center_y, width, height = bbox

            # 将边界框坐标从比例转换为像素坐标
            x = int(center_x * img.shape[1])
            y = int(center_y * img.shape[0])
            w = int(width * img.shape[1])
            h = int(height * img.shape[0])

            # 计算边界框的左上角和右下角坐标
            x1 = x - w // 2
            y1 = y - h // 2
            x2 = x + w // 2
            y2 = y + h // 2

            # 确保边界框在图像内部
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        return img

    def get_near_img_index(self, sequence_width=5):
        near_img_index = {}
        for index in range(len(self.im_files)):
            # 'E:\\head_project\\数据\\315\\yolo_format_addnew\\labels\\train\\_1.2.840.113619.2.25.4.83427925.1699058132.40_012.txt'
            img_index = int(self.im_files[index][-7:-4])  # 得到不带序号和文件后缀的文件路径
            patient_path = self.im_files[index][:-7]  # 得到不带序号和文件后缀的文件路径
            near_label_group = []
            for i in range(int(sequence_width / 2) + 1):
                index_str = "{:03}".format(img_index - i)
                jpg_befor_path = patient_path + index_str + ".jpg"
                if jpg_befor_path in self.im_files:
                    near_label_group.insert(0, self.im_files.index(jpg_befor_path))
                    # images.insert(0, Image.open(img_befor_path))
                else:
                    near_label_group.insert(0, near_label_group[0])

            for i in range(1, int(sequence_width / 2) + 1):
                index_str = "{:03}".format(img_index + i)
                jpg_after_path = patient_path + index_str + ".jpg"
                if jpg_after_path in self.im_files:
                    near_label_group.append(self.im_files.index(jpg_after_path))
                    # images.append(Image.open(img_after_path))
                else:
                    near_label_group.append(near_label_group[-1])
            near_img_index[index] = near_label_group
        return near_img_index

    #######################################################
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def load_image_sequence(self, index, labels, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        for i in range(len(labels)):
            im, f, fn = self.ims[index][i], self.im_files[index][i], self.npy_files[index][i]
            if im is None:  # not cached in RAM
                if fn.exists():  # load npy
                    try:
                        im = np.load(fn)
                    except Exception as e:
                        LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                        Path(fn).unlink(missing_ok=True)
                        im = cv2.imread(f)  # BGR
                else:  # read image #因为图片没有npy缓存，所以走这里
                    im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")

                h0, w0 = im.shape[:2]  # orig hw
                if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                    im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

                # Add to buffer if training with augmentations
                if self.augment:
                    self.ims[index][i], self.im_hw0[index][i], self.im_hw[index][i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

                labels[i]["img"], labels[i]["ori_shape"], labels[i]["resized_shape"] =im, (h0, w0), im.shape[:2]
                labels[i]["ratio_pad"] = (
                    labels[i]["resized_shape"][0] / labels[i]["ori_shape"][0],
                    labels[i]["resized_shape"][1] / labels[i]["ori_shape"][1],
                )
                # for evaluation 这里可能会报错
                if self.rect:
                    labels[i]["rect_shape"] = self.batch_shapes[self.batch[index]] # [768 768]还是
            else:
                labels[i]["img"], labels[i]["ori_shape"], labels[i]["resized_shape"] = self.ims[index][i], self.im_hw0[index][i], self.im_hw[index][i]
                labels[i]["ratio_pad"] = (
                    labels[i]["resized_shape"][0] / labels[i]["ori_shape"][0],
                    labels[i]["resized_shape"][1] / labels[i]["ori_shape"][1],
                )
                # for evaluation 这里可能会报错
                if self.rect:
                    labels[i]["rect_shape"] = self.batch_shapes[self.batch[index]]
        # Add to buffer if training with augmentations
        # print(index)
        # print(self.buffer)
        # print("bbbbb")
        if (index not in self.buffer) and self.augment:
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer.pop(0) # 如果buffer里的长度超了，就要清掉第一个，同时把对应的ims等清掉，这样下一次进入for循环，im还是得到None，又可以进buffer
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = [None] * len(self.ims[j]), [None] * len(self.ims[j]), [
                        None] * len(self.ims[j])
        return labels

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label_sequence = deepcopy(self.labels[index])  # requires deepcopy() 这里得到一个序列的label list
        # label.pop("shape", None)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        [label_i.pop("shape", None) for label_i in label_sequence] # shape is for rect, remove it
        # label["img"]是图像数据

        label_sequence = self.load_image_sequence(index,label_sequence)

        return self.update_labels_info(label_sequence)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
