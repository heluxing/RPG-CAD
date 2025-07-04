# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import repeat

import torch

from my_dataset.my_augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    v8_transforms,
)

from ...core import register
from ..dataloader import BaseCollateFunction, generate_scales

DATASET_CACHE_VERSION = "1.0.3"
import copy
import math
import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

# $$$修改
from my_dataset.my_data_utils import FORMATS_HELP_MSG
from my_dataset.my_data_utils import LOCAL_RANK, NUM_THREADS, TQDM

from my_dataset.my_data_utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image_label,
)

class ImageIDMapper:
    def __init__(self, mapping_file='/sequence_dataset-path/Annotatioins/image_id_mapping.json'):
        with open(mapping_file) as f:
            self.mapping = json.load(f)

    def get_image_id(self, image_path):
        return self.mapping.get(image_path)
mapper = ImageIDMapper()
def convert_bbox(bbox_list, img_size=512):
    """
    将归一化边界框标签转换为绝对坐标的[x0,y0,x1,y1]格式

    参数:
        bbox_list (list): 归一化边界框列表，格式为[[xc,yc,w,h],...]
        img_size (int): 图像尺寸（默认512）

    返回:
        torch.Tensor: 转换后的边界框张量，格式为[[x0,y0,x1,y1],...]
    """
    converted = []
    for bbox in bbox_list:
        xc, yc, w, h = bbox

        # 将归一化坐标转换为绝对坐标
        xc_abs = xc * img_size
        yc_abs = yc * img_size
        w_abs = w * img_size
        h_abs = h * img_size

        # 计算边界框坐标
        x0 = xc_abs - w_abs / 2
        y0 = yc_abs - h_abs / 2
        x1 = xc_abs + w_abs / 2
        y1 = yc_abs + h_abs / 2

        converted.append([x0, y0, x1, y1])

    # 转换为PyTorch张量
    return torch.tensor(converted, dtype=torch.float32)
def _draw_box(image, bboxes):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 将图片数据从 [3, 512, 512] 转换为 [512, 512, 3]，以便 matplotlib 显示
    image = np.transpose(image, (1, 2, 0))
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(1)
    # 显示图片
    ax.imshow(image)
    # 遍历边界框数据，绘制每个边界框
    for bbox in bboxes:
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox
        x, y, w, h = x * 512, y * 512, w * 512, h * 512
        # 创建一个矩形补丁，表示边界框
        # 注意：matplotlib 的 Rectangle 需要左下角坐标和宽度、高度
        # 因此，需要将 xywh 转换为 xyxy 格式，或者直接计算左下角坐标
        rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # 设置坐标轴范围，确保图片完整显示
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # 因为 imshow 的 y 轴默认是向下增长的，这里设置为向上增长以便正确显示

    # 显示图形
    plt.show()


@register()
class MyDataset(Dataset):
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
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.

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
            imgsz=512,
            augment=True,
            hyp=None,
            prefix="",
            batch_size=16,
            single_cls=False,
            data=None
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        default_data = {
            'train': '/sequence_dataset-path/images/train_6',
            # 'val': '/home/HELUXING/Brain_project/data/sequence_dataset_len6_cls8/images/val_6',
            'val': '/sequence_dataset-path/images/test',
            'test': '/sequence_dataset-path/images/test',
            'names': {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4',
                      5: 'class_5', 6: 'class_6', 7: 'class_7',},
            'yaml_file': '/sequence_dataset-path/sequence_dataset_8cls.yaml',
            'nc': 12}
        from types import SimpleNamespace

        # 数据增强超参数
        default_hyp = SimpleNamespace(
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=1.0
        )
        self.data = data if data else default_data
        _hyp = hyp if hyp else default_hyp
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        # 根据图片路径：images/train或images/val来将图片路径放到一个一个二维列表中[[序列0的图片路径0,序列0的图片路径1...],[序列1的图片路径0...]...]
        self.im_files = self.get_img_files(self.img_path)
        # 将标签也放入一个二维列表中
        self.labels = self.get_labels()
        self.ni = len(self.labels)  # 序列数量
        self.batch_size = batch_size

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size，设置一个buffer，马赛克增强时从里头取样
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        # self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.ims, self.im_hw0, self.im_hw = [[None] * (len(f)) for f in self.im_files], [[None] * (len(f)) for f in
                                                                                         self.im_files], [
                                                [None] * (len(f)) for f in self.im_files]
        # Transforms
        self.transforms = self.build_transforms(hyp=_hyp)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        label_sequence = self.get_image_and_label(index)
        transformed_labels = self.transforms(label_sequence)

        import torch
        img_tensor_list = [label['img'] for label in transformed_labels]
        img_combined_tensor = torch.stack(img_tensor_list, dim=0)  # dim=0是因为我们想要将tensor堆叠在channel维度之前【5,3,512,512】
        cls_combined = [label['cls'].tolist() for label in transformed_labels]  # 这里就得到长度为5的list
        bboxes_combined = [label['bboxes'].tolist() for label in transformed_labels]
        copied_label = copy.deepcopy(transformed_labels[0])
        # 做预测时才打开它，训练时用不到它的
        # copied_label['im_file'] = [img['im_file'] for img in transformed_labels]
        copied_label['img'] = img_combined_tensor  # [6,3,512,512]
        copied_label['cls'] = cls_combined  # 是一个list [[], [], [], [], [[0.0], [3.0]], []]
        copied_label[
            'bboxes'] = bboxes_combined  # [[], [], [], [], [[0.06522676348686218, 0.8330007791519165, 0.03523772954940796, 0.0396420955657959], [0.028208404779434204, 0.6634198427200317, 0.05641680955886841, 0.0792844295501709]], []]

        img_combined_tensor = img_combined_tensor.float() / 255
        # 绘制边界框,调试用
        # img_with_bboxes = _draw_box(copied_label['img'][0].numpy(), bboxes_combined[0])

        img_path_list=[img['im_file'] for img in transformed_labels]
        return img_combined_tensor, bboxes_combined, cls_combined,img_path_list
        # 图片/边框坐标/类别都在属性里
        # return copied_label

    # $
    def load_image_sequence(self, index, labels, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        for i in range(len(labels)):
            im, f = self.ims[index][i], self.im_files[index][i]
            if im is None:  # not cached in RAM
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
                    self.ims[index][i], self.im_hw0[index][i], self.im_hw[index][i] = im, (h0, w0), im.shape[
                                                                                                    :2]  # im, hw_original, hw_resized

                labels[i]["img"], labels[i]["ori_shape"], labels[i]["resized_shape"] = im, (h0, w0), im.shape[:2]
                labels[i]["ratio_pad"] = (
                    labels[i]["resized_shape"][0] / labels[i]["ori_shape"][0],
                    labels[i]["resized_shape"][1] / labels[i]["ori_shape"][1],
                )

            else:
                labels[i]["img"], labels[i]["ori_shape"], labels[i]["resized_shape"] = self.ims[index][i], \
                                                                                       self.im_hw0[index][i], \
                                                                                       self.im_hw[index][i]
                labels[i]["ratio_pad"] = (
                    labels[i]["resized_shape"][0] / labels[i]["ori_shape"][0],
                    labels[i]["resized_shape"][1] / labels[i]["ori_shape"][1],
                )

        # Add to buffer if training with augmentations
        if (index not in self.buffer) and self.augment:
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer.pop(0)  # 如果buffer里的长度超了，就要清掉第一个，同时把对应的ims等清掉，这样下一次进入for循环，im还是得到None，又可以进buffer
                self.ims[j], self.im_hw0[j], self.im_hw[j] = [None] * len(self.ims[j]), [None] * len(self.ims[j]), [
                    None] * len(self.ims[j])
        return labels

    # $
    def update_labels_info(self, labels):  # labels是一个序列的list
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        for label in labels:
            bboxes = label.pop("bboxes")
            bbox_format = label.pop("bbox_format")
            normalized = label.pop("normalized")

            label["instances"] = Instances(bboxes, bbox_format=bbox_format, normalized=normalized)
        return labels

    # $
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label_sequence = deepcopy(self.labels[index])  # requires deepcopy() 这里得到一个序列的label list
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        [label_i.pop("shape", None) for label_i in label_sequence]  # shape is for rect, remove it
        # label["img"]是图像数据

        # 读取的图像数据和图像尺寸信息
        label_sequence = self.load_image_sequence(index, label_sequence)
        # 将图像的边界框也放进去
        return self.update_labels_info(label_sequence)

    # $
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

        return im_files

    # $
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

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    # $
    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_files, lbs, shapes, nms, nfs, nes, ncs, msg in pbar:  # 这里的im_file是list，一个序列的所有图片
                nm += nms
                nf += nfs
                ne += nes
                nc += ncs
                if im_files:
                    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    x["labels"].append(
                        [{
                            "im_file": im_files[i],
                            "shape": shapes[i],
                            "cls": lbs[i][:, 0:1],  # n, 1
                            "bboxes": lbs[i][:, 1:],  # n, 4
                            "normalized": True,
                            "bbox_format": "xywh",
                        } for i in range(len(im_files))]
                    )
                if msg:
                    msgs.append(msg[0])
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        # if msgs:
        #     LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        # x["hash"] = get_hash(self.label_files + self.im_files)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        x["hash"] = [get_hash(self.label_files[i] + self.im_files[i]) for i in range(len(self.im_files))]
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    # $
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.label_files = []  # 这里改为了二维数组，每个子数组里放一个序列的标签，对应的缓存也要改，或者等缓存取出来之后再组合
        for sequence_im_file in self.im_files:
            # 通过图片路径找到labels路径
            self.label_files.append(img2label_paths(sequence_im_file))
        cache_path = Path(self.label_files[0][0]).parent.parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            # if cache["msgs"]:
            #     LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        # self.im_files = [lb["im_file"] for lb in labels]  # update im_files
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        for i in range(len(labels)):
            self.im_files[i] = [lb["im_file"] for lb in labels[i]]

        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment else 0.0
            hyp.mixup = hyp.mixup if self.augment else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    @staticmethod
    def collate_fn_yolov8(batch):  # batch 是个list，元素为从base.py中的get_item中返回值
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        # keys = batch[0][0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            new_batch[k] = value

        batch_idx_list = []
        cls_list = []
        bboxes_list = []
        img_depth = len(batch[0]['cls'])
        for i in range(len(batch)):  # 第i批次第j张图的第k个框
            for j in range(len(batch[i]['cls'])):
                for k in range(len(batch[i]['cls'][j])):
                    cls_list.append(batch[i]['cls'][j][k])
                    bboxes_list.append(batch[i]['bboxes'][j][k])
                    batch_idx_list.append(i * img_depth + j)  # 每个批次5张
        new_batch['cls'] = torch.tensor(cls_list)
        new_batch['bboxes'] = torch.tensor(bboxes_list)
        new_batch['batch_idx'] = torch.tensor(batch_idx_list)
        new_batch['img'] = new_batch['img'].reshape(-1, new_batch['img'].shape[2], new_batch['img'].shape[3],
                                                    new_batch['img'].shape[4])
        return new_batch

    @staticmethod
    def collate_fn_retinanet(batch):
        # 解压batch数据：将多个(img, box, cls)分开存储
        imgs, boxes, clss = zip(*batch)
        img_h = 512
        img_w = 512
        # 处理图像数据 --------------------------------------------------------
        # 将每个样本的img（形状[6,3,512,512]）展开为[6,3,512,512]
        # 然后将所有样本的展开后的img沿第0维拼接
        img_tensor = torch.cat([img.view(-1, 3, 512, 512) for img in imgs], dim=0)
        # 最终形状：[batch_size*6, 3, 512, 512]

        # 处理标签数据 --------------------------------------------------------
        labels = []
        # 遍历每个样本（每个样本包含6张图的标注）
        for sample_boxes, sample_cls in zip(boxes, clss):
            # 遍历样本中的每张图片（每个样本有6张图）
            for img_boxes, img_cls in zip(sample_boxes, sample_cls):
                # 收集当前图片的所有边界框标签
                img_labels = []

                # 遍历当前图片的每个边界框（可能多个框）
                for box, cls_id in zip(img_boxes, img_cls):
                    # 将xywh转为x0y0x1y1格式
                    x, y, w, h = box
                    x0, y0 = int((x - w / 2) * img_w), int((y - h / 2) * img_w)
                    x1, y1 = int((x + w / 2) * img_h), int((y + h / 2) * img_h)
                    # 创建单目标 Tensor [5]
                    obj_tensor = torch.tensor(np.array([x0, y0, x1, y1, cls_id[0]]), dtype=torch.float32)
                    img_labels.append(obj_tensor)

                # 将当前图片的所有目标堆叠为 Tensor [N,5]（若无目标则创建空Tensor）
                if len(img_labels) > 0:
                    img_labels = torch.stack(img_labels, dim=0)  # shape [N,5]
                else:
                    img_labels = torch.tensor(np.array([]), dtype=torch.float32)  # 空Tensor [0,5]

                labels.append(img_labels)

        return img_tensor, labels


    @staticmethod
    def collate_fn_DFINE(batch):
        # 作为验证用的dataloader，其实只会提供图片数据，image_id和原始尺寸，所以其他的都可以不要了
        # 合并图像数据
        imgs = torch.cat([item[0] for item in batch], dim=0)  # 形状为 [batch_size*6, 3, 512, 512]

        targets = []
        for item in batch:
            # boxes = item[1]  # 每个样本的box列表，长度6
            # clses = item[2]  # 每个样本的cls列表，长度6
            img_path_list = item[3]
            # 处理每个子图（共6张）
            for i in range(len(img_path_list)):
                # img_boxes = boxes[i]  # 当前子图的框列表
                # img_cls = clses[i]  # 当前子图的类别列表
                path_parts = img_path_list[i].split('/')
                images_index = path_parts.index('images')
                image_id = mapper.get_image_id('/'.join(path_parts[images_index:]))

                # 处理boxes
                # if len(img_boxes) == 0:
                #     boxes_tensor = torch.zeros((0, 4))
                # else:
                #     boxes_tensor = convert_bbox(img_boxes)

                # 处理labels
                # if len(img_cls) == 0:
                #     labels_tensor = torch.zeros(0, dtype=torch.int64)
                # else:
                #     labels_tensor = torch.tensor([int(cls[0]) for cls in img_cls], dtype=torch.int64)

                # 计算area（原图尺寸512x512，需乘以面积）
                # if boxes_tensor.shape[0] == 0:
                #     area = torch.zeros(0)
                # else:
                #     w = boxes_tensor[:, 2]
                #     h = boxes_tensor[:, 3]
                #     area = (w * h) * (512 * 512)  # 计算实际面积
                #     area = area.to(torch.float32)

                # iscrowd全为0
                # iscrowd = torch.zeros((boxes_tensor.shape[0]))

                # orig_size固定为[512, 512]
                orig_size = torch.tensor([512, 512])

                # idx为image_id减1
                # idx = image_id - 1

                # 构建目标字典
                target = {
                    # 'boxes': boxes_tensor,
                    # 'labels': labels_tensor,
                    'image_id': torch.tensor(image_id),
                    # 'area': area,
                    # 'iscrowd': iscrowd,
                    'orig_size': orig_size,
                    # 'idx': idx
                }
                targets.append(target)
        return imgs, targets

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

@register()
class collate_fn_DFINE(BaseCollateFunction):
    def __init__(
            self,
            stop_epoch=None,
            ema_restart_decay=0.9999,
            base_size=512,
            base_size_repeat=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay


        # self.interpolation = interpolation
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, batch):
        # 合并图像数据
        imgs = torch.cat([item[0] for item in batch], dim=0)  # 形状为 [batch_size*6, 3, 512, 512]

        targets = []
        for item in batch:
            boxes = item[1]  # 每个样本的box列表，长度6
            clses = item[2]  # 每个样本的cls列表，长度6
            img_path_list=item[3]
            # 处理每个子图（共6张）
            for i in range(6):
                img_boxes = boxes[i]  # 当前子图的框列表
                img_cls = clses[i]  # 当前子图的类别列表
                # img_path=img_path_list[i]
                path_parts = img_path_list[i].split('/')
                images_index = path_parts.index('images')
                image_id = mapper.get_image_id('/'.join(path_parts[images_index:]))

                # 处理boxes
                if len(img_boxes) == 0:
                    boxes_tensor = torch.zeros((0, 4))
                else:
                    boxes_tensor = torch.tensor(img_boxes)

                # 处理labels
                if len(img_cls) == 0:
                    labels_tensor = torch.zeros(0, dtype=torch.int64)
                else:
                    labels_tensor = torch.tensor([int(cls[0]) for cls in img_cls], dtype=torch.int64)

                # 计算area（原图尺寸512x512，需乘以面积）
                if boxes_tensor.shape[0] == 0:
                    area = torch.zeros(0)
                else:
                    w = boxes_tensor[:, 2]
                    h = boxes_tensor[:, 3]
                    area = (w * h) * (512 * 512)  # 计算实际面积
                    area = area.to(torch.float32)

                # iscrowd全为0
                iscrowd = torch.zeros((boxes_tensor.shape[0]))

                # orig_size固定为[512, 512]
                orig_size = torch.tensor([512, 512])

                # idx为image_id减1
                # idx = image_id - 1

                # 构建目标字典
                target = {
                    'boxes': boxes_tensor,
                    'labels': labels_tensor,
                    'image_id': torch.tensor(image_id),
                    'area': area,
                    'iscrowd': iscrowd,
                    'orig_size': orig_size,
                    # 'idx': idx
                }
                targets.append(target)
        return imgs, targets

def get_val_dataloader():
    data = {
        'train': '/sequence_dataset-path/images/train_6',
        'val': '/sequence_dataset-path/images/val_6',
        'test': '/sequence_dataset-path/images/test',
        'names': {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4',
                      5: 'class_5', 6: 'class_6', 7: 'class_7',},
        'yaml_file': '/sequence_dataset-path/sequence_dataset_8cls.yaml',
        'nc': 8}
    from types import SimpleNamespace

    # 数据增强超参数
    hyp = SimpleNamespace(
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=1.0
    )
    batch_size = 24#测试的时候给1，这里用于验证时可设大点
    # batch_size = 1#测试的时候给1，这里用于验证时可设大点
    # 创建数据集
    val_dataset = MyDataset(
        img_path=data['val'],
        # img_path=data['test'],
        imgsz=512,
        batch_size=batch_size,
        augment=False,  # augmentation
        hyp=hyp,  # TODO: probably add a get_hyps_from_cfg function
        data=data
    )

    # 创建数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=getattr(val_dataset, "collate_fn_DFINE", None),
        num_workers=8
    )
    return val_dataloader
