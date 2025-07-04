# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
# from ultralytics.utils.torch_utils import TORCHVISION_0_18
from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

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
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            # for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:  # 这里的im_file是list，一个序列的所有图片
            # im_files, lbs, shapes, segmentss, keypointss, nms, nfs, nes, ncs, msgs
            for im_files, lbs, shapes, segmentss, keypointss, nms, nfs, nes, ncs, msg in pbar:  # 这里的im_file是list，一个序列的所有图片
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
                            "segments": segmentss[i],
                            "keypoints": keypointss[i],
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

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.label_files = []  # 这里改为了二维数组，每个子数组里放一个序列的标签，对应的缓存也要改，或者等缓存取出来之后再组合
        for sequence_im_file in self.im_files:
            self.label_files.append(img2label_paths(sequence_im_file))
        # self.label_files = img2label_paths(self.im_files)
        # cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        cache_path = Path(self.label_files[0][0]).parent.parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
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
            self.im_files[i]=[lb["im_file"] for lb in labels[i]]


        # Check if the dataset is all boxes or all segments
        # lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        # len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        # if len_segments and len_boxes != len_segments:
        #     LOGGER.warning(
        #         f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
        #         f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
        #         "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
        #     )
        #     for lb in labels:
        #         lb["segments"] = []
        # if len_cls == 0:
        #     LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
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

    def update_labels_info(self, labels):# labels是一个序列的list
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        for label in labels:
            bboxes = label.pop("bboxes")
            segments = label.pop("segments", [])
            keypoints = label.pop("keypoints", None)
            bbox_format = label.pop("bbox_format")
            normalized = label.pop("normalized")

            # NOTE: do NOT resample oriented boxes
            segment_resamples = 100 if self.use_obb else 1000
            if len(segments) > 0:
                # list[np.array(1000, 2)] * num_samples
                # (N, 1000, 2)
                segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
            else:
                segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
            label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return labels

    @staticmethod
    def collate_fn(batch): # batch 是个list，元素为从base.py中的get_item中返回值
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        # keys = batch[0][0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
                # print("&&&&&&&&&&&&&"+str(value.shape))
            # if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
            #     # $$$$$$$$$$$$$$$$$$$$$这里不再采用tensor格式的边界框信息了，直接用list
            #     # value = torch.cat(value, 0)
            #     value = list(values[i])
            new_batch[k] = value
        # new_batch["batch_idx"] = list(new_batch["batch_idx"])
        # for i in range(len(new_batch["batch_idx"])):
        #     new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        # new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

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

