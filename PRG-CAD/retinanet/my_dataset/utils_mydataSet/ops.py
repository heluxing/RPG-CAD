# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch

from my_dataset.utils_mydataSet.metrics import batch_probiou


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```python
        from ultralytics.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (torch.Tensor): the segment label
        width (int): the width of the image. Defaults to 640
        height (int): The height of the image. Defaults to 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for obbs, powered by probiou and fast-nms.

    Args:
        boxes (torch.Tensor): (N, 5), xywhr.
        scores (torch.Tensor): (N, ).
        threshold (float): IoU threshold.

    Returns:
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]

def nms_new(boxes): #(xyxy, conf, cls)
    classes, indices = torch.unique(boxes[:, 5], sorted=True, return_inverse=True)
    # Compute the sort indices based on confidence within each class
    cls_sort_indices = (boxes[:, 5]).argsort(dim=0)
    sorted_boxes = boxes[cls_sort_indices]  # Apply the sort indices to the original indices
    # 存储最小外接矩形框的列表
    bounding_boxes = []
    for cls_idx, cls in enumerate(classes):
        # Get the indices of boxes for the current class
        class_indices = (sorted_boxes[:, 5] == cls).nonzero(as_tuple=True)[0]
        class_boxes = sorted_boxes[class_indices]

        # -----------------------------总的---------------------------------------
        # 创建一个空的[512, 512]张量来存储置信度累加值
        confidence_sum = torch.zeros((512, 512), dtype=torch.float32,device=boxes.device)
        # 创建一个空的[512, 512]张量来存储置信度最大值
        confidence_max_map = torch.zeros((512, 512), dtype=torch.float32,device=boxes.device)
        # 遍历每个矩形框，累加置信度并更新每个像素点的置信度最大值
        for box in class_boxes:
            x0, y0, x1, y1, conf, _ = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            confidence_sum[y0:y1, x0:x1] += conf
            confidence_max_map[y0:y1, x0:x1] = torch.max(confidence_max_map[y0:y1, x0:x1], conf)
        # -----------------------------查找联通区域，每个联通区域都会设定一个其中置信度最大的框作为置信度阈值---------------------------------------
        # 将confidence_sum和confidence_max_map转换为numpy数组以便使用ndimage进行处理
        confidence_sum_np = confidence_sum.cpu().numpy()
        # 设定置信度阈值
        confidence_threshold = 0.001  # 示例阈值，实际使用时替换为您的阈值
        # 将置信度图二值化
        _, binary_map = cv2.threshold(confidence_sum_np, confidence_threshold, 255, cv2.THRESH_BINARY)
        # 查找轮廓
        binary_map = binary_map.astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 创建一个空的[512, 512]张量来存储每个点的置信阈值
        confidence_th = torch.zeros((512, 512), dtype=torch.float32,device=boxes.device)
        binary_map_all = np.zeros((512, 512), dtype=np.float32)
        # 遍历每个轮廓，计算最小外接矩形框
        for contour in contours:
            x0, y0, w, h = cv2.boundingRect(contour)  # 求得几个框重叠区域的最大外接矩形框
            x1, y1 = x0 + w, y0 + h  # 置信度大于0.001筛选后，每个联通区域的最大外接矩形
            max_conf_in_box = torch.max(confidence_max_map[y0:y1, x0:x1])  # 最大外接矩形框内单个框置信度值最大值
            confidence_th[y0:y1, x0:x1] = max_conf_in_box  # 最大外接矩形框的置信度阈值设置为其中单个框置信度值最大值
            binary_map_tem = np.zeros((512, 512), dtype=np.float32)
            # 临时存储该外接矩形区域的每个像素点的置信度叠加值
            binary_map_tem[y0:y1, x0:x1] = confidence_sum_np[y0:y1, x0:x1]
            # 该外接矩形区域的每个像素点的置信度叠加值超过单个框置信度最大值时，要保留这个区域
            # max_conf_in_box.item()-0.001是因为二值化的时候取的大于阈值，没取到最大的那个置信度，所以减去一个小的值
            _, binary_map_tem = cv2.threshold(binary_map_tem, max_conf_in_box.item() - 0.001, 255, cv2.THRESH_BINARY)
            # 将这个外接矩形框内要保留的区域添加到总的，保留的区域不一定是矩形，所以还要进一步求外接矩形
            binary_map_all[y0:y1, x0:x1] = binary_map_tem[y0:y1, x0:x1]
        # 查找轮廓
        binary_map_all = binary_map_all.astype(np.uint8)
        contours, _ = cv2.findContours(binary_map_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历每个轮廓，计算最小外接矩形框
        for contour in contours:
            x0, y0, w, h = cv2.boundingRect(contour)
            x1, y1 = x0 + w, y0 + h
            max_conf_in_box = torch.max(confidence_max_map[y0:y1, x0:x1])
            bounding_boxes.append((x0, y0, x1, y1, max_conf_in_box.item(),cls.item()))
    return torch.tensor(bounding_boxes)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:#不走
        classes = torch.tensor(classes, device=prediction.device)
    # 不走
    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,16,5376) to shape(1,5376,16)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # 对每张图片的预测结果依次进行处理
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # 本来要每一类单独进行，但它对每一类的坐标都做了不同偏移，所以可以放到一起做NMS
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections,i得到的是索引

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # 对它进行处理，box,conf,cls，box：xyxy
        # xx = torch.stack(remove_contained_boxes1(x),dim=0)
        # output[xi] = xx  #

        output[xi] = x[i]#

        # refine_boxes=nms_new(x).to(device=x.device)
        # output[xi] =refine_boxes #
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
        #     break  # time limit exceeded

    return output

def non_max_suppression11(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
):

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:#不走
        classes = torch.tensor(classes, device=prediction.device)
    # 不走
    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,16,5376) to shape(1,5376,16)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x_three in enumerate(prediction):  # image index, image inference
        # 对每张图片的预测结果依次进行处理
        # x = x[xc[xi]]  # confidence
        # x_head0,x_head1,x_head2=torch.split(x, [4096, 1024, 256], dim=0)
        for x in torch.split(x_three, [4096, 1024, 256], dim=0): #依次对三个检测头的预测结果进行处理
            if labels and len(labels[xi]) and not rotated: #不走
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            #cls[n,12] box[n,4]
            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1) #只留下了confidence大于0.001的:[n,6]
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:# 不走
                x = x[(x[:, 5:6] == classes).any(1)]

            # Check shape，置信度满足的还剩多少个
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes 不走
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores
            if rotated:
                boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
                i = nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c  # boxes (offset by class)
                # 本来要每一类单独进行，但它对每一类的坐标都做了不同偏移，所以可以放到一起做NMS
                # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if x.shape[0]>0:
                boxes=torch.stack(remove_contained_boxes1(x),dim=0)  # NMS
            # i = i[:max_det]  # limit detections,i得到的是索引

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # 对它进行处理，box,conf,cls，box：xyxy

            # xx = torch.stack(remove_contained_boxes1(x),dim=0)
            # output[xi] = xx  #
            # output[xi] = x[i]
            # if output[xi].shape[0]>0:
            #     output[xi] =torch.cat((output[xi],x[i]),0) #三个头检测出来的结果依次拼接
            # else:
            #     output[xi] = x[i]#第xi张图片的输出框[n,6]，到时候在第0维concat
            if output[xi].shape[0]>0:
                output[xi] =torch.cat((output[xi],boxes),0) #三个头检测出来的结果依次拼接
            else:
                output[xi] = boxes#第xi张图片的输出框[n,6]，到时候在第0维concat
        if output[xi].shape[0]==0:
            continue
        output[xi]=remove_contained_boxes0(output[xi])#对三个头的检测结果做一一次NMS操作和
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
        #     break  # time limit exceeded

    return output


# $$$$$$$$$$$$$$$$$$$$$$$$$去掉重合的框$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def remove_contained_boxes(boxes):
    # 转换为numpy数组以便高效处理
    boxes = boxes.cpu().numpy()

    # 获取唯一的类别和对应的索引
    classes, indices = np.unique(boxes[:, 5], return_inverse=True)

    # 按类别和置信度排序
    sorted_indices = np.lexsort((boxes[:, 4], boxes[:, 5]))  # 先按置信度，再按类别排序
    sorted_boxes = boxes[sorted_indices]

    filtered_boxes = []
    for cls in classes:
        # 获取当前类别的所有框
        class_indices = np.where(sorted_boxes[:, 5] == cls)[0]
        class_boxes = sorted_boxes[class_indices]

        # 创建一个数组来标记被包含的框
        contained = np.zeros(class_boxes.shape[0], dtype=bool)

        # 遍历所有框，检查是否包含其他框
        for i in range(class_boxes.shape[0]):
            if contained[i]:
                continue
            current_box = class_boxes[i]
            for j in range(i + 1, class_boxes.shape[0]):
                if is_contained(class_boxes[j], current_box):
                    contained[j] = True
                elif is_contained(current_box, class_boxes[j]):
                    # 如果当前框被后面的框包含，并且置信度更低，则标记当前框为被包含
                    if current_box[4] < class_boxes[j][4]:
                        contained[i] = True
                        break

        # 添加非包含的框到结果列表
        filtered_boxes.extend(class_boxes[~contained])

    # 转换回torch tensor
    filtered_boxes = torch.tensor(filtered_boxes, dtype=torch.float32)
    return filtered_boxes
#保留置信度最大的框
def remove_contained_boxes0(boxes):
    classes, indices = torch.unique(boxes[:, 5], sorted=True, return_inverse=True)
    # Compute the sort indices based on confidence within each class
    confidence_sort_indices = (boxes[:, 4]).argsort(dim=0)  # Note the use of dim=0 here
    sorted_boxes = boxes[confidence_sort_indices]
    cls_sort_indices = (sorted_boxes[:, 5]).argsort(dim=0)
    sorted_boxes = sorted_boxes[cls_sort_indices]  # Apply the sort indices to the original indices

    filtered_boxes = []
    for cls_idx, cls in enumerate(classes):
        # Get the indices of boxes for the current class
        class_indices = (sorted_boxes[:, 5] == cls).nonzero(as_tuple=True)[0]
        class_boxes = sorted_boxes[class_indices]
        # Create a list to mark boxes that are contained
        contained = [False] * class_boxes.size(0)
        # Iterate over the sorted boxes for the current class
        for i in range(class_boxes.size(0)):
            if contained[i]:
                continue  # Skip already contained boxes
            current_box = class_boxes[i]
            current_conf = current_box[4]
            # Check if this box contains any other box in the same class
            for j in range(i + 1, class_boxes.size(0)):
                contained_box = class_boxes[j]
                if is_contained(current_box, contained_box):  # 如果当前框被包含
                    # Mark the contained box as contained
                    if current_conf<contained_box[4]:
                        contained[i] = True
                    else:
                        contained[j] = True
                    # Update the confidence of the current (larger) box
                    # class_boxes[i][4] = max(current_conf, contained_box[4])
        # Add the non-contained boxes to the keep list
        for i, is_contained_flag in enumerate(contained):
            if not is_contained_flag:
                filtered_boxes.append(class_boxes[i])
    return torch.stack(filtered_boxes,dim=0) if len(filtered_boxes)>0 else None

def compute_iou(box1, box2):
    # 转换为[xmin, ymin, xmax, ymax]格式
    # box1 = box1.unsqueeze(0)
    # box2 = box2.unsqueeze(1)

    # xmin1, ymin1, xmax1, ymax1 = box1.chunk(4, dim=1)
    # xmin2, ymin2, xmax2, ymax2 = box2.chunk(4, dim=1)
    xmin1, ymin1, xmax1, ymax1 =box1[0],box1[1],box1[2],box1[3]
    xmin2, ymin2, xmax2, ymax2 = box2[0],box2[1],box2[2],box2[3]
    # 计算交集区域
    inter_xmin = torch.max(xmin1, xmin2)
    inter_ymin = torch.max(ymin1, ymin2)
    inter_xmax = torch.min(xmax1, xmax2)
    inter_ymax = torch.min(ymax1, ymax2)

    inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)

    # 计算每个box的面积
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算IoU
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou.squeeze()  # 移除unsqueeze时增加的维度


import torch.nn.functional as F

#softmax加权
def remove_contained_boxes1(boxes):
    classes, indices = torch.unique(boxes[:, 5], sorted=True, return_inverse=True)
    # Compute the sort indices based on confidence within each class
    confidence_sort_indices = (boxes[:, 4]).argsort(dim=0)  # Note the use of dim=0 here
    sorted_boxes = boxes[confidence_sort_indices]
    cls_sort_indices = (sorted_boxes[:, 5]).argsort(dim=0)
    sorted_boxes = sorted_boxes[cls_sort_indices]  # Apply the sort indices to the original indices

    filtered_boxes = []
    contained_boxs = []
    over_lap_box = {}
    for cls_idx, cls in enumerate(classes):
        # Get the indices of boxes for the current class
        class_indices = (sorted_boxes[:, 5] == cls).nonzero(as_tuple=True)[0]
        class_boxes = sorted_boxes[class_indices]
        # Create a list to mark boxes that are contained
        contained = [False] * class_boxes.size(0)
        # Iterate over the sorted boxes for the current class

        for i in range(class_boxes.size(0)):
            if contained[i]:
                continue  # Skip already contained boxes
            current_box = class_boxes[i]
            # current_conf = current_box[4]
            # Check if this box contains any other box in the same class
            for j in range(i + 1, class_boxes.size(0)):
                contained_box = class_boxes[j]
                if is_contained1(current_box, contained_box):  # 如果当前框被包含
                    # Mark the contained box as contained
                    contained[j] = True
                    contained[i] = True
                    if i+cls_idx*10 in over_lap_box:
                        over_lap_box[i+cls_idx*10].append(contained_box)
                    else:
                        over_lap_box[i+cls_idx*10]=[current_box,contained_box]

        # Add the non-contained boxes to the keep list
        for i, is_contained_flag in enumerate(contained):
            if not is_contained_flag:
                filtered_boxes.append(class_boxes[i])

    for k,v in over_lap_box.items():# v[[xyxycc],[xyxycc]...]
        all_tensors = torch.stack([t for t in v], dim=0)
        # 提取各个分量
        x0 = all_tensors[:, 0]
        y0 = all_tensors[:, 1]
        x1 = all_tensors[:, 2]
        y1 = all_tensors[:, 3]
        conf = all_tensors[:, 4]
        cls = all_tensors[:, 5][0]

        # conf_sum = conf.sum()
        # if conf_sum > 0:  # 防止除以零
        #     conf_normalized = conf / conf_sum
        # else:
        #     conf_normalized = conf
        # 定义温度参数
        temperature = 0.02  # 可以根据需要调整这个值
        # 对conf应用带温度参数的softmax进行归一化
        conf_scaled = conf / temperature
        conf_softmax = F.softmax(conf_scaled, dim=0)
        # 计算新的分量
        x0_new = (x0 * conf_softmax).sum()
        y0_new = (y0 * conf_softmax).sum()
        x1_new = (x1 * conf_softmax).sum()
        y1_new = (y1 * conf_softmax).sum()
        con_new = conf.max()#这里考虑用归一化后的还是之前的最大值

        # 构造新的tensor
        new_tensor = torch.tensor([x0_new.item(), y0_new.item(), x1_new.item(), y1_new.item(), con_new.item(), cls.item()]).to(device=boxes.device)
        contained_boxs.append(new_tensor)

    # 遍历over_lap_box然后对其进行均值投票处理
    return filtered_boxes+ contained_boxs

def is_contained(box1, box2):
    return (box1[0] >= box2[0] and box1[1] >= box2[1] and
            box1[2] <= box2[2] and box1[3] <= box2[3])    or  (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3])

#IOU超过0.1就认为是要合并的
def is_contained1(box1, box2):
    iou_thres=0.5
    return compute_iou(box1, box2) > iou_thres


#ious*ious*conf / temperature
def remove_contained_boxes2(boxes):
    classes, indices = torch.unique(boxes[:, 5], sorted=True, return_inverse=True)
    # Compute the sort indices based on confidence within each class
    confidence_sort_indices = (boxes[:, 4]).argsort(dim=0)  # Note the use of dim=0 here
    sorted_boxes = boxes[confidence_sort_indices]
    cls_sort_indices = (sorted_boxes[:, 5]).argsort(dim=0)
    sorted_boxes = sorted_boxes[cls_sort_indices]  # Apply the sort indices to the original indices

    filtered_boxes = []
    contained_boxs = []
    over_lap_box = {}
    for cls_idx, cls in enumerate(classes):
        # Get the indices of boxes for the current class
        class_indices = (sorted_boxes[:, 5] == cls).nonzero(as_tuple=True)[0]
        class_boxes = sorted_boxes[class_indices]
        # Create a list to mark boxes that are contained
        contained = [False] * class_boxes.size(0)
        # Iterate over the sorted boxes for the current class

        for i in range(class_boxes.size(0)):
            if contained[i] :
                continue  # Skip already contained boxes
            current_box = class_boxes[i]
            # current_conf = current_box[4]
            # Check if this box contains any other box in the same class
            for j in range(i + 1, class_boxes.size(0)):
                contained_box = class_boxes[j]
                if is_contained(current_box, contained_box):  # 如果当前框被包含
                    # Mark the contained box as contained
                    contained[j] = True
                    contained[i] = True
                    if i+cls_idx*10 in over_lap_box:
                        over_lap_box[i+cls_idx*10].append(contained_box)
                    else:
                        over_lap_box[i+cls_idx*10]=[current_box,contained_box]

        # Add the non-contained boxes to the keep list
        for i, is_contained_flag in enumerate(contained):
            if not is_contained_flag:
                filtered_boxes.append(class_boxes[i])

    for k,v in over_lap_box.items():# v[[xyxycc],[xyxycc]...]
        all_tensors = torch.stack([t for t in v], dim=0)
        # 提取各个分量
        x0 = all_tensors[:, 0]
        y0 = all_tensors[:, 1]
        x1 = all_tensors[:, 2]
        y1 = all_tensors[:, 3]
        conf = all_tensors[:, 4]
        cls = all_tensors[:, 5][0]
        # 找出conf最大的元素及其索引
        max_conf, max_conf_idx = conf.max(0)
        max_conf_tensor = all_tensors[max_conf_idx]
        # conf_sum = conf.sum()
        # if conf_sum > 0:  # 防止除以零
        #     conf_normalized = conf / conf_sum
        # else:
        #     conf_normalized = conf
        ious = compute_iou(all_tensors[:, :4],
                           max_conf_tensor[:4].unsqueeze(0))  # all_tensors中的每个框与max_conf_tensor的框计算IoU
        # 定义温度参数
        temperature = 0.02  # 可以根据需要调整这个值
        # 对conf应用带温度参数的softmax进行归一化
        conf_scaled = ious*ious*conf / temperature
        import torch.nn.functional as F
        conf_softmax = F.softmax(conf_scaled, dim=0)
        # 计算新的分量
        x0_new = (x0 * conf_softmax).sum()
        y0_new = (y0 * conf_softmax).sum()
        x1_new = (x1 * conf_softmax).sum()
        y1_new = (y1 * conf_softmax).sum()
        con_new = conf.max()#这里考虑用归一化后的还是之前的最大值

        # 构造新的tensor
        new_tensor = torch.tensor([x0_new.item(), y0_new.item(), x1_new.item(), y1_new.item(), con_new.item(), cls.item()]).to(device=boxes.device)
        contained_boxs.append(new_tensor)

    # 遍历over_lap_box然后对其进行均值投票处理
    return filtered_boxes+ contained_boxs

def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates
    """
    if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        padw (int): Padding width. Defaults to 0
        padh (int): Padding height. Defaults to 0
    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding box coordinates in the xywh format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding boxes coordinates in the xyxy format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def ltwh2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

    Args:
        x (torch.Tensor): the input tensor

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xywh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    returned in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): the input image

    Returns:
        y (np.ndarray | torch.Tensor): the xyxy coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    return y


def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        masks (torch.Tensor): The returned masks with dimensions [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def regularize_rboxes(rboxes):
    """
    Regularize rotated boxes in range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.

    Returns:
        (torch.Tensor): The regularized boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge and angle if h >= w
    w_ = torch.where(w > h, w, h)
    h_ = torch.where(w > h, h, w)
    t = torch.where(w > h, t, t + math.pi / 2) % math.pi
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes


def masks2segments(masks, strategy="largest"):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
        masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
        strategy (str): 'concat' or 'largest'. Defaults to largest

    Returns:
        segments (List): list of segment masks
    """
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of FP32 torch tensors (0.0-1.0) to a NumPy uint8 array (0-255), changing from BCHW to BHWC layout.

    Args:
        batch (torch.Tensor): Input tensor batch of shape (Batch, Channels, Height, Width) and dtype torch.float32.

    Returns:
        (np.ndarray): Output NumPy array batch of shape (Batch, Height, Width, Channels) and dtype uint8.
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def clean_str(s):
    """
    Cleans a string by replacing special characters with underscore _

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
