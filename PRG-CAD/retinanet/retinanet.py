import colorsys
import os
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.retinanet import retinanet
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import decodebox, non_max_suppression


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和phi都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class Retinanet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": '/logs/xxx.pth',
        "classes_path": 'model_data/voc_classes.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小
        # ---------------------------------------------------------------------#
        "input_shape": [512, 512],
        # ---------------------------------------------------------------------#
        #   用于选择所使用的模型的版本
        #   0、1、2、3、4
        #   resnet18, resnet34, resnet50, resnet101, resnet152
        # ---------------------------------------------------------------------#
        "phi": 0,
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.01,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # ---------------------------------------------------------------------#
        #
        # ---------------------------------------------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Retinanet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # self.num_classes=11
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # ----------------------------------------#
        #   创建Retinanet模型
        # ----------------------------------------#
        num_classes0=11
        self.net = retinanet(num_classes0, self.phi)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   传入网络当中进行预测
            # ---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)

            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decodebox(regression, anchors, self.input_shape)
            results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   传入网络当中进行预测
            # ---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)

            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decodebox(regression, anchors, self.input_shape)
            results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   传入网络当中进行预测
                # ---------------------------------------------------------#
                _, regression, classification, anchors = self.net(images)

                # -----------------------------------------------------------#
                #   将预测结果进行解码
                # -----------------------------------------------------------#
                outputs = decodebox(regression, anchors, self.input_shape)
                results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    from torchvision import transforms as T
    base_transform = T.Compose([  # 这里是否加归一化
        # T.Lambda(lambda x: x.convert("L")),  # 转换为灰度图
        T.Resize((512, 512)),
        T.ToTensor(),

    ])

    # def get_map_txt(self, image_id, image, class_names, map_out_path):
    def get_map_txt(self, sub_sequence_dir, class_names, map_out_path):
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        test_img_path = "/home/HELUXING/Brain_project/data/sequence_dataset_len_cls8_modified/sequence6/images/test"
        img_dir = os.path.join(test_img_path, sub_sequence_dir)
        images = []
        # 加载图像序列（确保按切片顺序加载）
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        # if sub_sequence_dir == "0000149665_1.2.840.113619.2.404.3.658226.369.1521187961.254":
        #     print(img_files)
        for f in img_files:
            img_path = os.path.join(img_dir, f)
            img = Image.open(img_path)
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$这里归一化的话训练时也要归一化
            images.append(self.base_transform(img))
        # 转换为numpy数组用于增强torch.stack(images).numpy()
        images = torch.stack(images).float()
        # f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        image_shape = np.array([512, 512])
        predict_save_path = os.path.join(map_out_path, "detection-results", sub_sequence_dir)
        os.makedirs(predict_save_path, exist_ok=True)
        with torch.no_grad():
            # images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   传入网络当中进行预测
            # ---------------------------------------------------------#
            _, regressions, classifications, anchors = self.net(images)
            # if sub_sequence_dir == "0000149665_1.2.840.113619.2.404.3.658226.369.1521187961.254":
            #     print(regressions.shape[0],"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            for i in range(regressions.shape[0]):
                regression = regressions[i:i + 1, :, :]
                classification = classifications[i:i + 1, :, :]
                # -----------------------------------------------------------#
                #   将预测结果进行解码
                # -----------------------------------------------------------#
                outputs = decodebox(regression, anchors[0], self.input_shape)#不知怎么回事anchors会有两个元素，取第一个
                results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)
                f = open(os.path.join(predict_save_path, img_files[i].replace(".jpg", ".txt")), "w")
                if results[0] is None:
                    f.close()
                    continue

                top_label = np.array(results[0][:, 5], dtype='int32')
                top_conf = results[0][:, 4]
                top_boxes = results[0][:, :4]

                for i, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]
                    box = top_boxes[i]
                    score = str(top_conf[i])

                    top, left, bottom, right = box
                    if predicted_class not in class_names:
                        continue

                    f.write("%s %s %s %s %s %s\n" % (
                        predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),
                        str(int(bottom))))

                f.close()
        return
