import os

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import shutil
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import preprocess_input
from .utils_bbox import  decodebox, non_max_suppression
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$confidence=0.05改成了0.01，max_boxes改100为5
    def __init__(self, net, input_shape, class_names, num_classes, val_img_path, log_dir, cuda, \
                 map_out_path=".temp_map_out", max_boxes=5, confidence=0.01, nms_iou=0.5, letterbox_image=True,
                 MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_img_path = val_img_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
        # 基础转换：确保所有图像转换为Tensor
        from torchvision import transforms as T
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.base_transform = T.Compose([  # 这里是否加归一化
            # T.Lambda(lambda x: x.convert("L")),  # 转换为灰度图
            T.Resize((512, 512)),
            T.ToTensor(),

        ])

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def get_map_txt(self, sub_sequence_dir, class_names, map_out_path):
        img_dir = os.path.join(self.val_img_path, sub_sequence_dir)
        images = []
        # 加载图像序列（确保按切片顺序加载）
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        for f in img_files:
            img_path = os.path.join(img_dir, f)
            img = Image.open(img_path)
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$这里归一化的话训练时也要归一化
            images.append(self.base_transform(img))
        # 转换为numpy数组用于增强torch.stack(images).numpy()
        images = torch.stack(images).float()
        # f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        image_shape = np.array([512, 512])
        # #---------------------------------------------------------#
        #
        # #---------------------------------------------------------#
        # #   添加上batch_size维度，图片预处理，归一化。
        # #---------------------------------------------------------#
        # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   传入网络当中进行预测
            # ---------------------------------------------------------#
            _, regressions, classifications, anchors = self.net(images)
            # 要将结果拆解开来，一张图片一张图片的来
            predict_save_path=os.path.join(map_out_path, "detection-results" , sub_sequence_dir)
            os.makedirs(predict_save_path,exist_ok=True)
            for i in range(regressions.shape[0]):
                regression=regressions[i:i+1,:,:]
                classification=classifications[i:i+1,:,:]
                f = open(os.path.join(predict_save_path,img_files[i].replace(".jpg", ".txt")), "w")
                # -----------------------------------------------------------#
                #   将预测结果进行解码
                # -----------------------------------------------------------#
                outputs = decodebox(regression, anchors, self.input_shape)
                results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)

                if results[0] is None:
                    f.close()
                    continue

                top_label = np.array(results[0][:, 5], dtype='int32')
                top_conf = results[0][:, 4]
                top_boxes = results[0][:, :4]

                top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
                top_boxes = top_boxes[top_100]
                top_conf = top_conf[top_100]
                top_label = top_label[top_100]

                for i, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]
                    box = top_boxes[i]
                    score = str(top_conf[i])

                    top, left, bottom, right = box
                    if predicted_class not in class_names:
                        continue

                    f.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

                f.close()
        return

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            # $$$$$$$$$$$$$$$$$$$这里改成直接读取,推理结果也要保存到序列子文件夹中去
            # 遍历每个images/val_6下的子文件夹，整个子文件夹的图片送入模型，得到的预测txt放到detection-results
            for sub_sequence_dir in tqdm(os.listdir(self.val_img_path)):
                self.get_map_txt(sub_sequence_dir, self.class_names, self.map_out_path)

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)


if __name__ == '__main__':
    sub_sequence_dir = "RT-233627_1.2.840.113619.2.416.275068482504067448337293982574672080188_1"
    img_dir = os.path.join(r"C:\Users\xize\Desktop\1111\yolo_format\images\val_6", sub_sequence_dir)
    images = []
    # 加载图像序列（确保按切片顺序加载）
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    from torchvision import transforms as T

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    base_transform = T.Compose([  # 这里是否加归一化
        # T.Lambda(lambda x: x.convert("L")),  # 转换为灰度图
        T.Resize((512, 512)),
        T.ToTensor(),

    ])
    for f in img_files:
        img_path = os.path.join(sub_sequence_dir, f)
        img = Image.open(img_path)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$这里归一化的话训练时也要归一化
        images.append(preprocess_input(base_transform(img)))
    # 转换为numpy数组用于增强
    images = torch.stack(images).numpy().transpose(0, 2, 3, 1)  # [D, H, W, C]

    # f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
    image_shape = np.array([512, 512])
