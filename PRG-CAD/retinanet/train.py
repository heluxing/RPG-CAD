# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import datetime
import os
import warnings
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

# from my_dataset import CTSequenceDataset, my_collate_fn
from my_dataset.myDataset import MyDataset
from nets.retinanet import retinanet
from nets.retinanet_training import (FocalLoss, get_lr_scheduler,
                                     set_optimizer_lr)
from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import retinanet_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    Cuda = True
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    # distributed = False
    distributed = True
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'

    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = ''
    # ------------------------------------------------------#
    #   input_shape     输入的shape大小
    # ------------------------------------------------------#
    input_shape = [512, 512]
    # ------------------------------------------------------#
    #   0、1、2、3、4分别是
    #   resnet18, resnet34, resnet50, resnet101, resnet152
    # ------------------------------------------------------#
    phi = 0
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 200，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 200，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从主干网络的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 200，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 200，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    #             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。

    # ----------------------------------------------------------------------------------------------------------------------------#

    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 8
    # ------------------------------------------------------------------#
    #     #   解冻阶段训练参数
    #     #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #     #   占用的显存较大，网络所有的参数都会发生改变
    #     #   UnFreeze_Epoch          模型总共训练的epoch
    #                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
    #                           Adam可以使用相对较小的UnFreeze_Epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 200
    Unfreeze_batch_size = 32
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    Freeze_Train = False

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'step'
    save_period = 1
    save_dir = 'logs1'
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period =1
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = 8

    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    # ------------------------------------------------------#
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    seed_everything(seed)
    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ----------------------------------------------------#
    #   下载预训练权重
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(str(phi))
            dist.barrier()
        else:
            download_weights(str(phi))

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes=11
    # ------------------------------------------------------#
    #   创建retinanet模型
    #   训练前一定要修改classes_path和对应的txt文件
    # ------------------------------------------------------#
    model = retinanet(num_classes, phi, pretrained, fp16)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   获得损失函数
    # ----------------------#
    focal_loss = FocalLoss()
    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # with open(train_annotation_path) as f:
    #     train_lines = f.readlines()
    # with open(val_annotation_path
    # ) as f:
    #     val_lines = f.readlines()

    num_train = 26193
    num_val = 3539
    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数 
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4


    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone_net.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # train_dataset   = RetinanetDataset(train_lines, input_shape, num_classes, train = True)
        # val_dataset     = RetinanetDataset(val_lines, input_shape, num_classes, train = False)
        root_dir = r"sequence_dataset—path"
        val_img_path = root_dir+"/images/val_6"
        # map_out_path=r"C:\Users\xize\Desktop\1111\voc_format\ground-truth\val_6"#提前放好gt的位置
        # train_transform = None
        # train_dataset = CTSequenceDataset(root_dir=root_dir, transform=train_transform, img_size=512, mode="train_6")
        # val_dataset = CTSequenceDataset(root_dir=root_dir,  img_size=512, mode="val_6")


        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$自己定义的dataset############################
        data = {
            'train': 'sequence_dataset—path/images/train_6',
            'val': 'sequence_dataset—path/images/val_6',
            'test': 'sequence_dataset—path/images/test',
            'names': {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4',
                      5: 'class_5', 6: 'class_6', 7: 'class_7'},
            'yaml_file': 'sequence_dataset—path/sequence_dataset_8cls.yaml',
            'nc': 8}

        batch_size = Unfreeze_batch_size
        from types import SimpleNamespace

        # 数据增强超参数
        hyp = SimpleNamespace(
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=1.0
        )
        # 创建数据集
        train_dataset = MyDataset(
            img_path=data['train'],
            imgsz=512,
            batch_size=batch_size,
            augment=True,  # augmentation
            hyp=hyp,  # TODO: probably add a get_hyps_from_cfg function
            data=data
        )

        # 创建数据集
        val_dataset = MyDataset(
            img_path=data['val'],
            imgsz=512,
            batch_size=batch_size,
            augment=False,  # augmentation
            hyp=hyp,  # TODO: probably add a get_hyps_from_cfg function
            data=data
        )
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        # 创建数据加载器
        gen = DataLoader(
            train_dataset,
            batch_size=batch_size, sampler=train_sampler,
            # shuffle=True,
            collate_fn=getattr(train_dataset, "collate_fn_retinanet", None),
            num_workers=4
        )
        gen_val = DataLoader(
            val_dataset,
            batch_size=batch_size,sampler=val_sampler,
            # shuffle=True,
            collate_fn=getattr(val_dataset, "collate_fn_retinanet", None),
            num_workers=8
        )
        # gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
        #                  pin_memory=True,
        #                  drop_last=True, collate_fn=my_collate_fn, sampler=train_sampler,
        #                  worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        # gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
        #                      pin_memory=True,
        #                      drop_last=True, collate_fn=my_collate_fn, sampler=val_sampler,
        #                      worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_img_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone_net.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size



                if distributed:
                    batch_size = batch_size // ngpus_per_node

                # gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                #                  pin_memory=True,
                #                  drop_last=True, collate_fn=my_collate_fn, sampler=train_sampler,
                #                  worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                # gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                #                      pin_memory=True,
                #                      drop_last=True, collate_fn=my_collate_fn, sampler=val_sampler,
                #                      worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, focal_loss, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period,
                          save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
