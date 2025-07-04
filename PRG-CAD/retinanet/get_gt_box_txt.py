import os
from pathlib import Path
import shutil

#因为验证和测试环节都用到了ground-truth文件夹下的txt标签，所以直接在这里生成，到时候只需要访问即可，不需要反复生成
def convert_yolo_labels(yolo_root, output_root, classes_file, img_size=(512, 512)):
    """
    批量转换YOLO标签到VOC格式

    参数：
        yolo_root: YOLO标签根目录（包含病人子文件夹）
        output_root: 输出根目录（自动创建）
        classes_file: 类别文件路径
        img_size: 图像尺寸（宽，高）
    """
    # 读取类别映射
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    # 创建完整的目录结构
    yolo_path = Path(yolo_root)
    output_path = Path(output_root)

    # 清空并重建输出目录
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(yolo_path, output_path, ignore=ignore_non_txt, dirs_exist_ok=True)

    # 处理所有标签文件
    for txt_file in yolo_path.rglob('*.txt'):
        if txt_file.name == 'classes.txt':
            continue

        # 构建输出路径
        relative_path = txt_file.relative_to(yolo_path)
        output_file = output_path / relative_path

        # 转换单个文件
        try:
            with open(txt_file, 'r') as f_in:
                yolo_lines = [l.strip() for l in f_in if l.strip()]

            voc_lines = []
            for line in yolo_lines:
                # 解析并验证YOLO格式
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"无效数据行：{line}")

                class_id, xc, yc, w, h = map(float, parts)
                if not (0 <= class_id < len(classes)):
                    raise ValueError(f"无效类别ID：{int(class_id)}")

                # 坐标转换
                class_name = classes[int(class_id)]
                xmin = int((xc - w / 2) * img_size[0])
                ymin = int((yc - h / 2) * img_size[1])
                xmax = int((xc + w / 2) * img_size[0])
                ymax = int((yc + h / 2) * img_size[1])

                # 边界约束
                xmin = max(0, min(xmin, img_size[0]))
                ymin = max(0, min(ymin, img_size[1]))
                xmax = max(0, min(xmax, img_size[0]))
                ymax = max(0, min(ymax, img_size[1]))

                voc_lines.append(f"{class_name} {xmin} {ymin} {xmax} {ymax}\n")

            # 写入转换结果
            with open(output_file, 'w') as f_out:
                f_out.writelines(voc_lines)

        except Exception as e:
            print(f"转换失败：{txt_file} | 错误：{str(e)}")
            continue


def ignore_non_txt(src, names):
    """复制时忽略非txt文件"""
    return [n for n in names if not (n.endswith('.txt') or Path(src).joinpath(n).is_dir())]


if __name__ == "__main__":
    # 配置参数
    config = {
        "yolo_root": r"/home/HELUXING/Brain_project/data/sequence_dataset_len_cls8_modified/sequence6/labels",          # YOLO标签根目录
        "output_root": r"./ground-truth",  # 输出目录
        "classes_file": r"model_data/voc_classes.txt",  # 类别文件路径
        "img_size": (512, 512)            # 固定图像尺寸
    }

    # 执行转换
    try:
        convert_yolo_labels(
            yolo_root=config["yolo_root"],
            output_root=config["output_root"],
            classes_file=config["classes_file"],
            img_size=config["img_size"]
        )
        print(f"转换完成！结果已保存至 {config['output_root']}")
        print(f"共处理 {len(list(Path(config['yolo_root']).rglob('*.txt')))-1} 个标签文件")  # 排除classes.txt
    except FileNotFoundError as e:
        print(f"错误：关键文件缺失 - {str(e)}")
    except Exception as e:
        print(f"发生未预期错误：{str(e)}")