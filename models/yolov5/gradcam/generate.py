import os
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM, YOLOV5GradCAMPP
from models.yolov5_object_detector import YOLOV5TorchObjectDetector
import cv2


names = ['Cancer', 'Granuloma', 'Normal', 'Edema', 'Cyst', 'Leukopkia', 'Nodules', 'Polyps']  # class names
target_layers = ['model_24_m_0', 'model_24_m_1', 'model_24_m_2']

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="", help='Path to the model')
parser.add_argument('--img-path', type=str, default='', help='input image path')
parser.add_argument('--output-dir', type=str, default='', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default='model_17_cv3_act',
                    help='The layer hierarchical address to which gradcam will applied the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam method')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--no_text_box', action='store_true',
                    help='do not show label and box on the heatmap')
args = parser.parse_args()


# 检测单个图片
def main(img_path):
    img = cv2.imread(img_path)  # 读取图像格式：BGR
    torch_img = model.preprocessing(img[..., ::-1])
    # 遍历三层检测层
    for target_layer in target_layers:
        # 获取grad-cam方法
        if args.method == 'gradcam':
            saliency_method = YOLOV5GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif args.method == 'gradcampp':
            saliency_method = YOLOV5GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  # 得到预测结果
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        # 保存设置
        imgae_name = os.path.basename(img_path)  # 获取图片名
        save_path = f'{args.output_dir}{imgae_name[:-4]}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 遍历每张图片中的每个目标
        for i, mask in enumerate(masks):
            output_path = f'{save_path}/{target_layer[6:]}_{i}.jpg'
            mask_out = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
                np.uint8)
            cv2.imwrite(output_path, mask_out)


def is_folder_present(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    return os.path.isdir(folder_path)


if __name__ == '__main__':
    # 图片路径为文件夹
    device = args.device
    input_size = (args.img_size, args.img_size)
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size, names=names)
    tag = 0
    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        print(img_list)
        for item in img_list:
            if is_folder_present(args.output_dir, item.split('.')[0]):
                pass
            else:
                # 依次获取文件夹中的图片名，组合成图片的路径
                tag = tag + 1
                main(os.path.join(args.img_path, item))
                if tag == 300:
                    print(item, 'finish it')
                    break
    # 单个图片
    else:
        main(args.img_path)
