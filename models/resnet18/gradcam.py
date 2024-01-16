import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import os
from tqdm import tqdm
from torchcam.methods import SmoothGradCAMpp
import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='', help='input image path')
parser.add_argument('--weights_path', type=str, default='', help='input weights path')
parser.add_argument('--save_dir', type=str, default='', help='model weights output save dir')
args = parser.parse_args()

# 数据集路径和预处理
data_dir = args.img_path
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 请根据你的数据集调整均值和标准差
])

# 加载数据集
dataset = ImageFolder(root=data_dir, transform=transform)

# 数据加载器
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 初始化并加载预训练的 ResNet-18 模型
model = resnet18(pretrained=True)
num_classes = len(dataset.classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 加载已经训练好的模型权重
model.load_state_dict(torch.load(args.weights_path))

# 设置模型为评估模式
model.eval()

# 使用 CPU 运行模型
device = torch.device("cpu")
model = model.to(device)

# 计算准确率

correct = 0
total = 0
incorrect_predictions = []

for idx, (images, labels) in enumerate(tqdm(data_loader, desc="Inference Progress")):
    images, labels = images.to(device), labels.to(device)
    img_path = os.path.join(data_loader.dataset.root, data_loader.dataset.samples[idx][0])
    raw = cv2.imread(img_path)
    h, w, _ = raw.shape
    with SmoothGradCAMpp(model) as cam_extractor:
        # Preprocess your data and feed it to the model
        out = model(images)
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Visualize the raw CAM
    max_size = 640
    if w > h:
        new_width = max_size
        new_height = int((max_size / w) * h)
    else:
        new_height = max_size
        new_width = int((max_size / h) * w)

    reshaped_array = cv2.resize(activation_map[0].squeeze(0).numpy(), (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(args.save_dir + data_loader.dataset.samples[idx][0].split('\\')[-1], reshaped_array * 255)


    outputs = model(images)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    # 获取置信度和记录错误的图片名称、置信度
    confidence = probabilities[0][predicted[0]].item()
    if predicted.item() != labels.item():
        img_path = os.path.join(data_loader.dataset.root, data_loader.dataset.samples[idx][0])
        incorrect_predictions.append(
            (img_path, dataset.classes[predicted.item()], dataset.classes[labels.item()], confidence))

    # 显示每张图片的置信度
   # print(f"Image: {data_loader.dataset.samples[idx][0]} | Confidence: {confidence:.4f}")

accuracy = correct / total
print(f'Accuracy of the network on the dataset: {100 * accuracy:.2f}%')

# 打印错误的图片名称和置信度
# print("Incorrect Predictions:")
# for img_path, predicted_class, actual_class, confidence in incorrect_predictions:
#     print(f"Image: {img_path} | Predicted: {predicted_class} | Actual: {actual_class} | Confidence: {confidence:.4f}")
