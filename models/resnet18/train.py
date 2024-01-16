import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='', help='input image path')
parser.add_argument('--save_dir', type=str, default='', help='model weights output save dir')
args = parser.parse_args()

# 自定义数据集的数据增强与预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 这里的均值和标准差需要根据你的数据集进行调整
])

# 假设你有训练集和验证集
train_dataset = ImageFolder(root=args.img_path, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 初始化 ResNet-18 模型
model = resnet18(pretrained=True)

# 替换最后的全连接层以适应自定义数据集的类别数量
num_classes = len(train_dataset.classes)  # 假设你的数据集类别数
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 50  # 假设训练 10 个 epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader)}")
    accuracy = correct / total
    print(f'Accuracy of the network on the dataset: {100 * accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), args.save_dir + '\custom_resnet18.pth')
