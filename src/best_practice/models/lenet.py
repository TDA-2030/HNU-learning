# coding:utf8
from torch import nn
import torch.nn.functional as F
from .basic_module import BasicModule

# sys.path.append("../")
# from src.logs import my_logger


# def DatasetGet(batch_size: int,
#                num_workers: int = 1) -> Tuple[DataLoader, DataLoader]:
#     # 定义对数据的预处理
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # 转为Tensor
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
#     ])

#     # 训练集
#     trainset = tv.datasets.CIFAR10(root='/home/zhouli/work/HNU-learning/my_experiment/data',
#                                    train=True,
#                                    download=True,
#                                    transform=transform)

#     trainloader = DataLoader(trainset,
#                              batch_size=batch_size,
#                              shuffle=True,
#                              num_workers=num_workers)

#     # 测试集
#     testset = tv.datasets.CIFAR10('/home/zhouli/work/HNU-learning/my_experiment/data',
#                                   train=False,
#                                   download=False,
#                                   transform=transform)

#     testloader = DataLoader(testset,
#                             batch_size=batch_size,
#                             shuffle=False,
#                             num_workers=num_workers)
#     return (trainloader, testloader)


class LeNet(BasicModule):

    def __init__(self):
        super(LeNet, self).__init__()
        self.model_name = 'LeNet'
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# def train():
#     # Get cpu or gpu device for training.

#     criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
#     optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)
#     lenet.to(device)

#     # torch.set_num_threads(8)
#     for epoch in range(64):

#         my_logger.info(f"=====epoch={epoch}======")
#         running_loss = 0.0
#         lenet.train()
#         for i, data in enumerate(trainloader, 0):

#             # 输入数据
#             inputs, labels = data
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # 梯度清零
#             optimizer.zero_grad()

#             # forward + backward
#             outputs = lenet(inputs)
#             loss: torch.Tensor = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()  # 更新参数

#             # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
#             running_loss += loss.item()
#             if i % 200 == 0:
#                 my_logger.info(loss.item())
#                 writer.add_scalar("loss value", loss.item(), i)
#                 my_logger.info('[%d/%5d] loss: %.3f' %
#                                (epoch + 1, i + 1, running_loss / 200))
#                 running_loss = 0.0
#             # if i % 12000 == 19999:
#             #     for param_group in optimizer.param_groups:
#             #         param_group['lr'] *= 0.1
#     my_logger.info('Finished Training')



# def test():
#     correct = 0  # 预测正确的图片数
#     total = 0  # 总共的图片数

#     lenet.eval()
#     # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = lenet(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum()

#     print(f'10000张测试集中的准确率为: {100 * correct / total}')


# if __name__ == "__main__":
#     train()
#     test()