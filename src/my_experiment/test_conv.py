import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

# Download training data from open datasets.
training_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=0)

from torch.utils.tensorboard import SummaryWriter


# Default log_dir argument is "runs" - but it's good to be specific
# torch.utils.tensorboard.SummaryWriter is imported above
writer = SummaryWriter()

class My_nn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

mynn = My_nn()
print(mynn)

step=0
for data in test_loader:
    imgs, targets = data
    output = mynn(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images('input', imgs, step)
    writer.add_images('output', output, step)
    step=step+1
    

writer.close()