import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4*4*16,out_features=120)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        conv1_output=self.conv1(input)#[28,28,1]--->[24,24,6]--->[12,12,6]
        conv2_output = self.conv2(conv1_output)  # [12,12,6]--->[8,8,,16]--->[4,4,16]
        conv2_output=conv2_output.view(-1,4*4*16)#将[n,4,4,16]维度转化为[n,4*4*16]
        fc1_output=self.fc1(conv2_output)#[n,256]--->[n,120]
        fc2_output=self.fc2(fc1_output)#[n,120]-->[n,84]
        fc3_output = self.fc3(fc2_output)  # [n,84]-->[n,10]
        return fc3_output
