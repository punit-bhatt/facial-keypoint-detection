import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Layer Input - (1, 224, 224)
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.__init_params(self.conv1)

        # Layer Input - (32, 112, 112)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.__init_params(self.conv2)

        # Layer Input - (64, 56, 56)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.__init_params(self.conv3)

        # Layer Input - (128, 28, 28)
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.__init_params(self.conv4)

        # Layer Input - (256, 14, 14)
        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.__init_params(self.conv5)

        # Layer Input - (512, 7, 7)
        self.fc1 = nn.Linear(7 * 7 * 512, 2048)
        self.drop_fc1 = nn.Dropout(0.2)
        self.__init_params(self.fc1)

        # Layer Input - (2048)
        self.fc2 = nn.Linear(2048, 512)
        self.drop_fc2 = nn.Dropout(0.2)
        self.__init_params(self.fc2)

        # Layer Input - (512)
        self.fc3 = nn.Linear(512, 68 * 2)
        self.__init_params(self.fc3)


    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        x = x.view(x.shape[0], -1)

        x = self.drop_fc1(F.relu(self.fc1(x)))
        x = self.drop_fc2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def __init_params(self, layer):

        # A better option here would be to use He/Kaiming init.
        I.xavier_uniform_(layer.weight)
