import torch.nn as nn


class StixelNet(nn.Module):
    def __init__(self):
        super(StixelNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu11 = nn.ELU()
        self.conv12 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu12 = nn.ELU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.drop1 = nn.Dropout(0.4)

        self.conv13 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu13 = nn.ELU()
        self.conv14 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu14 = nn.ELU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 1))

        self.drop2 = nn.Dropout(0.4)

        self.conv15 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu15 = nn.ELU()
        self.conv16 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu16 = nn.ELU()
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.drop3 = nn.Dropout(0.4)

        self.conv17 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu17 = nn.ELU()
        self.conv18 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.elu18 = nn.ELU()
        self.pool7 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.drop4 = nn.Dropout(0.4)

        self.conv19 = nn.Conv2d(256, 2048, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0))
        self.elu19 = nn.ELU()
        self.conv20 = nn.Conv2d(2048, 2048, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1))
        self.elu20 = nn.ELU()
        self.conv21 = nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.elu21 = nn.ELU()

        self.drop5 = nn.Dropout(0.4)

        self.conv22 = nn.Conv2d(2048, 50, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=1)

        self.need_initialization = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)

        x = self.conv11(x)
        x = self.elu11(x)
        x = self.conv12(x)
        x = self.elu12(x)
        x = self.pool4(x)

        x = self.drop1(x)

        x = self.conv13(x)
        x = self.elu13(x)
        x = self.conv14(x)
        x = self.elu14(x)
        x = self.pool5(x)

        x = self.drop2(x)

        x = self.conv15(x)
        x = self.elu15(x)
        x = self.conv16(x)
        x = self.elu16(x)
        x = self.pool6(x)

        x = self.drop3(x)

        x = self.conv17(x)
        x = self.elu17(x)
        x = self.conv18(x)
        x = self.elu18(x)
        x = self.pool7(x)

        x = self.drop4(x)

        x = self.conv19(x)
        x = self.elu19(x)
        x = self.conv20(x)
        x = self.elu20(x)
        x = self.conv21(x)
        x = self.elu21(x)

        x = self.drop5(x)

        x = self.conv22(x)
        x = self.softmax(x)

        return x.view((100, 50))
