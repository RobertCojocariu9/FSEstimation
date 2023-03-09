import torch
import torch.nn as nn
import torchvision
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights


def get_scheduler(optimizer, opt):
    if opt.lr_scheduler == 'lstep':
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lr_lambda=lambda epoch: opt.lr_gamma ** ((epoch + 1) // opt.lr_decay_epoch))
    elif opt.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    elif opt.lr_scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        return NotImplementedError('Learning rate scheduler [%s] is not implemented', opt.lr_scheduler)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'pretrained':
                pass
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
                init.constant_(m.bias.data, 0.0)
        elif class_name.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('Initializing network with %s' % init_type)
    net.apply(init_func)


def init_net(num_labels, use_sne=True, init_type='xavier', init_gain=0.02, gpu_ids=None):
    net = FSNet(num_labels, use_sne)
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    for root_child in net.children():
        for children in root_child.children():
            if children in root_child.need_initialization:
                init_weights(children, init_type, gain=init_gain)
            else:
                init_weights(children, "pretrained", gain=init_gain)
    return net


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpsampleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        output = self.activation(x)
        return output


class FSNet(nn.Module):
    def __init__(self, num_labels, use_sn):
        super(FSNet, self).__init__()

        self.num_resnet_layers = 18
        resnet_raw_model1 = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet_raw_model2 = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        filters = [64, 64, 128, 256, 512]

        if use_sn:
            self.another_conv1 = resnet_raw_model1.conv1
        else:
            self.another_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.another_conv1.weight.data = torch.unsqueeze(
                torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)

        self.another_bn1 = resnet_raw_model1.bn1
        self.another_relu = resnet_raw_model1.relu
        self.another_maxpool = resnet_raw_model1.maxpool
        self.another_layer1 = resnet_raw_model1.layer1
        self.another_layer2 = resnet_raw_model1.layer2
        self.another_layer3 = resnet_raw_model1.layer3
        self.another_layer4 = resnet_raw_model1.layer4

        self.rgb_conv1 = resnet_raw_model2.conv1
        self.rgb_bn1 = resnet_raw_model2.bn1
        self.rgb_relu = resnet_raw_model2.relu
        self.rgb_maxpool = resnet_raw_model2.maxpool
        self.rgb_layer1 = resnet_raw_model2.layer1
        self.rgb_layer2 = resnet_raw_model2.layer2
        self.rgb_layer3 = resnet_raw_model2.layer3
        self.rgb_layer4 = resnet_raw_model2.layer4

        # self.conv1_1 = DoubleConvBlock(filters[0] * 2, filters[0], filters[0])
        # self.conv2_1 = DoubleConvBlock(filters[1] * 2, filters[1], filters[1])
        # self.conv3_1 = DoubleConvBlock(filters[2] * 2, filters[2], filters[2])
        self.conv4_1 = DoubleConvBlock(filters[3] * 2, filters[3], filters[3])

        # self.conv1_2 = DoubleConvBlock(filters[0] * 3, filters[0], filters[0])
        # self.conv2_2 = DoubleConvBlock(filters[1] * 3, filters[1], filters[1])
        # self.conv3_2 = ConvBlock(filters[2] * 3, filters[2], filters[2])
        self.conv3_2 = DoubleConvBlock(filters[2] * 2, filters[2], filters[2])

        # self.conv1_3 = DoubleConvBlock(filters[0] * 4, filters[0], filters[0])
        # self.conv2_3 = ConvBlock(filters[1] * 4, filters[1], filters[1])
        self.conv2_3 = DoubleConvBlock(filters[1] * 2, filters[1], filters[1])

        # self.conv1_4 = ConvBlock(filters[0] * 5, filters[0], filters[0])
        self.conv1_4 = DoubleConvBlock(filters[0] * 2, filters[0], filters[0])

        # self.up2_0 = UpsampleBlock(filters[1], filters[0])
        # self.up2_1 = UpsampleBlock(filters[1], filters[0])
        # self.up2_2 = UpsampleBlock(filters[1], filters[0])
        self.up2_3 = UpsampleBlock(filters[1], filters[0])

        # self.up3_0 = UpsampleBlock(filters[2], filters[1])
        # self.up3_1 = UpsampleBlock(filters[2], filters[1])
        self.up3_2 = UpsampleBlock(filters[2], filters[1])

        # self.up4_0 = UpsampleBlock(filters[3], filters[2])
        self.up4_1 = UpsampleBlock(filters[3], filters[2])

        self.up5_0 = UpsampleBlock(filters[4], filters[3])

        self.final = UpsampleBlock(filters[0], num_labels)

        # self.need_initialization = [self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv1_2,
        #                             self.conv2_2, self.conv3_2, self.conv1_3, self.conv2_3, self.conv1_4,
        #                             self.up2_0, self.up2_1, self.up2_2, self.up2_3, self.up3_0, self.up3_1,
        #                             self.up3_2, self.up4_0, self.up4_1, self.up5_0, self.final]
        self.need_initialization = [self.conv4_1, self.conv3_2, self.conv2_3, self.conv1_4, self.up4_1, self.up5_0,
                                    self.up2_3, self.up3_2, self.final]

    def forward(self, rgb, another):
        rgb = self.rgb_conv1(rgb)
        rgb = self.rgb_bn1(rgb)
        rgb = self.rgb_relu(rgb)
        another = self.another_conv1(another)
        another = self.another_bn1(another)
        another = self.another_relu(another)
        rgb = rgb + another
        x1_0 = rgb

        rgb = self.rgb_maxpool(rgb)
        another = self.another_maxpool(another)
        rgb = self.rgb_layer1(rgb)
        another = self.another_layer1(another)
        rgb = rgb + another
        x2_0 = rgb

        rgb = self.rgb_layer2(rgb)
        another = self.another_layer2(another)
        rgb = rgb + another
        x3_0 = rgb

        rgb = self.rgb_layer3(rgb)
        another = self.another_layer3(another)
        rgb = rgb + another
        x4_0 = rgb

        rgb = self.rgb_layer4(rgb)
        another = self.another_layer4(another)
        x5_0 = rgb + another

        # x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        # x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up5_0(x5_0)], dim=1))

        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        # x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up4_1(x4_1)], dim=1))
        x3_2 = self.conv3_2(torch.cat([x3_0, self.up4_1(x4_1)], dim=1))

        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        # x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up3_2(x3_2)], dim=1))
        x2_3 = self.conv2_3(torch.cat([x2_0, self.up3_2(x3_2)], dim=1))

        # x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up2_3(x2_3)], dim=1))
        x1_4 = self.conv1_4(torch.cat([x1_0, self.up2_3(x2_3)], dim=1))

        out = self.final(x1_4)
        return out
