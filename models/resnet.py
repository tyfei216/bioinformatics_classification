"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class ResNet(nn.Module):

    def __init__(self, block, out_channels, num_block, stride, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        
        all_layers = []
        for i in range(len(out_channels)):
            temp = self._make_layer(block,out_channels[i],num_block[i],stride[i])
            # print(temp)
            all_layers = all_layers + temp
        
        self.convs = nn.Sequential(*all_layers)
        
        

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # print("nn Linear in  ", 512 * block.expansion, " nn Linear out", num_classes)
        # print("numofclasses", num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return layers

    def forward(self, x):
        output = self.conv1(x)
        output = self.convs(output)
        #output = self.conv2_x(output)
        #output = self.conv3_x(output)
        #output = self.conv4_x(output)
        #output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        # print("the output shape ", output.shape)
        output = self.fc(output)

        return output 

def resnet18(args):
    """ return a ResNet 18 object
    """
    if args.imgs == 32:
        return ResNet(BasicBlock,[64,128,256,512], [2, 2, 2, 2],[1,2,2,2],args.nc)
    if args.imgs == 64:
        return ResNet(BasicBlock,[64,64,128,256,512], [2, 2, 2, 2, 2],[1,2,2,2,2],args.nc)
    if args.imgs == 128:
        return ResNet(BasicBlock,[64,64,128,128,256,512], [2, 2, 2, 2, 2, 2],[1,2,2,2,2,2],args.nc)

def resnet34(args):
    """ return a ResNet 34 object
    """
    if args.imgs == 32:
        return ResNet(BasicBlock,[64,128,256,512], [3, 4, 6, 3],[1,2,2,2],args.nc)
    if args.imgs == 64:
        return ResNet(BasicBlock,[64,64,128,256,512], [3, 4, 4, 6, 3],[1,2,2,2,2],args.nc)
    if args.imgs == 128:
        return ResNet(BasicBlock,[64,64,128,128,256,512], [3, 4, 4, 5, 6, 3],[1,2,2,2,2,2],args.nc)

def resnet50(args):
    """ return a ResNet 50 object
    """
    if args.imgs == 32:
        return ResNet(BottleNeck,[64,128,256,512], [3, 4, 6, 3],[1,2,2,2],args.nc)
    if args.imgs == 64:
        return ResNet(BottleNeck,[64,64,128,256,512], [3, 4, 4, 6, 3],[1,2,2,2,2],args.nc)
    if args.imgs == 128:
        return ResNet(BottleNeck,[64,64,128,128,256,512], [3, 4, 4, 5, 6, 3],[1,2,2,2,2],args.nc)

def resnet101(args):
    """ return a ResNet 101 object
    """
    if args.imgs == 32:
        return ResNet(BottleNeck,[64,128,256,512], [3, 4, 23, 3],[1,2,2,2],args.nc)
    if args.imgs == 64:
        return ResNet(BottleNeck,[64,64,128,256,512], [3, 4, 7, 23, 3],[1,2,2,2,2],args.nc)
    if args.imgs == 128:
        return ResNet(BottleNeck,[64,64,128,128,256,512], [3, 4, 7,15, 23, 3],[1,2,2,2,2,2],args.nc)

def resnet152(args):
    """ return a ResNet 152 object
    """
    if args.imgs == 32:
        return ResNet(BottleNeck,[64,128,256,512], [3, 8, 36, 3],[1,2,2,2],args.nc)
    if args.imgs == 64:
        return ResNet(BottleNeck,[64,64,128,256,512], [3, 6, 8, 36, 3],[1,2,2,2,2],args.nc)
    if args.imgs == 128:
        return ResNet(BottleNeck,[64,64,128,128,256,512], [3, 6, 8, 22, 36, 3],[1,2,2,2,2,2],args.nc)

if __name__=='__main__':
    a = ResNet(BottleNeck,[64,128,256,512], [3, 4, 23, 3],[1,2,2,2],2)
    b = torch.randn((3,1,32,32))
    print(b)
    c = a(b)



