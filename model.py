import torch
from torch import nn

class DoubleConvBlock(nn.Module): 
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        
        kernelSize = 3
        stride = 1
        padding = 1

        self.doubleConv = nn.Sequential(
            nn.Conv2d(inputDepth, outputDepth, kernelSize, stride, padding, bias=False),
            nn.BatchNorm2d(outputDepth),
            nn.ReLU(inplace = True),
            nn.Conv2d(outputDepth, outputDepth, kernelSize, stride, padding, bias=False),
            nn.BatchNorm2d(outputDepth),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, xIn):
        return self.doubleConv(xIn)
    
class DownBlock(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(inputDepth, outputDepth)
        )

    def forward(self, xIn):
        return self.down(xIn)

class UpBlock(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), #doubles width and height of tensor
            nn.Conv2d(inputDepth, inputDepth // 2, 1) #need to half depth
        )

        self.conv = DoubleConvBlock(inputDepth, outputDepth)

    def forward(self, xIn, xBridged):
        xIn = self.up(xIn)
        return self.conv(torch.concatenate([xIn, xBridged], dim=1))

class UNet(nn.Module):
    def __init__(self, layerDepths): #layerdepths is the depth/channels in each layer tensor
        super().__init__()

        outputDepth = layerDepths[-1]
        layerDepths = layerDepths[:-1]

        self.inConv = DoubleConvBlock(layerDepths[0], layerDepths[1])

        self.downBlocks = nn.ModuleList()
        for i in range(len(layerDepths[:-2])):
            self.downBlocks.append(DownBlock(layerDepths[i+1], layerDepths[i+2]))

        self.upBlocks = nn.ModuleList()
        upDepths = layerDepths[1:][::-1]
        for i in range(len(upDepths[:-1])):
            self.upBlocks.append(UpBlock(upDepths[i], upDepths[i+1]))

        self.outConv = nn.Conv2d(upDepths[-1], outputDepth, kernel_size=1)
    
    def forward(self, x):
        bridgedTensors = []

        x = self.inConv(x)
        bridgedTensors.append(x)

        for downBlock in self.downBlocks:
            x = downBlock(x)
            bridgedTensors.append(x)

        bridgedTensors.pop()

        for upBlock in self.upBlocks:
            xBridged = bridgedTensors.pop()
            x = upBlock(x, xBridged)

        return self.outConv(x)