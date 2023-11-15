import torch
from torch import nn

class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        inter_channel = 4 * growth_rate

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottleneck(x)], 1)
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

class DenseNet(nn.Module):
    
    def __init__(self, model_key, block, reduction=0.5,  num_class = 1000):
        super().__init__()
        self.model_info = self.__model__(model_key)
        self.growth_rate = self.model_info[1]
        self.nblocks = self.model_info[0]

        inter_channels = 2 * self.growth_rate

        self.conv1 = nn.Conv2d(3, inter_channels, kernel_size=7, stride=2, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.model_blcok = nn.Sequential()

        for idx in range(len(self.nblocks)-1):
            self.model_blcok.add_module(f"dense_block_{idx}", self._make_layers(block, inter_channels, self.nblocks[idx]))
            inter_channels += self.growth_rate * self.nblocks[idx]
            out_channels = int(reduction * inter_channels)
            self.model_blcok.add_module(f"transition_layer{idx}", Transition(inter_channels, out_channels))
            inter_channels = out_channels

        self.model_blcok.add_module(f"dense_block{len(self.nblocks) - 1}", self._make_layers(block, inter_channels, self.nblocks[-1]))
        inter_channels += self.growth_rate * self.nblocks[-1]
        self.model_blcok.add_module('bn', nn.BatchNorm2d(inter_channels))
        self.model_blcok.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(inter_channels, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.model_blcok(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def _make_layers(self, block, in_chaanels, nblocks):
        dense_block = nn.Sequential()
        for idx in range(nblocks):
            dense_block.add_module(f'botte_neck{idx}', block(in_chaanels, self.growth_rate))
            in_chaanels += self.growth_rate
        return dense_block
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __model__(self, key):

        model_dict = {
        'Densenet121' : [[6,12,24,16], 32],
        'Densenet169' : [[6,12,32,32], 32],
        'Densenet161' : [[6,12,36,24], 48],
        'Densenet201' : [[6,12,48,32], 32]
        }

        return model_dict[key]

li = ['Densenet121', 'Densenet169', 'Densenet161','Densenet201',] 

for l in li:
    model = DenseNet(l, BottleNeck, num_class=100)

    x = torch.randn(3, 3, 224, 224)
    output = model(x)

    print(output.size())