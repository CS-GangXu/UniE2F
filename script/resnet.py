import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from script.simple_vit import SimpleViT
from diffusers import StableVideoDiffusionPipeline
import copy
from script.util import load_config, dump_config, normalize_and_save_tensor, feature2image, tensor2image, encode_and_decode, decode, process_event
from einops import rearrange
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class TailBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TailBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(int(in_channels/2), int(in_channels/4), kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(int(in_channels/4), out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        
        self.initial_conv = nn.Conv2d(config['input_channels'], 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.initial_layer = ResidualBlock(64 , config['intermediate_layer_channels'], stride=1)
        self.intermediate_layers = nn.ModuleList()
        for i in range(config['intermediate_layer_num']):
            self.intermediate_layers.append(ResidualBlock(config['intermediate_layer_channels'], config['intermediate_layer_channels'], stride=1))
        self.final_layer = TailBlock(config['intermediate_layer_channels'], config['output_channels'])
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.relu(x)
        x = self.initial_layer(x)
        for intermediate_layer in self.intermediate_layers:
            x = intermediate_layer(x)
        x = self.final_layer(x)
        
        return x

class ResNetWithDownscale(nn.Module):
    def __init__(self, config):
        super(ResNetWithDownscale, self).__init__()
        
        self.initial_conv = nn.Conv2d(config['input_channels'], 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.initial_layer = ResidualBlock(64 , config['intermediate_layer_channels'], stride=1)
        self.intermediate_layers = nn.ModuleList()

        group_num = int(math.log2(config['downscale']))
        assert config['intermediate_layer_num'] % group_num == 0
        layer_num_per_group = int(config['intermediate_layer_num'] / group_num)
        for i in range(group_num):
            self.intermediate_layers.append(ResidualBlock(config['intermediate_layer_channels'], config['intermediate_layer_channels'], stride=2))
            for j in range(1, layer_num_per_group):
                self.intermediate_layers.append(ResidualBlock(config['intermediate_layer_channels'], config['intermediate_layer_channels'], stride=1))
        
        self.final_layer = TailBlock(config['intermediate_layer_channels'], config['output_channels'])
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.relu(x)
        x = self.initial_layer(x)
        for intermediate_layer in self.intermediate_layers:
            x = intermediate_layer(x)
        x = self.final_layer(x)
        
        return x