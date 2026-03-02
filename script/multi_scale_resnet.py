import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class ScaleGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ScaleGroup, self).__init__()
        self.change_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.blocks = nn.Sequential(
            *[ResBlock(out_channels, out_channels) for _ in range(num_blocks)]
        )
        self.output = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.change_channels(x)
        x = self.blocks(x)
        x = self.output(x)
        return x

class MultiScaleResNet(nn.Module):
    def __init__(self, config):
        super(MultiScaleResNet, self).__init__()
        
        intermediate_channel_num = 512
        self.group1 = ScaleGroup(128 * 3, intermediate_channel_num, 5)
        self.tail1 = ScaleGroup(intermediate_channel_num, 128, 3)
        self.group2 = ScaleGroup(128 * 3 + intermediate_channel_num, intermediate_channel_num, 5)
        self.tail2 = ScaleGroup(intermediate_channel_num, 128, 3)
        self.group3 = ScaleGroup(256 * 3 + intermediate_channel_num, intermediate_channel_num, 5)
        self.tail3 = ScaleGroup(intermediate_channel_num, 256, 3)
        self.group4 = ScaleGroup(512 * 3 + intermediate_channel_num, intermediate_channel_num, 5)
        self.tail4 = ScaleGroup(intermediate_channel_num, 512, 3)
        
        self.final_conv = nn.Conv2d(intermediate_channel_num, 4, kernel_size=1, bias=True)
    
    def forward(self, features):
        f1, f2, f3, f4 = features

        frame_num = f1.shape[1]

        f1 = rearrange(f1, "b f c h w -> (b f) c h w")
        f2 = rearrange(f2, "b f c h w -> (b f) c h w")
        f3 = rearrange(f3, "b f c h w -> (b f) c h w")
        f4 = rearrange(f4, "b f c h w -> (b f) c h w")

        out1 = self.group1(f1)
        tail1 = self.tail1(out1)

        out1_ = F.interpolate(out1, scale_factor=0.5, mode='bilinear', align_corners=True)
        out2_input = torch.cat([f2, out1_], dim=1)
        out2 = self.group2(out2_input)
        tail2 = self.tail2(out2)
        
        out2_ = F.interpolate(out2, scale_factor=0.5, mode='bilinear', align_corners=True)
        out3_input = torch.cat([f3, out2_], dim=1)
        out3 = self.group3(out3_input)
        tail3 = self.tail3(out3)
        
        out3_ = F.interpolate(out3, scale_factor=0.5, mode='bilinear', align_corners=True)
        out4_input = torch.cat([f4, out3_], dim=1)
        out4 = self.group4(out4_input)
        tail4 = self.tail4(out4)
        
        final_out = self.final_conv(out4)

        final_out = rearrange(final_out, "(b f) c h w -> b f c h w", f=frame_num)
        tail1 = rearrange(tail1, "(b f) c h w -> b f c h w", f=frame_num)
        tail2 = rearrange(tail2, "(b f) c h w -> b f c h w", f=frame_num)
        tail3 = rearrange(tail3, "(b f) c h w -> b f c h w", f=frame_num)
        tail4 = rearrange(tail4, "(b f) c h w -> b f c h w", f=frame_num)
        tail_out = [
            tail1,
            tail2,
            tail3,
            tail4,
        ]

        return final_out, tail_out

if __name__ == "__main__":
    features = [torch.rand(1, 128*3, 320, 320).cuda(), torch.rand(1, 128*3, 160, 160).cuda(), torch.rand(1, 256*3, 80, 80).cuda(), torch.rand(1, 512*3, 40, 40).cuda()]
    model = MultiScaleResNet().cuda()
    final_out, tail_out = model(features)