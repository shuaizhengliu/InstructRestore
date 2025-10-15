from einops import rearrange

import torch 
import torch.nn as nn 
import torch.nn.functional as F



def get_mask_decoder(
    module_type,
    in_channels,
):
    if module_type == "MaskDecoder":
        return MaskDecoder(
            in_channels=in_channels,
        )
    else:
        return None

class MaskDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        reduced_channel = 512,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.projects = nn.ModuleList()
        for i in range(len(in_channels)):
            if i == 0:
                channel = in_channels[i]
            else:
                channel = reduced_channel + in_channels[i]
            self.projects.append(MultiScaleProj(channel, reduced_channel))

        # last projection map
        self.final_conv = nn.Conv2d(reduced_channel, 1, 1)
    
    def forward(self, features):

        #print('length of features', len(features))
        #print('length of self.in_channels', len(self.in_channels))

        x = features[0]
        for i in range(len(self.in_channels)):
            #print(i)
            x = self.projects[i](x)
            #print('before interpolate, x shape', x.shape)
            if i < len(self.in_channels) - 1:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i < len(self.in_channels) - 1:
                #print('after inter, x shape', x.shape)
                #print('features shape', features[i+1].shape)
                x = torch.cat([x, features[i+1]], dim=1)

        x = self.final_conv(x)
        x = F.sigmoid(x)
        return x

class MultiScaleProj(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, 1, bias=False)
        
        self.norm1 = nn.GroupNorm(32, C_out)
        self.norm2 = nn.GroupNorm(32, C_out)
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        return x
        