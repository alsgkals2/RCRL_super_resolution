import torch
from model import common

import torch.nn as nn
import torch.nn.functional as F
def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, n_feats, conv=common.default_conv, synchronize_norm = False):
        super(EDSR, self).__init__()
        self.synchronize_norm = synchronize_norm
        self.device = 'cuda'

        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # rgb_range=1.0
        rgb_range=255
        n_colors=3
        res_scale=0.1
        # res_scale=1# origin
        n_resblock = 32 #(x4다룰 때 이거)
#         n_resblock = 16 #origin (x2다룰때 이거, 아마도..)
        n_feats = 256
        kernel_size = 3 
        scale = 4
        
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, sign=-1)#rgb_std)
        
        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)] 

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, sign=1)#rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    # def forward(self, x):
    #     print("EDSREDSREDSREDSREDSREDSR")
    #     # print("forward(self, x)forward(self, x)forward(self, x)")
    #     # print(x)
    #     if self.synchronize_norm:
    #         # When input is normalized mean 0.5, std 0.5
    #         x = x*0.5
    #         x += torch.tensor([[[0.0512]], [[0.0629]], [[0.096]]], device=self.device)
            
    #     else:
    #         # When input is in [0,1]
    #         x = x*1
    #         # print(x)
    #         x -= torch.tensor([[[0.4488]], [[0.4371]], [[0.4040]]], device=self.device)
    #     x = self.head(x)
    #     print("head(x)head(x)head(x)head(x)")
    #     print(x.mean())
    #     res = self.body(x)
    #     res += x

    #     x = self.tail(res)
    #     print("tail(res)tail(res)tail(res)tail(res)")
    #     print(x.mean())
    #     x += torch.tensor([[[0.4488]], [[0.4371]], [[0.4040]]], device=self.device) # denormalization
    #     return x

# #original code
# """
    def forward(self, x):
        print(x.shape)
        x = x * 255
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        x = x / 255.0
        return x 
# """
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

