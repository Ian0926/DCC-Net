import torch
import torch.nn as nn
import torch.nn.functional as F

class color_net(nn.Module):

    def __init__(self, d_hist=64):
        super(color_net, self).__init__()
        self.g_net = g_net()
        self.c_net = c_net(d_hist)
        self.r_net = r_net()
        
    def forward(self, img_low):
        gray = self.g_net(img_low)
        color_hist, color_feature = self.c_net(img_low)
        img_enhance = self.r_net(img_low, gray, color_feature)
        
        return gray, color_hist, img_enhance
    
class pce(nn.Module):
    # parmid color embedding

    def __init__(self):
        super(pce, self).__init__()

        self.cma_3 = cma(128, 64)
        self.cma_2 = cma(64, 32)
        self.cma_1 = cma(32, 16)
        
    def forward(self, c, shortcuts):
        
        # change channels
        x_3_color, c_2 = self.cma_3(c, shortcuts[2])
        x_2_color, c_1 = self.cma_2(c_2, shortcuts[1])
        x_1_color, _ = self.cma_1(c_1, shortcuts[0])
        
        return [x_1_color, x_2_color, x_3_color]
        
class cma(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(cma, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode='nearest'))
        
    def forward(self, c, x):
        # x: gray image features 
        # c: color features

        # l1 distance
        channels = c.shape[1]
        sim_mat_l1 = -torch.abs(x-c) # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True) # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1) # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1,channels,1, 1)
        sim_mat_l1 = 2*sim_mat_l1 # (0, 1)

        # cos distance
        sim_mat_cos = x*c # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True) # (b,1,h,w)       
        sim_mat_cos = torch.tanh(sim_mat_cos) # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1,channels,1, 1) # (0, 1)
        
        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos # (0, 1)
        
        # color embeding
        x_color = x + c*sim_mat
        
        # color features upsample
        c_up = self.conv(c)
        
        return x_color, c_up
           
class r_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(r_net, self).__init__()
        
        base_channel = 32
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(4, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
       
    def forward(self, img_low, gray, color_feature):
        x = torch.cat([img_low, gray], 1)
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        shortcuts = self.pce(color_feature, shortcuts)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        img_color = (torch.tanh(x) + 1) / 2
        return img_color

class c_net(nn.Module):

    def __init__(self, d_hist, depth=[2, 2, 2]):
        super(c_net, self).__init__()
        
        base_channel = 32
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
        ])

        self.conv_first = BasicConv(3, base_channel, 3, 1)
        
        # color hist
        self.conv_color = BasicConv(base_channel*4, 256*3, 3, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_hist)
        self.softmax = nn.Softmax(dim=2)

        self.d_hist = d_hist
        
    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def color_forward(self, x):
        x = self.conv_color(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))
        color_hist = self.softmax(self.fc(x))
        return color_hist
        
    def forward(self, x):
        
        x = self.conv_first(x)
        x, _ = self.encoder(x)
        color_hist = self.color_forward(x)
        
        return color_hist, x
 
class g_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(g_net, self).__init__()

        base_channel = 32
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 1, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
        
    def forward(self, x):
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray
 
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)
        
    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)
