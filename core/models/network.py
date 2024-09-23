import torch
import torch.nn as nn
import scipy.io as io
import numpy as np
from core.layers.DualViT import DualViT


class Network(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(Network, self).__init__()
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour

        # self.spynet = SPyNet(pretrained=None)
        cell_list = []
        for i in range(num_layers):
            in_channel = self.patch_ch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                DualViT(in_channel, num_hidden[i], self.patch_height, self.patch_width, configs.filter_size, configs.stride)
            )
        enc_list = []
        for i in range(num_layers - 1):
            enc_list.append(
                nn.Conv2d(num_hidden[i], num_hidden[i] // 4, kernel_size=configs.filter_size, stride=2,
                          padding=configs.filter_size // 2),
            )
        dec_list = []
        for i in range(num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(num_hidden[i] // 4, num_hidden[i], kernel_size=4, stride=2,
                                   padding=1),
            )
        gate_list = []
        for i in range(num_layers - 1):
            gate_list.append(
                nn.Conv2d(num_hidden[i] * 2, num_hidden[i], kernel_size=configs.filter_size, stride=1,
                          padding=configs.filter_size // 2),
            )

        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch, 1, stride=1, padding=0, bias=False)
        # self.conv_last_st_feature = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch, 1, stride=1, padding=0, bias=False)


    def forward(self, all_frames, mask_true, batch_size):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames
        mask_true = mask_true
        next_frames = []
        next_st = []
        final_frames = []
        h_t = []
        each_layers_pre_result= []
        # true_frames = []
        accumulate_temporal_features = []

        memory = torch.empty([batch_size, self.num_hidden[0], self.patch_height, self.patch_width]).cuda()
        nn.init.xavier_normal_(memory)

        for i in range(self.num_layers):
            pre_result = []
            zeros = torch.empty(
                [batch_size, self.num_hidden[i], self.patch_height, self.patch_width]).cuda()
            nn.init.xavier_normal_(zeros)
            h_t.append(zeros)
            accumulate_temporal_features.append(zeros)
            each_layers_pre_result.append(pre_result)

        # 设置每一层的参数
        for i in range(self.num_layers - 1):
            zeros = torch.empty(
                [batch_size, self.motion_hidden, self.patch_height // 2, self.patch_width // 2])
            nn.init.xavier_normal_(zeros)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                # net.shape=[35, 16, 32, 32]
                net = frames[:, t]
                true_frames = net

            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], memory, accumulate_temporal_features[0] = self.cell_list[0](net, h_t[0], each_layers_pre_result[0], memory,
                                                                             accumulate_temporal_features[0], true_frames)
            net = self.enc_list[0](h_t[0])
            h_t_tmp = self.dec_list[0](net)
            o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            for i in range(1, self.num_layers - 1):
                h_t[i], memory, accumulate_temporal_features[i] = self.cell_list[i](h_t[i - 1], h_t[i], each_layers_pre_result[i], memory,
                                                                                 accumulate_temporal_features[i], true_frames)
                net = self.enc_list[i](h_t[i])
                h_t_tmp = self.dec_list[i](net)
                o_t = torch.sigmoid(self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1)))
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            h_t[self.num_layers - 1], memory, accumulate_temporal_features[self.num_layers - 1] = self.cell_list[self.num_layers - 1]\
                (h_t[self.num_layers - 2], h_t[self.num_layers - 1], each_layers_pre_result[self.num_layers - 1], memory,
                 accumulate_temporal_features[self.num_layers - 1], true_frames)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            # x_gen_0 = self.conv_last_0(h_t[0])
            # st = self.conv_last_st_feature(h_t[self.num_layers - 2])
            next_frames.append(x_gen)
            # next_st.append(st)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        # next_st = torch.stack(next_st, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        final_frames.append(next_frames)
        # final_frames.append(next_st)
        return final_frames