import torch
import torch.nn as nn


class STA_Module(nn.Module):

    def __init__(self, in_dim):
        super(STA_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s, t, st):
        m_batchsize, C, height, width = s.size()
        proj_query = self.query_conv(s).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(t).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(s).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out * st

        return out

class STCLSTM(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, padding):
        super(STCLSTM, self).__init__()

        self.sta = STA_Module(num_hidden)
        self.num_hidden = num_hidden
        self.padding = padding
        self._forget_bias = 1.0
        # self.alpha = nn.Parameter(torch.zeros(1))
        self.en_t = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.en_s = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.en_o = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        self.att_c = nn.Sequential(
            nn.Conv2d(num_hidden // 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.att_m = nn.Sequential(
            nn.Conv2d(num_hidden // 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        # self.conv_sto = nn.Conv2d(num_hidden * 3, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, st_temporal_vit, st_spatial_vit, st_features, h_t, m_t, c_att, m_att):
        # st_temporal_vit, st_spatial_vit, st_features,
        # temporal_features[-1], memory_t, c_att, st_spatial

        # frames_feature_t, frames_feature_s, frames_feature_o,
        # h_t[0], c_t[0], memory, c_att, m_att

        st_features = st_features + self.sta(st_spatial_vit, st_temporal_vit, st_features)  # 0.00585
        x_t = self.en_t(st_temporal_vit)
        x_s = self.en_s(st_spatial_vit)
        o_x = self.en_o(st_features)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x = x_t, x_t, x_t
        i_x_prime, f_x_prime, g_x_prime = x_s, x_s, x_s
        i_h, f_h, g_h, o_h = h_concat, h_concat, h_concat, h_concat
        i_m, f_m, g_m = m_concat, m_concat, m_concat

        # Spatial Module
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_att_merge = self.att_m(m_att)
        m_new = i_t_prime * g_t_prime + f_t_prime * m_att_merge

        # Temporal module
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_att_merge = self.att_c(c_att)
        c_new = i_t * g_t + f_t * c_att_merge

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, m_new