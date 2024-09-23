import torch
import torch.nn as nn
from core.layers.vision_transformer import VisionTransformer
import scipy.io as io
import numpy as np
from core.layers.STCLSTM import STCLSTM

class DualViT(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(DualViT, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        self.padding = filter_size // 2
        self.alpha = nn.Parameter(torch.ones(1))
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, self.num_hidden, filter_size, stride, self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_temporal = nn.Sequential(
            nn.Conv2d(self.num_hidden, self.num_hidden // 8, filter_size, stride, self.padding, bias=False)
        )
        self.conv_accumulate_temporal_features = nn.Sequential(
            nn.Conv2d(self.num_hidden, self.num_hidden // 8, filter_size, stride, self.padding, bias=False)
        )

        self.conv_prior = nn.Sequential(
            nn.Conv2d(self.num_hidden // 4, self.num_hidden // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.num_hidden // 2, self.num_hidden // 2, filter_size, stride, self.padding, bias=False)
        )
        self.conv_spatial_cat = nn.Sequential(
            nn.Conv2d(self.num_hidden, self.num_hidden // 2, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv_spatial = nn.Conv2d(self.num_hidden, self.num_hidden // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.h_t_2_c_att = nn.Conv2d(self.num_hidden, self.num_hidden // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_temporal_up = nn.Conv2d(height, self.num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_spatial_up = nn.Conv2d(height, self.num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_prior_up = nn.Conv2d(height, self.num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.vit_temporal = VisionTransformer(img_size=32, patch_size=16, in_chans=32)
        self.STCLSTM = STCLSTM(in_channel=64, num_hidden=64, height=32, width=32, filter_size=[5, 5], stride=1, padding=(2, 2))

    def forward(self, x_t, h_t, temporal_features, memory_t, accumulate_temporal_features, true_frames):
        # x_t = [35, 16, 32, 32] pre_result = [35, 16, 32, 32]
        x_concat = self.conv_x(x_t)
        st_spatial = self.conv_spatial(x_concat)
        prior_features = self.conv_prior(true_frames)
        st_spatial = self.conv_spatial_cat(torch.cat((st_spatial, prior_features), 1))
        # st_vit
        # st_spatial_vit = st_spatial
        if len(temporal_features) > 2:
            temporal_features_0 = self.conv_temporal(x_concat)
            temporal_features_1 = self.conv_temporal(temporal_features[-1])
            temporal_features_2 = self.conv_temporal(temporal_features[-2])
            temporal_features_3 = self.conv_temporal(temporal_features[-3])
            temporal_features_4 = self.conv_accumulate_temporal_features(accumulate_temporal_features)
            st_temporal = torch.cat((temporal_features_0, temporal_features_3,
                                     temporal_features_2, temporal_features_1), 1)
            c_att = torch.cat((temporal_features_3, temporal_features_2,
                                     temporal_features_1, temporal_features_4), 1)
            st_temporal.permute(0, 3, 2, 1).contiguous()
            st_spatial_vit, st_temporal_vit = self.vit_temporal(st_spatial, st_temporal)
            st_temporal_vit.permute(0, 3, 2, 1).contiguous()
            st_temporal_vit = self.conv_temporal_up(st_temporal_vit)
            st_spatial_vit = self.conv_spatial_up(st_spatial_vit)
            st_features = st_temporal_vit * st_spatial_vit * self.conv_prior_up(prior_features)
            st_results, m_new = self.STCLSTM(st_temporal_vit, st_spatial_vit, st_features,
                                             h_t, memory_t, c_att, st_spatial)
        else:
            st_spatial_vit, _ = self.vit_temporal(st_spatial, None)
            st_spatial_vit = self.conv_spatial_up(st_spatial_vit)
            st_features = st_spatial_vit * self.conv_prior_up(prior_features)
            c_att = self.h_t_2_c_att(h_t)
            st_results, m_new = self.STCLSTM(h_t, st_spatial_vit, st_features,
                                             h_t, memory_t, c_att, st_spatial)
        temporal_features.append(st_results)
        accumulate_temporal_features = self.alpha * accumulate_temporal_features + st_results
        results = st_results + st_features
        return results, m_new, accumulate_temporal_features


