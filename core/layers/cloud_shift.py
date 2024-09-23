from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import torch
import cv2
import scipy.io as io

def find_cloud_center(cloud, batch_size, configs):
    cloud_shift_value = []
    cloudless_shift_value = []
    input_length = configs.input_length - 1
    # convert the grayscale image to binary image
    # first_cloud.shape=[17, 128, 128]
    for i in range(batch_size):
        cloud_X_list = []
        cloud_Y_list = []
        cloudless_X_list = []
        cloudless_Y_list = []
        """cloud_d_value_X = []
        cloud_d_value_Y = []
        cloudless_d_value_X = []
        cloudless_d_value_Y = []"""
        cloud_diff_X = 0
        cloud_diff_Y = 0
        cloudless_diff_X = 0
        cloudless_diff_Y = 0
        for j in range(configs.input_length):
            _, cloud_thresh = cv2.threshold(cloud[:, j, :, :], 0.1, 1, 0)
            cloudless_cloud = 1 - cloud[:, j, :, :]
            _, cloudless_thresh = cv2.threshold(cloudless_cloud, 0.95, 1, 0)
            # calculate moments of binary image
            cloud_point = cv2.moments(cloud_thresh[i, :, :])
            cloudless_point = cv2.moments(cloudless_thresh[i, :, :])
            # calculate x,y coordinate of center
            # 计算有云的区域
            if cloud_point["m00"] == 0:
                cloud_center_X = 1.0
                cloud_center_Y = 1.0
            else:
                cloud_center_X = int(cloud_point["m10"] / cloud_point["m00"])
                cloud_center_Y = int(cloud_point["m01"] / cloud_point["m00"])
            cloud_X_list.append(cloud_center_X)
            cloud_Y_list.append(cloud_center_Y)

            # 计算没有云的区域
            if cloudless_point["m00"] == 0:
                cloudless_center_X = 0.0
                cloudless_center_Y = 0.0
            else:
                cloudless_center_X = int(cloudless_point["m10"] / cloudless_point["m00"])
                cloudless_center_Y = int(cloudless_point["m01"] / cloudless_point["m00"])
            cloudless_X_list.append(cloudless_center_X)
            cloudless_Y_list.append(cloudless_center_Y)

            # 求位移差
            if j > 0:
                cloud_diff_X += cloud_X_list[j] - cloud_X_list[j-1]
                cloud_diff_Y += cloud_Y_list[j] - cloud_Y_list[j-1]
                cloudless_diff_X += cloudless_X_list[j] - cloudless_X_list[j - 1]
                cloudless_diff_Y += cloudless_Y_list[j] - cloudless_Y_list[j - 1]

        cloud_d_value_X = float(cloud_diff_X)
        cloud_d_value_Y = float(cloud_diff_Y)
        cloudless_d_value_X = float(cloudless_diff_X)
        cloudless_d_value_Y = float(cloudless_diff_Y)

        cloud_theta = [[-1, 0, cloud_d_value_Y],
                       [0, -1, cloud_d_value_X]]
        cloud_shift_value.append(cloud_theta)

        cloudless_theta = [[-1, 0, cloudless_d_value_Y],
                           [0, -1, cloudless_d_value_X]]
        cloudless_shift_value.append(cloudless_theta)
    return cloud_shift_value, cloudless_shift_value


class CloudShift(nn.Module):
    def __init__(self):
        super(CloudShift, self).__init__()


    def forward(self, cloud, cloudless, cloud_shift, cloudless_shift):
        N, C, H, W = cloud.shape
        # 右移为负，左移为正
        cloud_shift = 0 - cloud_shift
        cloudless_shift = 0 - cloudless_shift
        # 向右移动:-0.2，向下移动:-0.4
        # F.affine_grid以宽高的比率进行平移变换，需要对其变换矩阵进行一个转换
        grid = F.affine_grid(cloud_shift, [N, C, H, W], align_corners=True)
        cloud_shift_output = F.grid_sample(cloud, grid.cuda(), align_corners=True)

        gridless = F.affine_grid(cloudless_shift, [N, C, H, W], align_corners=True)
        cloudless_shift_output = F.grid_sample(cloudless, gridless.cuda(), align_corners=True)

        # cloud_shift_output1 = np.array(cloud_shift_output[0, :, :, :].cpu())
        # io.savemat('/home/code/PredRANN-master/cloud_shift_output.mat', {'cloud_shift_output': cloud_shift_output1})

        return cloud_shift_output, cloudless_shift_output

#  np.array(cloud.data.cpu())