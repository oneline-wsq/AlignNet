from pyquaternion import Quaternion
import numpy as np

def get_proj_mat3(img_txt):
    f = open(img_txt)
    data = f.readlines()
    ego2globals = {}
    for i in range(len(data)):
        data2 = data[i].split(' ')
        # print(data2)
        trans = [float(data2[1]), float(data2[2]), 0]
        angle = float(data2[4])
        quat = Quaternion(axis=[0, 0, 1], radians=angle).rotation_matrix
        # quat = Quaternion(axis=[0, 0, 1], degrees=angle).rotation_matrix
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = quat
        ego2global[:3, 3] = np.array(trans)
        #ego2globals[data2[0]] = ego2global
        ego2globals[data2[0]] = [trans, angle]

    return ego2globals

def sample_pts_from_line(line):
    fixed_num = -1
    sample_dist = 1
    padding = False
    num_samples = 250
    if fixed_num < 0:
        distances = np.arange(0, line.length, sample_dist)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    else:
            # fixed number of points, so distance is line.length / self.fixed_num
        distances = np.linspace(0, line.length, fixed_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)


    num_valid = len(sampled_points)

    if not padding or fixed_num > 0:
        # fixed num sample can return now!
        return sampled_points, num_valid

    # fixed distance sampling need padding!
    num_valid = len(sampled_points)

    if fixed_num < 0:
        if num_valid < num_samples:
            padding = np.zeros((num_samples - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        else:
            sampled_points = sampled_points[:num_samples, :]
            num_valid = num_samples

    return sampled_points, num_valid