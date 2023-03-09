import cv2
import numpy as np

import torch

from shapely import affinity
from shapely.geometry import LineString, box


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        # color BGR三个通道的值
        # thickness 画线的粗细
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box
    canvas_h = canvas_size[0] # 画布大小
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h # 200/30
    scale_width = canvas_w / patch_w  # 400/60

    trans_x = -patch_x + patch_w / 2.0 # -x+60/2=-x+30
    trans_y = -patch_y + patch_h / 2.0 # -y+30/2=-y+15

    map_mask = np.zeros(canvas_size, np.uint8) # 对于每一类，生成一个mask

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:
                    # 相当于一条 lane 对应一个 idx
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
                # print('end')
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape # C==类别数, (1, 1000, 1000)
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0) # None 增加了一个维度
        mask[:c][filter] = 0 # 将filter所有不等于0的地方置零

    return mask


def preprocess_map(vectors, patchxy, patch_size, canvas_size, num_classes, thickness=2, angle_class=36):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))

    local_box = (patchxy[0], patchxy[1], patch_size[0], patch_size[1])

    idx = 1 # idx从1开始
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        # 这里最后的 idx 相当于该类的所有线的总数
        # map_mask中很多都是255, 是因为线的总数太多了，导致大于255的都成了255（uint8）
        
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1) # 画粗一点的线
        filter_masks.append(filter_mask)
        # forward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        # forward_masks.append(forward_mask)
        # backward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        # backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    # forward_masks = np.stack(forward_masks)
    # backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    # forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    # backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    return torch.tensor(instance_masks) #, torch.tensor(forward_masks), torch.tensor(backward_masks)


def rasterize_map(vectors, patch_size, canvas_size, num_classes, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
