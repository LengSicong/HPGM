import os
import cv2
from tqdm import tqdm
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import jaccard_score
from collections import defaultdict
from calculate_iou import cluster_pixel_values, generate_mask, generate_one_dim_mask, find_central_point, calculate_iou

# define rooms types and their corresponding colors
last_5k_type = ['84, 139, 84','255, 255, 255', '0, 0, 0', '255, 0, 255', '115, 198, 205', '139, 26, 85', '130, 134, 139', '128, 0, 0', '255, 144, 30', '58, 58, 139', '235, 206, 135', '42, 42, 165', '0, 100, 0']
# last_5k_type = ['255,255,255', '0,0,0', '0,0,255', '170,232,238', '128,128,240', '230,216,173', '0,215,255', '0,165,255', '35,142,107', '221,160,221', '0,255,255', '214,112,218']
last_dict = dict.fromkeys(last_5k_type)
for key in last_dict.keys():
    last_dict[key] = [int(val) for val in key.split(',')]

# calculate IoU
from tqdm import tqdm
Macro_IoUs = []
Micro_IoUs = []
for i in tqdm(range(12110)):
    img_id = i + 1
    img_id = format(img_id, '09d')
    gt_img_path = 'count_' + img_id + '_real_floor_plan.png'
    pred_img_path = 'count_' + img_id + '_fake_floor_plan.png'
    gt_image = cv2.imread('../output_bbox_gcn/g2p_layout_haroon_split_3stages_2023_02_22_17_30_32/Image_eval/' + gt_img_path)
    pred_image = cv2.imread('../output_bbox_gcn/g2p_layout_haroon_split_3stages_2023_02_22_17_30_32/Image_eval/' + pred_img_path)

    # new_image = cluster_pixel_values(pred_image, last_dict)

    dict_pred_mask = generate_mask(pred_image, last_dict)
    dict_gt_mask = generate_mask(gt_image, last_dict)

    one_dim_pred_mask = generate_one_dim_mask(dict_pred_mask)
    one_dim_gt_mask = generate_one_dim_mask(dict_gt_mask)

    rooms = ['84, 139, 84','255, 255, 255', '0, 0, 0', '255, 0, 255', '115, 198, 205', '139, 26, 85', '130, 134, 139', '128, 0, 0', '255, 144, 30', '58, 58, 139', '235, 206, 135', '42, 42, 165', '0, 100, 0']
    rooms_wo_outside = ['84, 139, 84', '0, 0, 0', '255, 0, 255', '115, 198, 205', '139, 26, 85', '130, 134, 139', '128, 0, 0', '255, 144, 30', '58, 58, 139', '235, 206, 135', '42, 42, 165', '0, 100, 0']
    # rooms = ['170,232,238', '128,128,240', '230,216,173', '0,215,255', '0,165,255', '35,142,107', '221,160,221', '0,255,255', '214,112,218']
    # IoUs, max_key = find_central_point(one_dim_gt_mask, one_dim_pred_mask)
    max_key = (0,0) # no shifting
    macro_iou, micro_iou = calculate_iou(rooms, dict_gt_mask, dict_pred_mask, max_key)
    
    Macro_IoUs.append(macro_iou)
    Micro_IoUs.append(micro_iou)

print(f"Macro IoU: {np.mean(Macro_IoUs)}")
print(f"Micro IoU: {np.mean(Micro_IoUs)}")