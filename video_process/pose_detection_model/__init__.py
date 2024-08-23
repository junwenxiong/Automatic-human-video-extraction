# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody

openpose2coco_order = [0, 15, 14, 17, 16, 1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
# 满足13个点，到腰部
# 满足15个点，到膝盖
# 满足15个点，到脚踝


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


def has_complete_body(candidate, subset, H, W):

    num_skeleton = subset.shape[0]
    skeleton_results = []
    for i in range(num_skeleton):
        skeleton = []
        for j in openpose2coco_order:
            if int(subset[i][j]) == -1:
                x, y = -1, -1
            else:
                x, y = candidate[int(subset[i][j])]
            skeleton.append([int(x), int(y)])
        skeleton_results.append({"keypoints": skeleton})

    keypoints = skeleton_results[0]["keypoints"]

    # print("keypoints: ", subset)
    # print("keypoints: ", keypoints)
    # TODO 应该17个关键点都存在
    # 如何存在一个关键点的位置没有，返回False

    # 0 表示 上半身都没有
    # 1 表示 上半身有，下半身没有
    # 2 表示 上半身有，下半身有，但是膝盖没有
    # 3 表示 上半身有，下半身有，膝盖有，但是脚踝没有
    # 4 表示 上半身有，下半身有，膝盖有，脚踝有
    for i in range(18):
        if i < 12 and keypoints[i][0] == -1:
            return 0
        elif i >= 12 and i < 14 and keypoints[i][0] == -1:
            return 1
        elif i >= 14 and i < 16 and keypoints[i][0] == -1:
            return 2
        elif i >= 16 and keypoints[i][0] == -1:
            return 3
    return 4

    # for i in range(12, 15):
    #     if keypoints[i][0] == -1:
    #         return False

    # for i in range(15, 18):
    #     if keypoints[i][0] == -1:
    #         return False

    # 用mask来约束
    # tmp_keypoints = []
    # for kp in keypoints:
    #     Y = kp[0] * float(W)
    #     X = kp[1] * float(H)
    #     tmp_keypoints.append([Y, X])

    # # 需要增加面积约束
    # image_with_contour = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # keypoints_array = np.array(keypoints, dtype=np.int32)

    # cv2.drawContours(
    #     image_with_contour, [keypoints_array], 0, (255, 255, 255), thickness=cv2.FILLED
    # )
    # contour_area = cv2.contourArea(keypoints_array)
    # print("contour_area: ", contour_area)

    # # 计算整个图像面积
    # total_area = H * W
    # # 计算人体面积与整个图像面积的比例
    # area_ratio = contour_area / total_area

    # 打印结果
    # print("人体面积与整个图像面积的比例：", area_ratio)

    # return True


class DWposeDetector:
    def __init__(self):
        pass

    def to(self, device):
        self.pose_estimation = Wholebody(device)
        return self

    def cal_height(self, input_image):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            # candidate[..., 0] /= float(W)
            # candidate[..., 1] /= float(H)
            body = candidate
        return body[0, ..., 1].min(), body[..., 1].max() - body[..., 1].min()

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        **kwargs,
    ):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            score = subset[:, :18]
            max_ind = np.mean(score, axis=-1).argmax(axis=0)
            score = score[[max_ind]]
            body = candidate[:, :18].copy()
            body = body[[max_ind]]
            nums = 1
            body = body.reshape(nums * 18, locs)
            body_score = copy.deepcopy(score)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[[max_ind], 24:92]

            hands = candidate[[max_ind], 92:113]
            hands = np.vstack([hands, candidate[[max_ind], 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            detected_map = None
            is_body = -1

            candidate = bodies["candidate"]
            subset = bodies["subset"]
            candidate = np.array(candidate)
            subset = np.array(subset)
            num_skeleton = subset.shape[0]

            if num_skeleton == 1:  # 只处理一个人体
                is_body = has_complete_body(candidate, subset, H, W)
                # print("one body detected")

                if is_body:
                    detected_map = draw_pose(pose, H, W)
                    detected_map = HWC3(detected_map)

                    img = resize_image(input_image, image_resolution)
                    H, W, C = img.shape

                    detected_map = cv2.resize(
                        detected_map, (W, H), interpolation=cv2.INTER_LINEAR
                    )

                    if output_type == "pil":
                        detected_map = Image.fromarray(detected_map)

            return detected_map, body_score, is_body
