# -*- coding: utf-8 -*-
import random
import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import torch.nn as nn
import numpy as np
from .text_detection_model import craft_utils
from .text_detection_model import imgproc
import json
from .text_detection_model.craft import CRAFT
from collections import OrderedDict
import torch.multiprocessing as mp

from .pose_detection_model import DWposeDetector
from .human_segment_model.model.model import HumanMatting
from .human_segment_model.inference import single_inference
from PIL import Image
from collections import OrderedDict
from ultralytics import YOLO
from collections import Counter
import mediapipe

detect_results = {}
tmp_results = {}

# MediaPipe Hands 初始化
mp_hands = mediapipe.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
# )
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)

# MediaPipe Face Mesh 初始化
mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
)


# 检测面部
def detect_face(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    return results.multi_face_landmarks


# 检测手部
def detect_hands(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    return results.multi_hand_landmarks


# 计算图像的模糊度
def calculate_blurriness(image):
    if image is None or image.size == 0:
        return float("inf")  # 返回一个非常大的值，表示图像是空的或无效的

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


# 判断手部是否清晰
def is_hand_clear(frame, hand_landmarks, threshold=100.0):
    for hand_landmark in hand_landmarks:
        x_coords = [lm.x for lm in hand_landmark.landmark]
        y_coords = [lm.y for lm in hand_landmark.landmark]
        x_min, x_max = int(min(x_coords) * frame.shape[1]), int(
            max(x_coords) * frame.shape[1]
        )
        y_min, y_max = int(min(y_coords) * frame.shape[0]), int(
            max(y_coords) * frame.shape[0]
        )
        hand_region = frame[y_min:y_max, x_min:x_max]
        if hand_region.size > 0 and calculate_blurriness(hand_region) > threshold:
            return True
    return False


# 判断是否正脸
def is_face_facing_camera(face_landmarks, threshold=0.1):
    if not face_landmarks:
        return False

    face_landmark = face_landmarks[0]
    nose_tip = face_landmark.landmark[1]
    left_cheek = face_landmark.landmark[234]
    right_cheek = face_landmark.landmark[454]

    nose_x = nose_tip.x
    left_cheek_x = left_cheek.x
    right_cheek_x = right_cheek.x

    face_width = abs(right_cheek_x - left_cheek_x)
    nose_center_ratio = abs(nose_x - (left_cheek_x + right_cheek_x) / 2) / face_width

    return nose_center_ratio < threshold


def face_hands_detect(i, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    hand_landmarks = detect_hands(frame)
    face_landmarks = detect_face(frame)

    # 得继续实验
    hand_clear = False
    if hand_landmarks:
        # import ipdb; ipdb.set_trace()
        if len(hand_landmarks) == 2:  # 确保检测到两只手
            # if all(
            #     is_hand_clear(frame, [hand_landmark])
            #     for hand_landmark in hand_landmarks
            # ):
            hand_clear = True

    face_facing_camera = False
    if face_landmarks:
        face_facing_camera = is_face_facing_camera(face_landmarks)

    return hand_clear, face_facing_camera


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(
    net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None
):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image,
        1280,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=1.5,
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def detect_video(arg):
    thread_id, gpu_id, video_lists = arg
    # 设置线程使用的GPU
    torch.cuda.set_device(gpu_id)

    # 加载CRAFT模型
    net = CRAFT()
    pretrained_path = (
        "./video_process/CRAFT_pytorch/pretrained_weight/craft_mlt_25k.pth"
    )
    print("Loading weights from checkpoint (" + pretrained_path + ")")

    net.load_state_dict(copyStateDict(torch.load(pretrained_path)))
    net = net.cuda()
    cudnn.benchmark = False
    net.eval()

    # LinkRefiner
    refine_net = None

    results = {}
    for video_path in video_lists:
        cap = cv2.VideoCapture(video_path)  # 替换为你的视频路径
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0  # 选择起始帧
        mid_frame = total_frames // 2  # 选择中间帧
        end_frame = total_frames - 1  # 选择末尾帧
        frame_list = [start_frame, mid_frame, end_frame]

        # 根据视频帧数调整选择的帧数
        if total_frames > 32:
            start_frame2 = total_frames // 8  # 选择起始帧
            start_frame3 = total_frames // 8 * 3  # 选择起始帧
            mid_frame1 = total_frames // 4  # 选择中间帧
            start_frame4 = total_frames // 8 * 5  # 选择起始帧
            mid_frame2 = total_frames // 4 * 3  # 选择中间帧
            end_frame2 = total_frames // 8 * 7  # 选择末尾帧
            frame_list.append(start_frame2)
            frame_list.append(mid_frame1)
            frame_list.append(mid_frame2)
            frame_list.append(end_frame2)

        detection_ratio = 0
        for frame_num in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                print("Error reading frame {}".format(frame_num))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(
                "Thread {} - Processing frame {:d}/{:d}".format(
                    thread_id, frame_num + 1, total_frames
                ),
                end="\r",
            )

            # 使用frame进行文本检测
            image = frame
            bboxes, polys, score_text = test_net(
                net,
                image,
                0.7,
                0.4,
                0.4,
                True,
                False,
                refine_net,
            )

            # 计算检测结果的面积
            detection_area = 0
            for poly in polys:
                if poly is not None:
                    detection_area += cv2.contourArea(np.array(poly, dtype=np.int32))

            # 计算整个帧的面积
            total_area = frame.shape[0] * frame.shape[1]

            # 计算比例
            detection_ratio += detection_area / total_area

            # 保存文本检测结果和得分图
            # filename = "frame_{:04d}".format(frame_num)
            # mask_file = save_path + "/res_" + filename + "_mask.jpg"
            # cv2.imwrite(mask_file, score_text)

        detection_ratio /= len(frame_list)

        results[video_path] = round(detection_ratio, 3)
        print("{}: {:.2%}".format(video_path, detection_ratio))

        cap.release()

    return results


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    在给定的帧上绘制 bounding box

    参数:
    frame (numpy.ndarray): 输入帧
    bbox (tuple): bounding box 的坐标 (x, y, width, height)
    color (tuple): 框的颜色 (B, G, R)
    thickness (int): 框的线条宽度

    返回:
    numpy.ndarray: 绘制了 bounding box 的帧
    """
    x, y, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x2, y2), color, thickness)
    return frame


def calculate_bbox_area_ratio(image_width, image_height, bbox):
    """
    计算边界框所占图像的面积之比
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :param bbox: 边界框坐标 (x, y, width, height)
    :return: 边界框所占图像的面积比
    """
    # 边界框坐标
    x, y, width, height = bbox

    # 计算边界框面积
    bbox_area = width * height

    # 计算图像总面积
    image_area = image_width * image_height

    # 计算边界框所占图像的面积比
    bbox_area_ratio = bbox_area / image_area

    return bbox_area_ratio


def calculate_overlap2(bbox, mask):
    overlaps = []
    h, w = mask.shape[:2]

    for box in bbox:
        if box is not None:
            # 取整bbox
            box = np.round(box).astype(int)
            # 计算 bbox 区域与 mask 的交集
            intersection = np.logical_and(
                mask[box[0, 1] : box[2, 1] + 1, box[0, 0] : box[2, 0] + 1], 255
            )
            intersection_pixels = np.sum(intersection)

            # 计算交集区域的像素数量与 bbox 区域面积的比值
            bbox_area = (box[2, 0] - box[0, 0] + 1) * (box[2, 1] - box[0, 1] + 1)
            overlap_ratio = intersection_pixels / bbox_area

            if overlap_ratio > 0:
                overlaps.append(overlap_ratio)

    overlap_ratio = np.mean(overlaps) if overlaps else 0.0  # 处理没有重叠的情况

    return overlap_ratio


def calculate_overlap(bbox, mask):
    overlaps = []
    intersection_list = []
    for box in bbox:
        if box is not None:
            x1, y1 = np.min(box[:, 0]), np.min(box[:, 1])
            x2, y2 = np.max(box[:, 0]), np.max(box[:, 1])
            bbox_area = max((x2 - x1 + 1) * (y2 - y1 + 1), 1)  # 避免面积为零

            bbox_mask = np.zeros_like(mask)
            cv2.rectangle(bbox_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

            intersection = np.logical_and(bbox_mask, mask)
            intersection_list.append(bbox_mask)

            intersection_pixels = np.sum(intersection)

            overlap_ratio = intersection_pixels / bbox_area
            overlaps.append(overlap_ratio)

    overlap_ratio = np.mean(overlaps) if overlaps else 0.0  # 处理没有重叠的情况

    return overlap_ratio, intersection_list


def get_union_bbox(bbox1, bbox2):
    """
    计算两个 bounding box 的并集

    参数:
    bbox1 (tuple): 第一个 bounding box 的坐标 (x1, y1, x2, y2)
    bbox2 (tuple): 第二个 bounding box 的坐标 (x1, y1, x2, y2)

    返回:
    tuple: 并集 bounding box 的坐标 (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 计算两个 bounding box 的左上角和右下角坐标
    x1 = min(x1_1, x1_2)
    y1 = min(y1_1, y1_2)
    x2 = max(x2_1, x2_2)
    y2 = max(y2_1, y2_2)

    return (x1, y1, x2, y2)


def detect_one_person_wo_occlusion(arg):
    thread_id, gpu_id, video_lists = arg
    # 设置线程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # 人脸检测器
    yolo = YOLO("./video_process/yolo_weights/yolov8x.pt")
    yolo.to(device)

    # 加载pose检测模型
    pose_detector = DWposeDetector()
    pose_detector = pose_detector.to(device)

    # Load Model
    body_segmentor = HumanMatting(backbone="resnet50")
    body_segmentor_weight_path = (
        "video_process/human_segment_model/pretrained/SGHM-ResNet50.pth"
    )
    body_segmentor.load_state_dict(
        copyStateDict(torch.load(body_segmentor_weight_path))
    )
    body_segmentor = body_segmentor.cuda()
    body_segmentor.eval()

    # 加载CRAFT模型
    text_detector = CRAFT()
    pretrained_path = (
        "./video_process/text_detection_model/pretrained_weight/craft_mlt_25k.pth"
    )
    print("Loading weights from checkpoint (" + pretrained_path + ")")

    text_detector.load_state_dict(copyStateDict(torch.load(pretrained_path)))
    text_detector = text_detector.cuda()
    cudnn.benchmark = False
    text_detector.eval()

    # LinkRefiner
    refine_net = None
    text_results = {}
    for video_path in video_lists:
        cap = cv2.VideoCapture(video_path)  # 替换为你的视频路径
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 获取视频的分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 获取视频的帧率
        fps = cap.get(cv2.CAP_PROP_FPS)

        if width == 1280 and height == 720 and total_frames > 60:

            start_frame = 0  # 选择起始帧
            mid_frame = total_frames // 2  # 选择中间帧
            end_frame = total_frames - 1  # 选择末尾帧
            frame_list = [start_frame, mid_frame, end_frame]

            # 根据视频帧数调整选择的帧数
            if total_frames > 32:
                start_frame2 = total_frames // 8  # 选择起始帧
                mid_frame1 = total_frames // 4  # 选择中间帧
                mid_frame2 = total_frames // 4 * 3  # 选择中间帧
                end_frame2 = total_frames // 8 * 7  # 选择末尾帧
                frame_list.append(start_frame2)
                frame_list.append(mid_frame1)
                frame_list.append(mid_frame2)
                frame_list.append(end_frame2)

            hands_result_list = []
            pose_result_list = []
            text_result_list = []
            for _ in range(len(frame_list)):
                pose_result_list.append([])
                text_result_list.append([])
                hands_result_list.append([])
            bbox = []
            bbox_area = 0.0
            detection_ratio = 0
            is_body = 0
            for i, frame_num in enumerate(frame_list):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    print("Error reading frame {}".format(frame_num))
                    continue

                # 人体检测
                body_predictions = yolo.predict(source=frame)[0]
                cls_predictions = body_predictions.boxes.cls.tolist()  # Class labels
                cls_count = Counter(cls_predictions)
                # print("face detection result: ", cls_count[0.0])

                # 满足条件，应该跳到下一个视频
                if 0.0 not in cls_count.keys():
                    break
                if cls_count[0.0] > 1:
                    break
                if 0.0 in cls_count.keys() and cls_count[0.0] == 1:
                    # 返回的是中心点坐标以及宽高
                    boxes_prediction = body_predictions.boxes.xywh.tolist()[0]
                    # 计算人体所占图像的面积比
                    bbox_area_ratio = calculate_bbox_area_ratio(
                        frame.shape[1], frame.shape[0], boxes_prediction
                    )
                    if bbox_area_ratio < 0.3:
                        break
                    # 只有大于0.3的才进行姿态检测
                    if bbox_area_ratio >= 0.3:
                        pose_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 使用frame进行姿态检测
                        pose_image = Image.fromarray(pose_frame)
                        pose_result, pose_score, is_body = pose_detector(pose_image)
                        # 当is_body为-1时代表多人存在，而当is_body为0时代表上半身不存在
                        # 但我们后续只处理单人无遮挡的视频，所以当is_body为-1/0时，
                        # 我们认为是多人存在以及上半身不完整，放弃该视频
                        if is_body <= 0:
                            break
                        if is_body > 0:
                            pose_result_list[i].append(is_body)

                            # 保存姿态检测结果
                            hand_clear, face_facing_camera = face_hands_detect(
                                i, pose_frame
                            )
                            if hand_clear:
                                hands_result_list[i].append(1)

                            body_frame = Image.fromarray(pose_frame)
                            # 人体分割，进行后续的判断是否存在文本与人体重叠
                            pred_alpha, pred_mask = single_inference(
                                body_segmentor, body_frame, device=device
                            )

                            # 对分割结果进行二值化处理
                            body_mask = (pred_alpha * 255).astype("uint8")
                            # print("body segmentation result:", body_mask.shape, np.unique(body_mask))
                            body_mask[body_mask >= 128] = 255
                            body_mask[body_mask < 128] = 0
                            body_mask = body_mask[:, :, np.newaxis]

                            rand_num = random.randint(0, 10000)

                            text_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # 使用frame进行文本检测
                            bboxes, polys, text_score_map = test_net(
                                text_detector,
                                text_frame,
                                0.7,
                                0.4,
                                0.4,
                                True,
                                False,
                                refine_net,
                            )

                            # 计算检测结果的面积
                            detection_area = 0
                            detection_area, intersection_maps = calculate_overlap(
                                polys, body_mask
                            )
                            detection_ratio += detection_area
                            text_result_list[i].append(detection_area)

                            # 由于union_bbox的计算需要bbox的xyxy坐标，所以这里需要转换
                            boxes_xyxy = body_predictions.boxes.xyxy.tolist()[0]
                            if i == 0:
                                bbox = boxes_xyxy
                            else:
                                union_bbox = get_union_bbox(bbox, boxes_xyxy)
                                bbox = union_bbox

                            # 保存文本检测结果和得分图
                            if False and detection_area < 0.10:
                                # mask_file = f"Temp_dir/random_images_detect_3_14/{rand_num}_text.jpg"
                                # cv2.imwrite(mask_file, text_score_map)

                                save_img_root = "tmp/bbox_7_04_correct"

                                bbox_frame = draw_bbox(frame, bbox)
                                cv2.imwrite(
                                    f"{save_img_root}/{rand_num}_{detection_area}_bbox.jpg",
                                    bbox_frame,
                                )

                                detection_area = round(detection_area, 3)

                                body_predictions[0].save(
                                    filename=f"{save_img_root}/{rand_num}_{detection_area}_yolo.jpg"
                                )
                                # text_mask = intersection_maps[0]
                                # if len(text_mask.shape) != 3:
                                #     text_mask = text_mask[:, :, np.newaxis]

                                # for i, inter_mask in enumerate(intersection_maps[1:]):
                                #     if len(inter_mask.shape) != 3:
                                #         inter_mask = inter_mask[:, :, np.newaxis]
                                #     text_mask += inter_mask

                                # cv2.imwrite(
                                #     f"{save_img_root}/{rand_num}_{detection_area}_text.jpg",
                                #     text_mask,
                                # )
                                # pose_image.save(
                                #     f"Temp_dir/random_images_detect_3_15_2/{rand_num}_{detection_area}_origin.jpg"
                                # )

                    bbox_area += bbox_area_ratio

            bbox_ratio = bbox_area / len(frame_list)
            detection_ratio /= len(frame_list)
            result_dict = {
                "pose_score": pose_result_list,
                "hand_score": hands_result_list,
                "text_score": round(detection_ratio, 3),
                "area_score": round(bbox_ratio, 3),
                "video_length": total_frames,
                "fps": fps,
                "resolution": [width, height],
                "bbox": bbox,
            }
            text_results[video_path] = result_dict
            print("{}: {}".format(video_path, result_dict))

            cap.release()

    return text_results


def mp_text_detect_process(
    file_json,
    save_path,
    threads=2,
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    date=0,
):

    mp.set_start_method("spawn", force=True)

    with open(
        file_json,
        "r",
    ) as f:
        data = json.load(f)

    # 提取视频路径和得分
    video_paths = []
    for i, (video_path, score) in enumerate(data.items()):
        video_paths.append(os.path.join(video_path))

    print("All videos loaded. {} videos in total.".format(len(video_paths)))

    # 单线程测试
    # detect_one_person_wo_occlusion((0, 0, video_paths[:1000]))

    video_list = []
    num_threads = threads
    batch_size = len(video_paths) // num_threads
    for i in range(num_threads):
        if i == num_threads - 1:
            video_list.append(video_paths[i * batch_size :])
        else:
            video_list.append(video_paths[i * batch_size : (i + 1) * batch_size])

    with mp.Pool(num_threads) as pool:
        results = pool.map(
            detect_one_person_wo_occlusion,
            zip(range(num_threads), gpu_ids[:num_threads], video_list),
        )

    results_dict = {}
    for p in results:
        results_dict.update(p)

    print("All threads completed.")

    save_json_path = save_path + "/text_detection_{}.json".format(date)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")
    return save_json_path


if __name__ == "__main__":
    json_file = "Youtube_videos/batch_01/selected_videos_by_optical_flow_0704_v1.json"
    save_path = "Select_json_for_a2p_0704_v1"

    mp_text_detect_process(json_file, save_path)
