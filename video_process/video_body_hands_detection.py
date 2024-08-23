# -*- coding: utf-8 -*-
import random
import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import mediapipe
import torch.nn as nn
import numpy as np
from video_process.text_detection_model import craft_utils
from video_process.text_detection_model import imgproc
import json
from video_process.text_detection_model.craft import CRAFT
from collections import OrderedDict
import torch.multiprocessing as mp

from video_process.pose_detection_model import DWposeDetector
from video_process.human_segment_model.model.model import HumanMatting
from video_process.human_segment_model.inference import single_inference
from PIL import Image
from collections import OrderedDict
from ultralytics import YOLO
from collections import Counter

from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed

detect_results = {}
tmp_results = {}


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


def prepare_args():
    parser = argparse.ArgumentParser(description="CRAFT Text Detection")
    parser.add_argument(
        "--trained_model",
        default="weights/craft_mlt_25k.pth",
        type=str,
        help="pretrained model",
    )
    parser.add_argument(
        "--text_threshold", default=0.7, type=float, help="text confidence threshold"
    )
    parser.add_argument(
        "--low_text", default=0.4, type=float, help="text low-bound score"
    )
    parser.add_argument(
        "--link_threshold", default=0.4, type=float, help="link confidence threshold"
    )
    parser.add_argument(
        "--cuda", default=True, type=str2bool, help="Use cuda for inference"
    )
    parser.add_argument(
        "--canvas_size", default=1280, type=int, help="image size for inference"
    )
    parser.add_argument(
        "--mag_ratio", default=1.5, type=float, help="image magnification ratio"
    )
    parser.add_argument(
        "--poly", default=False, action="store_true", help="enable polygon type"
    )
    parser.add_argument(
        "--show_time", default=False, action="store_true", help="show processing time"
    )
    parser.add_argument(
        "--test_folder", default="/data/", type=str, help="folder path to input images"
    )
    parser.add_argument(
        "--refine", default=False, action="store_true", help="enable link refiner"
    )
    parser.add_argument(
        "--refiner_model",
        default="weights/craft_refiner_CTW1500.pth",
        type=str,
        help="pretrained refiner model",
    )

    args = parser.parse_args()
    return args


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
        if total_frames > 16:
            start_frame2 = total_frames // 8  # 选择起始帧
            mid_frame1 = total_frames // 4  # 选择中间帧
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


# 检测手部
def detect_hands(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    return results.multi_hand_landmarks


# 检测面部
def detect_face(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    return results.multi_face_landmarks


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


def detect_one_person_wo_occlusion(arg):
    thread_id, gpu_id, video_lists = arg
    # 设置线程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # 人脸检测器
    # yolo = YOLO("./video_process/yolo_weights/yolov8x.pt")
    # yolo.to(device)

    # 加载pose检测模型
    pose_detector = DWposeDetector()
    pose_detector = pose_detector.to(device)

    # LinkRefiner
    refine_net = None
    text_results = {}
    for video_path in video_lists:

        with VideoFileClip(video_path) as video:
            video_duration = video.duration
            fps = video.fps

            # 获取视频文件名（不包含扩展名）
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            start_time = 0
            end_time = video_duration

            # 提取并检测视频帧
            for time in [0, video_duration]:
                frame = video.get_frame(time)
                pose_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 使用frame进行姿态检测
                pose_image = Image.fromarray(pose_frame)
                pose_result, pose_score, is_body = pose_detector(pose_image)

                if is_body <= 0:
                    if time == 0:
                        start_time += 1 / fps
                    else:
                        end_time -= 1 / fps
                if is_body > 0:
                    if time == 0:
                        start_time = 0
                    else:
                        end_time = video_duration

            print(
                "video_path: ",
                video_path,
                "start_time: ",
                start_time - 0,
                "end_time: ",
                end_time - video_duration,
            )

        result_dict = {
            "start_time": start_time - 0,
            "end_time": end_time - video_duration,
        }
        text_results[video_path] = result_dict

        # 裁剪视频
        # cleaned_video = video.subclip(start_time, end_time)

        # 检查输出目录是否存在，不存在则创建
        # if not os.path.exists(output_path):
        # os.makedirs(output_path)

        # 保存裁剪后的视频
        # output_file = os.path.join(output_path, f"cleaned_{video_name}.mp4")
        # cleaned_video.write_videofile(
        # output_file, codec="libx264", audio_codec="aac"
        # )

        # cap = cv2.VideoCapture(video_path)  # 替换为你的视频路径
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # # 用于保存检测结果
        # rand_num = random.randint(0, 100000)

        # start_frame = 0  # 选择起始帧
        # mid_frame = total_frames // 2  # 选择中间帧
        # end_frame = total_frames - 1  # 选择末尾帧
        # frame_list = [start_frame, mid_frame, end_frame]

        # # 根据视频帧数调整选择的帧数
        # if total_frames > 32:
        #     start_frame2 = total_frames // 8  # 选择起始帧
        #     start_frame3 = total_frames // 8 * 3  # 选择起始帧
        #     mid_frame1 = total_frames // 4  # 选择中间帧
        #     start_frame4 = total_frames // 8 * 5  # 选择起始帧
        #     mid_frame2 = total_frames // 4 * 3  # 选择中间帧
        #     end_frame2 = total_frames // 8 * 7  # 选择末尾帧

        #     frame_list.append(start_frame2)
        #     frame_list.append(start_frame3)
        #     frame_list.append(mid_frame1)
        #     frame_list.append(start_frame4)
        #     frame_list.append(mid_frame2)
        #     frame_list.append(end_frame2)

        # pose_result_list = []
        # text_result_list = []
        # for _ in range(len(frame_list)):
        #     pose_result_list.append([])
        #     text_result_list.append([])

        # detection_ratio = 0
        # is_body = 0
        # for i, frame_num in enumerate(frame_list):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        #     ret, frame = cap.read()

        #     if not ret:
        #         print("Error reading frame {}".format(frame_num))
        #         continue

        #     # 人体检测
        #     body_predictions = yolo.predict(source=frame)[0]
        #     cls_predictions = body_predictions.boxes.cls.tolist()  # Class labels
        #     cls_count = Counter(cls_predictions)
        #     # print("face detection result: ", cls_count[0.0])

        #     # 满足条件，应该跳到下一个视频
        #     if 0.0 not in cls_count.keys():
        #         break
        #     if cls_count[0.0] > 1:
        #         break
        #     if 0.0 in cls_count.keys() and cls_count[0.0] == 1:
        #         # 返回的是中心点坐标以及宽高
        #         boxes_prediction = body_predictions.boxes.xywh.tolist()[0]
        #         # 计算人体所占图像的面积比
        #         bbox_area_ratio = calculate_bbox_area_ratio(
        #             frame.shape[1], frame.shape[0], boxes_prediction
        #         )
        #         if bbox_area_ratio < 0.2:
        #             pose_result_list[i].append(2)
        #             break
        #         # 只有大于0.3的才进行姿态检测
        #         if bbox_area_ratio >= 0.2:
        #             pose_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #             # 使用frame进行姿态检测
        #             pose_image = Image.fromarray(pose_frame)
        #             pose_result, pose_score, is_body = pose_detector(pose_image)

        #             if is_body <= 0:
        #                 pose_result_list[i].append(3)
        #                 break
        #             if is_body > 0:

        #                 # 保存姿态检测结果
        #                 hand_clear, face_facing_camera = face_hands_detect(
        #                     i, pose_frame
        #                 )

        #                 if hand_clear:
        #                     pose_result_list[i].append(1)

        #                 # 保存文本检测结果和得分图
        #                 if True and hand_clear:
        #                     detection_area = i
        #                     save_path = "tmp/images_detect_6_29"
        #                     os.makedirs(save_path, exist_ok=True)

        #                     detection_area = round(detection_area, 3)

        #                     body_predictions[0].save(
        #                         filename=f"{save_path}/{rand_num}_{detection_area}_yolo.jpg"
        #                     )

        #                     # pose_result.save(
        #                     #     f"{save_path}/{rand_num}_{detection_area}_pose.jpg"
        #                     # )

        # detection_ratio /= len(frame_list)
        # result_dict = {
        #     "pose_result": pose_result_list,
        #     "bbox_area_ratio": bbox_area_ratio,
        # }
        # text_results[video_path] = result_dict
        # print("{}: {}".format(video_path, result_dict))

        # cap.release()

    return text_results


def mp_body_hands_detect_process(
    file_json, save_path, threads=2, gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], video_flag="TED"
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

    save_json_path = save_path + f"/{video_flag}_refined_body_hands_det_0713.json"
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")
    return save_json_path


if __name__ == "__main__":

    json_file = "Select_json_for_a2p_0704_v1/TED_videos_selected_videos_0704_v1_cutting_filtered.json"
    save_path = "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/tmp"

    mp_body_hands_detect_process(json_file, save_path)
