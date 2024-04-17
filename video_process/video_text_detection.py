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
from .CRAFT_pytorch import craft_utils
from .CRAFT_pytorch import imgproc
import json
from .CRAFT_pytorch.craft import CRAFT
from collections import OrderedDict
import torch.multiprocessing as mp

from .dwpose import DWposeDetector
from .humanSeg.model.model import HumanMatting
from .humanSeg.inference import single_inference
from PIL import Image
from collections import OrderedDict
from ultralytics import YOLO
from collections import Counter

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
    body_segmentor_weight_path = "video_process/humanSeg/pretrained/SGHM-ResNet50.pth"
    body_segmentor.load_state_dict(
        copyStateDict(torch.load(body_segmentor_weight_path))
    )
    body_segmentor = body_segmentor.cuda()
    body_segmentor.eval()

    # 加载CRAFT模型
    text_detector = CRAFT()
    pretrained_path = (
        "./video_process/CRAFT_pytorch/pretrained_weight/craft_mlt_25k.pth"
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

        pose_result_list = []
        text_result_list = []
        for _ in range(len(frame_list)):
            pose_result_list.append([])
            text_result_list.append([])

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
                if bbox_area_ratio < 0.2:
                    break
                # 只有大于0.3的才进行姿态检测
                if bbox_area_ratio >= 0.2:
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
                        # print("pose detection result:", is_body, i, len(pose_result_list), len(frame_list))
                        pose_result_list[i].append(is_body)

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

                        # 保存文本检测结果和得分图
                        if False and detection_area > 0.10:
                            # mask_file = f"Temp_dir/random_images_detect_3_14/{rand_num}_text.jpg"
                            # cv2.imwrite(mask_file, text_score_map)

                            detection_area = round(detection_area, 3)

                            body_predictions[0].save(
                                filename=f"Temp_dir/random_images_detect_3_15_4/{rand_num}_{detection_area}_yolo.jpg"
                            )
                            # cv2.imwrite(
                            #     f"Temp_dir/random_images_detect_3_15_2/{rand_num}_{detection_area}_body.jpg",
                            #     body_mask,
                            # )

                            text_mask = intersection_maps[0]
                            if len(text_mask.shape) != 3:
                                text_mask = text_mask[:, :, np.newaxis]

                            for i, inter_mask in enumerate(intersection_maps[1:]):
                                if len(inter_mask.shape) != 3:
                                    inter_mask = inter_mask[:, :, np.newaxis]
                                text_mask += inter_mask

                            cv2.imwrite(
                                f"Temp_dir/random_images_detect_3_15_4/{rand_num}_{detection_area}_text.jpg",
                                text_mask,
                            )
                            # pose_image.save(
                            #     f"Temp_dir/random_images_detect_3_15_2/{rand_num}_{detection_area}_origin.jpg"
                            # )

        detection_ratio /= len(frame_list)
        result_dict = {
            "pose_result": pose_result_list,
            "text_result": round(detection_ratio, 3),
            "text_res_list": text_result_list,
        }
        text_results[video_path] = result_dict
        print("{}: {}".format(video_path, result_dict))

        cap.release()

    return text_results


def mp_text_detect_process(
    file_json, save_path, threads=2, gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7]
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

    save_json_path = save_path + "/text_detection.json"
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")
    return save_json_path


if __name__ == "__main__":
    json_file = "Download_0308/douyin/filter_info/filter_videos_by_motion_score_30.json"
    save_path = "Download_0308/douyin"

    json_file = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Download/douyin/selected_videos_by_optical_flow.json"
    save_path = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir"

    mp_text_detect_process(json_file, save_path)
