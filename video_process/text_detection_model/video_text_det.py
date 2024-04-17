"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import numpy as np
import craft_utils
import imgproc
import json
from craft import CRAFT
from collections import OrderedDict
import torch.multiprocessing as mp


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
parser.add_argument("--low_text", default=0.4, type=float, help="text low-bound score")
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


def test_net(
    net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None
):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image,
        args.canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=args.mag_ratio,
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

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def detect_video(arg):
    thread_id, gpu_id, video_lists = arg
    # 设置线程使用的GPU
    torch.cuda.set_device(gpu_id)

    # 加载CRAFT模型
    net = CRAFT()
    print("Loading weights from checkpoint (" + args.trained_model + ")")

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        net = net.cuda()
        # net = net.to("cuda:{}".format(gpu_id))
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet

        refine_net = RefineNet()
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        refine_net.eval()
        args.poly = True

    results = {}
    for video_path in video_lists:
        # print(video_path)
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
                args.text_threshold,
                args.link_threshold,
                args.low_text,
                args.cuda,
                args.poly,
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
            # mask_file = result_folder + "/res_" + filename + "_mask.jpg"
            # cv2.imwrite(mask_file, score_text)

            # file_utils.saveResult(filename, frame[:, :, ::-1], polys, dirname=result_folder)

        detection_ratio /= len(frame_list)

        results[video_path] = round(detection_ratio, 3)
        # print("{}: {:.2%}".format(video_path, detection_ratio))

        cap.release()

    return results


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    save_json_path = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Download/douyin/text_detection_7_frames.json"
    with open(
        "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Download/douyin/filter_videos_by_motion_score.json",
        "r",
    ) as f:
        data = json.load(f)

    video_root = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload"
    # 提取视频路径和得分
    video_paths = []
    for i, (video_path, score) in enumerate(data.items()):
        video_paths.append(os.path.join(video_root, video_path))
        # if i == 1000:
        #     break

    video_list = []
    num_threads = 7
    batch_size = len(video_paths) // num_threads
    for i in range(num_threads):
        if i == num_threads - 1:
            video_list.append(video_paths[i * batch_size :])
        else:
            video_list.append(video_paths[i * batch_size : (i + 1) * batch_size])

    m_queues = mp.Queue()
    threads = []
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # gpu_ids = [4, 5, 6, 7]

    with mp.Pool(num_threads) as pool:
        results = pool.map(
            detect_video, zip(range(num_threads), gpu_ids[:num_threads], video_list)
        )

    # print(results)

    results_dict = {}
    for p in results:
        results_dict.update(p)

    print("All threads completed.")

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")
