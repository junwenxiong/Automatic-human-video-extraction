# -*- coding: utf-8 -*-
import os
import random
import json
import torch
import cv2
import torch.multiprocessing as mp
from .dwpose import DWposeDetector
from PIL import Image

detect_results = {}
tmp_results = {}


def detect_video(arg):
    thread_id, gpu_id, video_lists = arg
    # 设置线程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    detector = DWposeDetector()
    detector = detector.to(device)

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

        result_list = []
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
            image = Image.fromarray(frame)
            result, score, is_body = detector(image)
            result_list.append(is_body)

            # if result is not None:
            #     rand_num = random.randint(0, 1000)
            #     save_path = os.path.join("/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir", f"{rand_num}.png")
            #     print("Save result to: ", save_path)
            #     result.save(save_path)

        results[video_path] = result_list
        print("{}: {}".format(video_path, result_list))

        cap.release()

    return results


def mp_pose_detection_process(
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

        if i == 1000:
            break

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
            detect_video, zip(range(num_threads), gpu_ids[:num_threads], video_list)
        )

    results_dict = {}
    for p in results:
        results_dict.update(p)

    print("All threads completed.")

    save_json_path = save_path + "/pose_detection.json"
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")

    return save_json_path


if __name__ == "__main__":
    input_json = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Download/douyin/selected_videos_by_optical_flow.json"
    output_dir = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir"
    mp_pose_detection_process(input_json, output_dir, threads=1)
