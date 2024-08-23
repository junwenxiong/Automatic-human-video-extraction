import torch
import os
import threading
import cv2
import os
import json
import matplotlib.pyplot as plt
import time
import json
import shutil
import random
import numpy as np
from collections import defaultdict
import pandas
from datetime import datetime
from ultralytics import YOLO

import torch.multiprocessing as mp
from collections import Counter


class VideoProcessingThread(threading.Thread):
    def __init__(self, video_paths, motion_amplitudes, lock_pool):
        super(VideoProcessingThread, self).__init__()
        self.video_paths = video_paths
        self.motion_amplitudes = motion_amplitudes
        self.lock_pool = lock_pool

    def run(self):
        for video_path in self.video_paths:
            self.process_video(video_path)

    def calculate_average_flow(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Initialize variables
        frame_count = 0
        total_flow_magnitude = 0.0

        # Read the first frame
        ret, prev_frame = cap.read()

        # Iterate over the video frames
        while ret:
            # Read the current frame
            ret, curr_frame = cap.read()

            if ret:
                # Convert frames to grayscale
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # Compute the magnitude of the optical flow
                flow_magnitude = cv2.norm(flow, cv2.NORM_L2)

                # Accumulate the total flow magnitude
                total_flow_magnitude += flow_magnitude

                # Update the previous frame
                prev_frame = curr_frame

                # Increment the frame count
                frame_count += 1

        # Release the video capture object
        cap.release()

        # Calculate the average flow magnitude
        average_flow_magnitude = total_flow_magnitude / frame_count

        return average_flow_magnitude

    def calculate_average_flow_speed_up_2fps(self, video_path):
        # Extract dense optical flow at 2fps using the Farneback algorithm
        cap = cv2.VideoCapture(video_path)
        # farneback = cv2.FarnebackOpticalFlow_create()

        frame_count = 1
        total_flow_magnitude = 0.0
        optical_flow_fps = 5

        prev_gray = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform optical flow calculation at the desired frame rate
            if (
                cap.get(cv2.CAP_PROP_POS_FRAMES)
                % (cap.get(cv2.CAP_PROP_FPS) / optical_flow_fps)
                == 0
            ):

                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.resize(curr_gray, (0, 0), fx=0.5, fy=0.5)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        curr_gray,
                        flow=None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )
                    # TODO 为什么要缩小光流图维度
                    # Perform spatial downsampling
                    min_side = min(flow.shape[:2])
                    scale = 16 / min_side
                    flow_downscaled = cv2.resize(flow, None, fx=scale, fy=scale)

                    # Compute the magnitude of the optical flow
                    flow_magnitude = cv2.norm(flow_downscaled, cv2.NORM_L2)

                    # Accumulate the total flow magnitude
                    total_flow_magnitude += flow_magnitude

                    # Increment the frame count
                    frame_count += 1

                prev_gray = curr_gray

        average_flow_magnitude = total_flow_magnitude / frame_count

        return average_flow_magnitude

    def calculate_average_flow_speed_up(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Initialize variables
        frame_count = 1
        total_flow_magnitude = 0.0

        # Read the first frame
        ret, prev_frame = cap.read()

        # Iterate over the video frames
        while ret:
            # Read the current frame
            ret, curr_frame = cap.read()

            if ret:
                # Convert frames to grayscale
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                # 缩小图片维度
                prev_gray = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
                curr_gray = cv2.resize(curr_gray, (0, 0), fx=0.5, fy=0.5)

                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    curr_gray,
                    flow=None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                # TODO 为什么要缩小光流图维度
                # Perform spatial downsampling
                min_side = min(flow.shape[:2])
                scale = 16 / min_side
                flow_downscaled = cv2.resize(flow, None, fx=scale, fy=scale)

                # Compute the magnitude of the optical flow
                flow_magnitude = cv2.norm(flow_downscaled, cv2.NORM_L2)

                # Accumulate the total flow magnitude
                total_flow_magnitude += flow_magnitude

                # Update the previous frame
                prev_frame = curr_frame

                # Increment the frame count
                frame_count += 1

        # Release the video capture object
        cap.release()

        # Calculate the average flow magnitude
        average_flow_magnitude = total_flow_magnitude / frame_count

        return average_flow_magnitude

    def process_video(self, video_path):
        try:
            # Calculate the average flow magnitude for the video
            # average_flow = self.calculate_average_flow_speed_up_2fps(video_path)
            average_flow = self.calculate_average_flow_speed_up(video_path)
            # Round the average flow magnitude to the nearest integer
            motion_amplitude = round(average_flow, 2)
            print(f"Video: {video_path}, Motion Amplitude: {motion_amplitude}")
            # Update the motion amplitude count in the dictionary
            # with self.lock_pool.get_lock(motion_amplitude):
            self.motion_amplitudes[video_path] = motion_amplitude

        except Exception as e:
            print(f"Video: {video_path}, An error occurred: {e}")


class LockPool:
    def __init__(self):
        self.locks = {}

    def get_lock(self, index):
        if index not in self.locks:
            with threading.Lock():
                if index not in self.locks:
                    self.locks[index] = threading.Lock()
        return self.locks[index]


def save_motion_amplitudes_to_json(motion_amplitudes, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(motion_amplitudes, f, ensure_ascii=False)

    print(f"Motion amplitudes saved to {output_file}")


def compute_motion_amplitudes(input_folder, save_path):
    # Initialize a dictionary to store motion amplitudes and their counts
    motion_amplitudes = {}

    # Get the list of video paths in the input folder
    video_paths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)

    # video_paths = video_paths[:64]

    # Determine the number of threads to use
    num_threads = 64  # Adjust as desired

    start_time = time.time()
    # Create locks for each motion amplitude key
    lock_pool = LockPool()
    # Create and start video processing threads
    threads = []
    batch_size = len(video_paths) // num_threads
    for i in range(num_threads):
        start_index = i * batch_size
        end_index = (
            start_index + batch_size if i < num_threads - 1 else len(video_paths)
        )
        thread = VideoProcessingThread(
            video_paths[start_index:end_index], motion_amplitudes, lock_pool
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("time cost: ", time.time() - start_time)

    save_json_path = save_path + "/motion_amplitudes.json"
    save_motion_amplitudes_to_json(motion_amplitudes, save_json_path)

    return save_json_path


def is_camera_stable(video_path, threshold_area=5000):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 创建背景减法器
    backSub = cv2.createBackgroundSubtractorMOG2()

    while True:
        # 读取一帧视频
        ret, frame = cap.read()
        if not ret:
            break

        # 对当前帧应用背景减法
        fgMask = backSub.apply(frame)

        # 对前景掩码进行形态学操作
        kernel = np.ones((5, 5), np.uint8)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        # 查找前景区域的轮廓
        contours, _ = cv2.findContours(
            fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # 计算轮廓的外接矩形框
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # 判断矩形框的面积是否大于阈值
            if area > threshold_area:
                # 存在运动目标，返回False
                return False

    # 相机静止，返回True
    return True


def detect_camera_movement():
    static_back = None
    # List when any moving object appear
    motion_list = [None, None]
    # Time of movement
    time = []
    cnt = 0

    # Capturing video
    video = cv2.VideoCapture(0)

    # Infinite while loop to treat stack of image as video
    while True:
        # Reading frame(image) from video
        check, frame = video.read()

        motion = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if static_back is None or cnt == 100:
            cnt = -1
            static_back = gray
            continue

        cnt = cnt + 1

        diff_frame = cv2.absdiff(static_back, gray)

        thresh_frame = cv2.threshold(diff_frame, 70, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=20)

        cnts, _ = cv2.findContours(
            thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rects = []
        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue
            motion = 1

            (x, y, w, h) = cv2.boundingRect(contour)
            rects.append([x, y, w, h])
            rects.append([x, y, w, h])

        rects, _ = cv2.groupRectangles(rects, 1, 0.1)
        for contour2 in rects:
            (x, y, w, h) = contour2
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # Appending status of motion
        motion_list.append(motion)

        motion_list = motion_list[-2:]

        # Appending Start time of motion
        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time.append(datetime.now())

        # Appending End time of motion
        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time.append(datetime.now())


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


def set_bbox_to_zero(frame, bbox):
    """
    将给定帧中的 bounding box 区域设为 0 值

    参数:
    frame (numpy.ndarray): 输入帧
    bbox (tuple): bounding box 的坐标 (x1, y1, x2, y2)

    返回:
    numpy.ndarray: 处理后的帧
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # 确保 bounding box 的坐标在帧范围内
    x1 = max(0, x1 - 10)
    y1 = max(0, y1 - 10)
    x2 = min(frame.shape[1], x2 + 10)
    y2 = min(frame.shape[0], y2 + 10)

    # 将 bounding box 区域设为 0 值
    frame[y1:y2, x1:x2] = 0

    return frame


def detect_camera_rotating(arg):
    thread_id, gpu_id, video_lists = arg

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # 人脸检测器
    # yolo = YOLO("./video_process/yolo_weights/yolov8x.pt")
    # yolo.to(device)

    text_results = {}
    for video_path, bbox in video_lists:
        cap = cv2.VideoCapture(video_path)  # 替换为你的视频路径
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0  # 选择起始帧
        frame_list = []

        # 根据视频帧数调整选择的帧数
        if total_frames > 32:
            for frame in range(start_frame, total_frames, 5):
                frame_list.append(frame)
        else:
            mid_frame = total_frames // 2  # 选择中间帧
            end_frame = total_frames - 1  # 选择末尾帧
            frame_list = [start_frame, mid_frame, end_frame]

        pose_result_list = []
        text_result_list = []
        for _ in range(len(frame_list)):
            pose_result_list.append([])
            text_result_list.append([])

        # detection_ratio = 0
        # is_body = 0
        # bbox = None
        # for i, frame_num in enumerate(frame_list):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        #     ret, frame = cap.read()

        #     if not ret:
        #         print("Error reading frame {}".format(frame_num))
        #         continue

            # # 人体检测
            # body_predictions = yolo.predict(source=frame)[0]

            # # body_predictions.save(filename=f"{save_dir}/{rand_num}_result.jpg")

            # cls_predictions = body_predictions.boxes.cls.tolist()  # Class labels
            # cls_count = Counter(cls_predictions)

            # # 满足条件，应该跳到下一个视频
            # if 0.0 not in cls_count.keys():
            #     break
            # if cls_count[0.0] > 1:
            #     break
            # if 0.0 in cls_count.keys() and cls_count[0.0] == 1:
            #     boxes_prediction = body_predictions.boxes.xyxy.tolist()[0]

            #     if i == 0:
            #         bbox = boxes_prediction
            #     else:
            #         union_bbox = get_union_bbox(bbox, boxes_prediction)
            #         bbox = union_bbox

        # bbox_frame = draw_bbox(frame, bbox)
        # rand_num = random.randint(0, 1000)

        # TODO 加入bbox信息
        frame_count = 1
        total_flow_magnitude = 0.0
        optical_flow_fps = 10

        prev_gray = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform optical flow calculation at the desired frame rate
            if (
                cap.get(cv2.CAP_PROP_POS_FRAMES)
                % (cap.get(cv2.CAP_PROP_FPS) / optical_flow_fps)
                == 0
            ):

                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                curr_gray = set_bbox_to_zero(curr_gray, bbox)
                # cv2.imwrite(f"{save_dir}/{rand_num}_bbox.jpg", curr_gray)
                curr_gray = cv2.resize(curr_gray, (0, 0), fx=0.5, fy=0.5)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        curr_gray,
                        flow=None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )
                    # Perform spatial downsampling
                    min_side = min(flow.shape[:2])
                    scale = 16 / min_side
                    flow_downscaled = cv2.resize(flow, None, fx=scale, fy=scale)

                    # Compute the magnitude of the optical flow
                    flow_magnitude = cv2.norm(flow_downscaled, cv2.NORM_L2)

                    # Accumulate the total flow magnitude
                    total_flow_magnitude += flow_magnitude

                    # Increment the frame count
                    frame_count += 1

                prev_gray = curr_gray

        average_flow_magnitude = total_flow_magnitude / frame_count

        print(video_path, average_flow_magnitude)
        text_results[video_path] = {
            "camera rotating": average_flow_magnitude,
        }

    return text_results


def mp_camera_rotating_detect_process(
    file_json, bbox_json, save_path, threads=2, gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], save_flag=0,
    date=0,
):

    mp.set_start_method("spawn", force=True)

    with open(
        file_json,
        "r",
    ) as f:
        data = json.load(f)

    with open(bbox_json, "r") as f:
        bbox_data = json.load(f)

    # 提取视频路径和得分
    video_paths = []
    video_bbox = []
    for i, (video_path, score) in enumerate(data.items()):
        if video_path not in bbox_data.keys():
            continue
        video_paths.append([os.path.join(video_path), bbox_data[video_path]['bbox']])

    print("All videos loaded. {} videos in total.".format(len(video_paths)))

    video_list = []
    bbox_list = []
    num_threads = threads
    batch_size = len(video_paths) // num_threads
    for i in range(num_threads):
        if i == num_threads - 1:
            video_list.append(video_paths[i * batch_size :])
        else:
            video_list.append(video_paths[i * batch_size : (i + 1) * batch_size])

    with mp.Pool(num_threads) as pool:
        results = pool.map(
            detect_camera_rotating,
            zip(range(num_threads), gpu_ids[:num_threads], video_list),
        )

    results_dict = {}
    for p in results:
        results_dict.update(p)

    print("All threads completed.")

    if save_flag == 0:
        save_json_path = save_path + f"/camera_rotating_detection_{date}.json"
    else:
        save_json_path = save_path + f"/camera_rotating_detection_{save_flag}.json"
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")
    return save_json_path


if __name__ == "__main__":

    video_root = "Download/douyin/cascade_cut"
    output_file = "motion_amplitudes_cascade_cut.json"
    # compute_motion_amplitudes(video_root, output_file)

    file_json = "TED/TEDxTalks/selected_videos_by_av_consistency.json"
    with open(
        file_json,
        "r",
    ) as f:
        data = json.load(f)

    save_dir = "Temp_dir/motion_detect"
    video_input_path = "test"
    # 提取视频路径和得分
    video_paths = []
    for i, (video_name) in enumerate(os.listdir(video_input_path)):
        video_paths.append(os.path.join(video_input_path, video_name))

    arg = (0, 0, video_paths)
    avg_flow_mag = detect_camera_rotating(arg)
    print(avg_flow_mag)
