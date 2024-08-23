import os
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import threading
import shutil
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed


class VideoProcessingThread(threading.Thread):
    def __init__(self, video_paths):
        super(VideoProcessingThread, self).__init__()
        self.video_paths = video_paths

    def run(self):
        for video_path in self.video_paths:
            video_save_path = video_path.replace("post", "cascade_cut")
            video_save_path = video_save_path.replace(".mp4", "")

            os.makedirs(video_save_path, exist_ok=True)

            if len(os.listdir(video_save_path)) != 0:
                print(f"{video_save_path} exists")
            else:
                cascade_cut(video_path, video_save_path)


def detect_cut(video_path, video_save_path, threshold=27.0):
    """按镜头切割视频，同时保存视频片段

    Args:
        video_path (_type_): _description_
        threshold (float, optional): _description_. Defaults to 27.0.
    """

    video = open_video(video_path)
    if video.frame_size == (640, 360):
        print("video is too small (640, 360), skip")
        return

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    if len(scene_list) != 0:
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_file_template=video_save_path + "/" + "Scene-$SCENE_NUMBER.mp4",
            show_progress=False,
        )
        print("scene detected")
    else:
        shutil.copy(video_path, video_save_path + "/" + "Scene-0.mp4")
        print("no scene detected")
    print(video_save_path, scene_list)


# Apply a cascade of cut detectors with different frame rates and thresholds
def cascade_cut(video_path, video_save_path):
    # Create a video manager and scene manager
    video = open_video(video_path)
    if video.frame_size == (640, 360):
        print("video is too small (640, 360), skip")
        return

    scene_manager = SceneManager()

    # Add the content detectors to the scene manager
    # Detector 1: Sudden changes with a high frame rate and threshold
    scene_manager.add_detector(ContentDetector(threshold=30))

    # Detector 2: Slow changes with a lower frame rate and threshold
    scene_manager.add_detector(ContentDetector(threshold=20, min_scene_len=15))

    # Detector 3: Slow changes with an even lower frame rate and threshold
    scene_manager.add_detector(ContentDetector(threshold=15, min_scene_len=15))

    # Perform scene detection
    scene_manager.detect_scenes(video, show_progress=False)
    # Retrieve the detected scenes
    scene_list = scene_manager.get_scene_list()

    if len(scene_list) != 0:
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_file_template=video_save_path + "/" + "Scene-$SCENE_NUMBER.mp4",
            show_progress=False,
        )
        print(f"{video_save_path}, cutting video")
    else:
        shutil.copy(video_path, video_save_path + "/" + "Scene-0.mp4")
        print(f"{video_save_path}, no scene detected")
    # print(video_save_path, scene_list)


def cutting_videos_in_directory(directory, num_threads=32):
    """
    对原视频进行切分，按照镜头切分
    缺点：
        1.  切分后的视频前后帧可能不含有人体，是镜头转换的场景
        2. 不能控制切分后的视频长度
    """
    video_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)

    # 计算每个线程处理的视频数量
    videos_per_thread = len(video_paths) // num_threads
    remaining_videos = len(video_paths) % num_threads

    threads = []
    start_index = 0

    # 创建并启动线程
    for i in range(num_threads):
        end_index = start_index + videos_per_thread

        # 将多余的视频分配给最后一个线程
        if i == num_threads - 1:
            end_index += remaining_videos

        thread = VideoProcessingThread(video_paths[start_index:end_index])
        thread.start()
        threads.append(thread)

        start_index = end_index

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def load_video_info(json_path):
    with open(json_path, "r") as f:
        video_info = json.load(f)
    return video_info


def save_slice_info(slice_info, output_json_path):
    with open(output_json_path, "w") as f:
        json.dump(slice_info, f, indent=4)


def slice_video(video_path, slice_info):
    """
    按10秒一个片段切分视频
    """
    with VideoFileClip(video_path) as video:
        video_duration = video.duration

        output_path = os.path.dirname(video_path)
        dir_name = os.path.basename(output_path)
        # 获取视频文件名（不包含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if 2 <= video_duration <= 12:
            # 如果视频长度在2~12秒之间，不进行处理
            print(
                f"Video {video_path} duration is between 2 and 12 seconds, no processing needed."
            )
        elif video_duration > 12:
            segment_index = 1

            # 保存前10秒的片段
            start_time = 0
            end_time = min(start_time + 10, video_duration)
            initial_clip = video.subclip(start_time, end_time)
            output_file = os.path.join(
                output_path, f"{video_name}-{dir_name}-{segment_index}.mp4"
            )
            initial_clip.write_videofile(
                output_file, codec="libx264", audio_codec="aac"
            )
            slice_info[output_file] = {"duration": end_time - start_time}
            segment_index += 1

            # 保存剩余部分的片段，每10秒一个片段
            start_time = end_time
            while start_time < video_duration:
                end_time = min(start_time + 10, video_duration)
                video_clip = video.subclip(start_time, end_time)
                output_file = os.path.join(
                    output_path, f"{video_name}-{dir_name}-{segment_index}.mp4"
                )
                video_clip.write_videofile(
                    output_file, codec="libx264", audio_codec="aac"
                )
                slice_info[output_file] = {"duration": end_time - start_time}
                start_time = end_time
                segment_index += 1


def process_videos(json_path, output_json_path, max_workers=4):
    """
    处理长时间的视频
    """
    video_info = load_video_info(json_path)
    slice_info = {}

    print("Processing videos...")
    print("len(video_info):", len(video_info), "before filtering")
    video_info = {
        k: v for k, v in video_info.items() if v["frame_length"] / v["fps"] > 12
    }
    print("len(video_info):", len(video_info), "after filtering")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for video_path in video_info.keys():
            tasks.append(executor.submit(slice_video, video_path, slice_info))

        for future in as_completed(tasks):
            try:
                future.result()
                print(f"Processed {future}")
            except Exception as e:
                print(f"Error processing video: {e}")

    print("Finished processing videos", len(slice_info))
    save_slice_info(slice_info, output_json_path)


def merge_and_filter(ori_json_path, cutting_json_path, output_json_path):
    """
    先过滤，之后在想办法
    """
    with open(ori_json_path, "r") as f:
        ori_video_info = json.load(f)
    with open(cutting_json_path, "r") as f:
        cutting_video_info = json.load(f)

    for video_path, video_info in cutting_video_info.items():

        output_path = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        ori_video_name = (
            video_name.split("-")[0] + "-" + video_name.split("-")[1] + ".mp4"
        )
        ori_video_path = os.path.join(output_path, ori_video_name)

        if ori_video_path in ori_video_info:
            ori_video_info.pop(ori_video_path)

        # ori_video_info[video_path] = video_info


    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ori_video_info, f, ensure_ascii=False)

    print(f"Merge and filter finished, saved to {output_json_path}")


if __name__ == "__main__":

    # video_root = "Download/douyin/post"
    # cutting_videos_in_directory(video_root, 64)

    max_workers = 64  # 最大线程数

    # json_path = '/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/TED_videos_selected_videos_0704_v1.json'
    # output_json_path = '/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/TED_videos_0704_v1_cutting.json'
    # process_videos(json_path, output_json_path, max_workers)

    json_path_list = [
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/TED_videos_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/TEDxTalks_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch01_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch02_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch03_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch04_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch05_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch06_selected_videos_0704_v1.json",
        # "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch07_selected_videos_0704_v1.json",
        "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch08_selected_videos_0704_v1.json",
        "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch09_selected_videos_0704_v1.json",
        "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch10_selected_videos_0704_v1.json",
        "/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Select_json_for_a2p_0704_v1/Youtube_batch11_selected_videos_0704_v1.json",
    ]
    # for json_path in json_path_list:
    #     output_json_path = json_path.replace(".json", "_cutting.json")
    #     process_videos(json_path, output_json_path, max_workers)

    for json_path in json_path_list:
        cutting_json_path = json_path.replace(".json", "_cutting.json")
        output_json_path = json_path.replace(".json", "_cutting_filtered.json")
        merge_and_filter(json_path, cutting_json_path, output_json_path)
        print(f"Finished processing {json_path}")