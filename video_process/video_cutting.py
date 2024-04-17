import os
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import threading
import shutil


class VideoProcessingThread(threading.Thread):
    def __init__(self, video_paths):
        super(VideoProcessingThread, self).__init__()
        self.video_paths = video_paths

    def run(self):
        for video_path in self.video_paths:
            video_save_path = video_path.replace("post", "cascade_cut")
            video_save_path = video_save_path.replace(".mp4", "")
            os.makedirs(video_save_path, exist_ok=True)
            print(f"cutting video {video_save_path}")
            cascade_cut(video_path, video_save_path)


def detect_cut(video_path, video_save_path, threshold=27.0):
    """按镜头切割视频，同时保存视频片段

    Args:
        video_path (_type_): _description_
        threshold (float, optional): _description_. Defaults to 27.0.
    """

    video = open_video(video_path)
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
    scene_manager = SceneManager()

    # Add the content detectors to the scene manager
    # Detector 1: Sudden changes with a high frame rate and threshold
    scene_manager.add_detector(ContentDetector(threshold=30))

    # Detector 2: Slow changes with a lower frame rate and threshold
    scene_manager.add_detector(
        ContentDetector(threshold=20, min_scene_len=30)
    )

    # Detector 3: Slow changes with an even lower frame rate and threshold
    scene_manager.add_detector(
        ContentDetector(
            threshold=15,
            min_scene_len=30
        )
    )

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
        print("scene detected")
    else:
        shutil.copy(video_path, video_save_path + "/" + "Scene-0.mp4")
        print("no scene detected")
    print(video_save_path, scene_list)


def cutting_videos_in_directory(directory, num_threads=32):
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


if __name__ == "__main__":

    video_root = "Download/douyin/post"
    cutting_videos_in_directory(video_root, 64)
