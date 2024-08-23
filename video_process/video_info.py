import os
import glob
import json
from moviepy.editor import VideoFileClip
import re
import string
from datetime import datetime
import cv2
import shutil
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import concurrent.futures

invalid_dir = "Temp_dir/douyin/invalid"


def check_video_format(video_path):
    try:
        # 尝试打开视频文件
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print("视频格式有问题")
            return False
    except cv2.error as e:
        print("视频格式有问题")
        return False

    print("视频格式正常")
    return True


def process_video(file_path):
    try:
        clip = VideoFileClip(file_path)
        duration = clip.duration
        print(f"{file_path} duration: {duration} seconds")
        return duration
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # shutil.move(file_path, invalid_dir)
        return 0


def get_video_duration_mp(
    directory, num_threads=4
):  # 可以通过参数指定线程数量，默认为 4
    os.makedirs(invalid_dir, exist_ok=True)

    total_duration = 0
    video_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:  # 指定线程数量
        for duration in executor.map(process_video, video_files):
            if duration:
                total_duration += duration

    print(f"Total duration of all videos: {total_duration/60/60} hours")


def get_video_duration(directory):

    invalid_dir = "Temp_dir/douyin/invalid"
    os.makedirs(invalid_dir, exist_ok=True)

    # Loop through each subfolder and calculate the total duration of all videos
    total_duration = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the current item is a video file
            if file.endswith(".mp4"):
                # Calculate the duration of the video file
                try:
                    clip = VideoFileClip(os.path.join(root, file))
                    total_duration += clip.duration
                    print(
                        f"{os.path.join(root, file)} duration: {clip.duration} seconds"
                    )
                except Exception as e:
                    shutil.move(os.path.join(root, file), invalid_dir)
            else:
                file_path = os.path.join(root, file)
                os.remove(file_path)

    print(f"Total duration of all videos: {total_duration/60/60} h")


def get_video_duration_by_json(file_path, save_path, num_threads=16):

    invalid_dir = "Temp_dir/douyin/invalid"
    os.makedirs(invalid_dir, exist_ok=True)

    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    print("Total number of videos: ", len(data))

    total_duration = 0
    # 提取视频路径和得分

    video_files = []
    for video_path, score in data.items():
        video_files.append(video_path)

    duration_list = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:  # 指定线程数量
        for duration in executor.map(process_video, video_files):
            if duration:
                duration_list.append(duration)
                total_duration += duration

    print(f"Total duration of all videos: {total_duration/60/60} hours")

    # 绘制直方图
    plt.hist(scores, bins=100, edgecolor="black")

    # 设置图形的标题和坐标轴标签
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.xlim(0, max_score + 0.01)
    plt.savefig(save_path)


def normalize_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    normalized_filename = "".join(c if c in valid_chars else "_" for c in filename)
    return normalized_filename


def filter_invalid_characters(filename):
    #     valid_chars = r"[^-()a-zA-Z0-9\u4e00-\u9fa5\s]"
    # valid_chars = r"[^-_.()a-zA-Z0-9\u4e00-\u9fa5\s]"
    valid_chars = r"[^-_.a-zA-Z0-9\u4e00-\u9fa5\s]"
    # print(filename)
    filtered_filename = re.sub(r"\([^)]*\)", "", filename)
    filtered_filename = re.sub(r"_+", "_", filtered_filename)
    filtered_filename = re.sub(valid_chars, "", filtered_filename)
    filtered_filename = filtered_filename.replace(" ", "_")
    return filtered_filename


def format_datetime(datetime_str):
    datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H-%M-%S")
    formatted_datetime = datetime_obj.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime


def differentiate_names(name):
    """用于区分已经处理和未处理的视频名称

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    # ”2020-08-26_18-13-20_video“
    # 第一种名称的正则表达式模式
    pattern1 = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_video.mp4"

    # 2018-11-28 21-07-14_学播音表演_的中天飞言___大家热爱播音表演的同......学都可以点点关注点点赞_接下来会分享很多干货哦__video
    # 第二种名称的正则表达式模式
    pattern2 = r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}_*_video.mp4"

    # 尝试匹配第一种名称
    if re.match(pattern1, name):
        return 1

    # 尝试匹配第二种名称
    elif re.match(pattern2, name):
        return 2

    # 无法匹配任何一种名称
    else:
        return 0


def save_motion_amplitudes_to_json(motion_amplitudes, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(motion_amplitudes, f, ensure_ascii=False)


def normalize_video_files_by_json(file_path, save_path):
    # 正规化视频文件名
    with open(file_path, "r") as f:
        data = json.load(f)
    selected_videos = {}
    for video_path, score in data.items():
        norm_video_path = re.sub(r"\([^)]*\)", "", video_path)
        norm_video_path = re.sub(r"_+", "_", norm_video_path)

        video_name = os.path.basename(norm_video_path)
        video_dir = os.path.dirname(norm_video_path)
        video_dir = video_dir.replace("cascade_cut", "normalized_casecade_cut")

        norm_video_path = os.path.join(video_dir, video_name)
        selected_videos[norm_video_path] = score

        if os.path.exists(norm_video_path):
            continue

        os.makedirs(video_dir, exist_ok=True)
        shutil.copy2(video_path, video_dir)

        print(norm_video_path)

    save_json_path = (
        save_path + "/selected_videos_by_body_pose_text_detection_normalized.json"
    )
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")

    return save_json_path


def normalize_video_files(directory):
    # 正规化视频文件名
    normalized_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                # continue
                name_type = differentiate_names(file)

                if name_type == 1:
                    continue
                elif name_type == 2:
                    parts = file.split("_")
                    if len(parts) >= 3:

                        datetime_str = "_".join(parts[:1])
                        print(datetime_str)
                        formatted_datetime = format_datetime(datetime_str)

                        new_filename = f"{formatted_datetime}_{parts[-1]}"
                        normalized_filename = filter_invalid_characters(new_filename)
                        normalized_path = os.path.join(root, normalized_filename)
                        normalized_files.append(normalized_path)
                        os.rename(os.path.join(root, file), normalized_path)
                else:
                    normalized_filename = filter_invalid_characters(file)
                    normalized_path = os.path.join(root, normalized_filename)
                    normalized_files.append(normalized_path)
                    os.rename(os.path.join(root, file), normalized_path)

            else:
                file_path = os.path.join(root, file)
                os.remove(file_path)

    print("Video files have been normalized")

    return normalized_files


def count_num(directory):
    count_num = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            count_num += 1
    print("Total number of video clips: ", count_num)


def summarize_info(
    motion_info_path,
    pose_info_path,
    consistency_info_path,
    camera_motion_info_path,
    denoised_audio_json_path,
    save_path,
    date=0,
):
    with open(motion_info_path, "r") as f:
        motion_data = json.load(f)

    # 正规化视频文件名
    with open(pose_info_path, "r") as f:
        pose_data = json.load(f)

    with open(consistency_info_path, "r") as f:
        consistency_data = json.load(f)

    with open(camera_motion_info_path, "r") as f:
        camera_motion_data = json.load(f)
    
    with open(denoised_audio_json_path, "r") as f:
        audio_data = json.load(f)

    selected_videos = {}
    for video_path, motion_score in camera_motion_data.items():
        print(video_path)

        if video_path not in pose_data or video_path not in consistency_data:
            continue

        pose_score = pose_data[video_path]
        consistency_score = consistency_data[video_path]
        optical_flow = motion_data[video_path]

        selected_videos[video_path] = {
            "optical_flow": optical_flow,
            "pose_score": pose_score["pose_score"][0][0],
            "frame_length": pose_score["video_length"],
            "xyxy_bbox": [
                int(pose_score["bbox"][0]),
                int(pose_score["bbox"][1]),
                int(pose_score["bbox"][2]),
                int(pose_score["bbox"][3]),
            ],
            "fps": pose_score["fps"],
            "resolution": pose_score["resolution"],
            "hand_score": pose_score["hand_score"],
            "text_score": pose_score["text_score"],
            "bbox_area_ratio": pose_score["area_score"],
            "consistency_score": consistency_score["consistency"],
            "motion_score": int(motion_score["camera rotating"]),
            "audio_path": audio_data[video_path]["audio_denoising"],
        }

        print(selected_videos[video_path])

    save_json_path = save_path + "/final_selected_videos_{}.json".format(date)
    save_motion_amplitudes_to_json(selected_videos, save_json_path)


class VideoInfoTask:
    def __init__(self, video_path):
        self.video_path = video_path
        self.resolution = None
        self.fps = 0
        self.audio_sample_rate = 0
        self.len = 0
        self.audio_path = None

    def get_video_info(self):
        clip = VideoFileClip(self.video_path)
        self.len = int(clip.duration * clip.fps)
        self.fps = clip.fps
        self.resolution = clip.size
        self.audio_sample_rate = clip.audio.fps if clip.audio else None
        print(
            f"Video: {self.video_path}, Len: {self.len}, FPS: {self.fps}, \
                Resolution: {self.resolution}, Audio Sample Rate: {self.audio_sample_rate}"
        )
        clip.close()

    def extract_audio(self):
        out_path = self.video_path.replace("cascade_cut", "audio_files")
        out_path = out_path.replace(".mp4", ".wav")

        if os.path.exists(out_path):
            return

        out_path_dir = os.path.dirname(out_path)
        os.makedirs(out_path_dir, exist_ok=True)
        out_path_name = os.path.basename(out_path)
        out_path_name = out_path_name.replace(".mp4", ".wav")

        try:
            with VideoFileClip(self.video_path) as video:
                audio = video.audio
                audio.write_audiofile(out_path)
            self.audio_path = out_path
        except Exception as e:
            print(f"Error extracting audio from {self.video_path}: {e}")


def get_video_info(file_path, save_path, date=0):
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    print("Total number of videos: ", len(data))

    video_files = []
    for video_path, score in data.items():
        video_files.append(video_path)

    video_info_tasks = [VideoInfoTask(video_path) for video_path in video_files]

    selected_videos = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=32
    ) as executor:  # 指定最大线程数为8
        executor.map(lambda task: task.get_video_info(), video_info_tasks)

    for task in video_info_tasks:
        selected_videos[task.video_path] = {
            "video_len": task.len,
            "resolution": task.resolution,
            "fps": task.fps,
            "audio_sample_rate": task.audio_sample_rate,
        }

    # 选取得分大于阈值的视频
    final_selected_videos = {}
    for video_path, score in selected_videos.items():
        scale = score["resolution"]
        video_len = score["video_len"]

        if scale == [1280, 720] and video_len > 60:
            final_selected_videos[video_path] = score

    save_json_path = save_path + "/selected_videos_info_{}.json".format(date)
    save_motion_amplitudes_to_json(final_selected_videos, save_json_path)

    print(
        f"Selected videos have been saved to {save_path}, length: {len(final_selected_videos)}"
    )
    return save_json_path


def resample_audio(
    video_path, target_fps=25, target_audio_fps=16000, output_path="output.mp4"
):
    """将视频文件的帧率和音频采样率统一为指定值
    TODO 目前音频采样率不用统一，后续的wav2vec会进行下采样到16k
    """
    # 加载视频文件
    clip = VideoFileClip(video_path)

    # 统一视频帧率
    clip = clip.set_fps(target_fps)

    # 检查视频是否有音频
    # if clip.audio is not None:
    #     # 降采样音频
    #     clip = clip.set_audio_fps(target_audio_fps)

    # 保存处理后的视频文件
    clip.write_videofile(output_path)


def get_audio_info(file_path, save_path, date=0):
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    print("Total number of videos: ", len(data))

    video_files = []
    for video_path, score in data.items():
        video_files.append(video_path)

    video_info_tasks = [VideoInfoTask(video_path) for video_path in video_files]

    selected_videos = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=32
    ) as executor:  # 指定最大线程数为8
        executor.map(lambda task: task.extract_audio(), video_info_tasks)

    for task in video_info_tasks:
        selected_videos[task.video_path] = {
            "audio_path": task.audio_path,
        }

    save_json_path = save_path + "/selected_videos_audio_{}.json".format(date)
    save_motion_amplitudes_to_json(selected_videos, save_json_path)
    return save_json_path


if __name__ == "__main__":
    # Set the path to the directory containing the subfolders with video files
    directory = "Download/douyin/post"

    # # check_video_format("Download/douyin/post/焱鹿文化艺术培训/2019-03-10 21-48-46__焱鹿少儿口才__video.mp4")

    # # 打印正规化后的视频文件路径
    # for video in normalized_videos:
    #     print(video)

    # # 计算视频文件的总时长
    # get_video_duration(directory)

    # 统计视频片段个数
    # cut_directory = "Download/douyin/cascade_cut"
    # count_num(cut_directory)

    # 通过JSON文件计算视频文件的总时长
    # file_path = "Download/douyin/filter_videos_by_pose_detection.json"
    # get_video_duration_by_json(file_path)

    # get_video_duration(cut_directory)

    # normalized_videos = normalize_video_files(cut_directory)

    # # 打印正规化后的视频文件路径
    # for video in normalized_videos:
    #     print(video)

    # video_folder = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/TED/TEDxNTHU/cascade_cut/_._TEDxUTCC-zFFHpJhOuZE"
    # videos_info = get_video_info(video_folder)
    # for video_info in videos_info:
    #     print(f"视频路径：{video_info['video_path']}")
    #     print(f"分辨率：{video_info['resolution']}")
    #     print(f"帧率：{video_info['fps']}")
    #     print(f"音频采样率：{video_info['audio_sample_rate']}")

    video_path = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/TED/TED_videos/cascade_cut/JQ0iMulicgg/Scene-158.mp4"
    output_path = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir/resample_video/Scene-158_v2.mp4"
    resample_audio(
        video_path, target_fps=25, target_audio_fps=16000, output_path=output_path
    )
