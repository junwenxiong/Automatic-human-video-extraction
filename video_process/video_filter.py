import os
import os
import json
import matplotlib.pyplot as plt
import json
import shutil


def plot_text_scores(file_path, save_path, title="score distribution"):
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 提取视频路径和得分
    video_paths = []
    scores = []
    for video_path, score in data.items():
        video_paths.append(video_path)
        scores.append(score["text_result"])

    max_score = max(scores)
    # 绘制直方图
    plt.hist(scores, bins=100, edgecolor="black")

    # 设置图形的标题和坐标轴标签
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xlim(0, max_score + 0.01)
    plt.savefig(save_path)


def plot_flow_scores(file_path, save_path, title="score distribution"):
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 提取视频路径和得分
    video_paths = []
    scores = []
    for video_path, score in data.items():
        video_paths.append(video_path)
        scores.append(score)

    max_score = max(scores)
    # 绘制直方图
    plt.hist(scores, bins=100, edgecolor="black")

    # 设置图形的标题和坐标轴标签
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xlim(0, max_score + 0.01)
    plt.savefig(save_path)


def save_motion_amplitudes_to_json(motion_amplitudes, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(motion_amplitudes, f, ensure_ascii=False)

    print(f"Motion amplitudes saved to {output_file}")


def select_video_by_optical_flow(file_path, save_path, threshold_low):
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filtered_by_optical_flow_{threshold_low}"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        if score > threshold_low:
            selected_videos[video_path] = score

        elif score <= threshold_low:
            vide_dirs = video_path.split("/")
            tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
            os.makedirs(tmp_low_folder, exist_ok=True)
            shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + "/selected_videos_by_optical_flow.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")

    return save_json_path


def select_video_by_text_detection(file_path, save_path, threshold_low=0.025):
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filtered_by_text_detection_{threshold_low}"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        if score < threshold_low:
            selected_videos[video_path] = score

        elif score >= threshold_low:
            vide_dirs = video_path.split("/")
            tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
            os.makedirs(tmp_low_folder, exist_ok=True)
            shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + "/selected_videos_by_text_detection.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")

    return save_json_path


def select_video_by_body_pose_text_detection(
    file_path, save_path, text_threshold=0.15, pose_threshold=1
):
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filtered_by_body_pose_text_detection_{text_threshold}"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        text_score = score["text_result"]
        pose_score = score["pose_result"]

        is_pose_score = []
        if text_score <= text_threshold and len(pose_score) != 0:
            for pose in pose_score:
                if (
                    isinstance(pose, list)
                    and len(pose) == 1
                    and pose[0] > pose_threshold
                ):
                    is_pose_score.append(True)
                elif isinstance(pose, int) and pose > pose_threshold:
                    is_pose_score.append(True)
                else:
                    is_pose_score.append(False)

            if False not in is_pose_score:
                selected_videos[video_path] = {
                    "text_score": text_score,
                    "pose_score": pose_score,
                }
            else:
                vide_dirs = video_path.split("/")
                tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
                # os.makedirs(tmp_low_folder, exist_ok=True)
                # shutil.copy(video_path, tmp_low_folder)

        else:
            vide_dirs = video_path.split("/")
            # tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
            # os.makedirs(tmp_low_folder, exist_ok=True)
            # shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + "/selected_videos_by_body_pose_text_detection.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")

    return save_json_path


def select_video_by_pose_detection(file_path, save_path):
    """利用姿态检测结果选择视频

    Args:
        file_path (_type_): _description_
        save_path (_type_): _description_
        threshold_low (_type_): _description_
    """
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filtered_by_pose_detection"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        if False not in score:
            selected_videos[video_path] = score
        else:
            vide_dirs = video_path.split("/")
            tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
            os.makedirs(tmp_low_folder, exist_ok=True)
            shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + "/selected_videos_by_pose_detection.json"

    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")

    return save_json_path


def judge_av_consistency(scores):
    if scores == []:
        return [], 0

    frame_len = len(scores)

    current_sequence = 0
    max_sequence = 0
    start_index = 0
    end_index = 0
    seq_list = []
    for i, item in enumerate(scores):
        if item == [[1]]:
            if current_sequence == 0:
                start_index = i
            current_sequence += 1
            end_index = i
            if current_sequence > max_sequence:
                max_sequence = current_sequence
        else:
            current_sequence = 0
            if max_sequence > 0:
                seq_list.append([start_index, end_index])
                max_sequence = 0

    return seq_list, frame_len


def select_video_by_av_consistency(file_path, save_path, save_flag=0):
    """利用音视一致性结果选择视频

    Args:
        file_path (_type_): _description_
        save_path (_type_): _description_
        threshold_low (_type_): _description_
    """
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filted_by_av_consistency"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():

        ones_seq_list, frame_len = judge_av_consistency(score)
        if ones_seq_list != []:
            print(video_path, score, ones_seq_list, frame_len)

            selected_videos[video_path] = {
                "consistency": ones_seq_list,
                "frame_len": frame_len,
            }

        else:
            vide_dirs = video_path.split("/")
            tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
            os.makedirs(tmp_low_folder, exist_ok=True)
            shutil.copy(video_path, tmp_low_folder)

    if save_flag == 0:
        save_json_path = save_path + f"/selected_videos_by_av_consistency.json"
    else:
        save_json_path = save_path + f"/selected_videos_by_av_consistency_{save_flag}.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")
    return save_json_path


def select_video_by_camera_rotating(file_path, save_path, save_flag=0):
    """利用音视一致性结果选择视频

    Args:
        file_path (_type_): _description_
        save_path (_type_): _description_
        threshold_low (_type_): _description_
    """
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filted_by_camera_rotating"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():

        rotating_score = score["camera rotating"]
        bbox = score["human bbox"]

        if rotating_score < 10.0:
            print(video_path, rotating_score, bbox)

            selected_videos[video_path] = {
                "camera rotating": rotating_score,
                "human bbox": bbox,
            }
        else:
            vide_dirs = video_path.split("/")
            tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
            os.makedirs(tmp_low_folder, exist_ok=True)
            shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + f"/selected_videos_by_camera_rotating_{save_flag}.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")
    return save_json_path


if __name__ == "__main__":

    img_save_path = "Download/douyin/text_detection_score_distribution.png"
    save_path = "Download_0308/douyin/filter_info"
    score_files = "Download_0308/douyin/filter_info/AV_Consistency.json"

    # plot_flow_scores(score_files, img_save_path, title="detected text score distribution")
    select_video_by_av_consistency(score_files, save_path)
