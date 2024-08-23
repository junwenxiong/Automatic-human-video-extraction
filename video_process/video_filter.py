import os
import cv2
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


def dict_slice(d, start, end):
    """
    对字典进行切片操作。

    :param d: 原始字典
    :param start: 起始索引
    :param end: 结束索引
    :return: 切片后的字典
    """
    items = list(d.items())[start:end]
    return dict(items)


def select_video_by_optical_flow(
    file_path, save_path, threshold_low=10, threshold_high=40, date=1
):
    print("select_video_by_optical_flow")
    print("Config: ", threshold_low, threshold_high, date)

    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filtered_by_optical_flow_{threshold_low}_{threshold_high}"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        if threshold_low < score < threshold_high:
            selected_videos[video_path] = score
        # else:
        #     vide_dirs = video_path.split("/")
        #     tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
        #     os.makedirs(tmp_low_folder, exist_ok=True)
        #     shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + "/selected_videos_by_optical_flow_{}.json".format(date)
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(
        f"Selected videos have been saved to {save_path}, length: {len(selected_videos)}"
    )

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

    print(
        f"Selected videos have been saved to {save_path}, length: {len(selected_videos)}"
    )

    return save_json_path


def select_video_by_body_pose_text_detection(
    file_path, save_path, text_threshold=0.15, pose_threshold=1, date=0
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
        text_score = score["text_score"]
        pose_score = score["pose_score"]
        hand_score = score["hand_score"]
        count = sum(1 for s in hand_score if 1 in s)

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

            if False not in is_pose_score and count >= int(len(hand_score) // 2):
                selected_videos[video_path] = score
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

    save_json_path = (
        save_path + "/selected_videos_by_body_pose_text_detection_{}.json".format(date)
    )
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")

    return save_json_path


def select_video_by_pose_detection(file_path, save_path, date):
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
        save_path + f"/filtered_by_refined_body_hands_detection"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        # score: [[1], [], [], [], [], [], [], [], []]
        # 统计这里面的[1]的个数
        body_hands_scroe = score["pose_result"]
        count = sum(1 for s in body_hands_scroe if 1 in s)
        if count >= int(len(body_hands_scroe) // 2):  # 大于一半就认为手部是出现的
            selected_videos[video_path] = score
        # else:
        #     vide_dirs = video_path.split("/")
        #     tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
        #     os.makedirs(tmp_low_folder, exist_ok=True)
        #     shutil.copy(video_path, tmp_low_folder)

    save_json_path = (
        save_path
        + "/selected_videos_by_refined_body_hands_detection_{}.json".format(date)
    )

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


def select_video_by_av_consistency(file_path, save_path, save_flag=0, date=0):
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
        # else:
        #     vide_dirs = video_path.split("/")
        #     tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
        #     os.makedirs(tmp_low_folder, exist_ok=True)
        #     shutil.copy(video_path, tmp_low_folder)

    if save_flag == 0:
        save_json_path = (
            save_path + "/selected_videos_by_av_consistency_{}.json".format(date)
        )
    else:
        save_json_path = (
            save_path + f"/selected_videos_by_av_consistency_{save_flag}.json"
        )
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")
    return save_json_path


def select_video_by_camera_rotating(file_path, save_path, date=0):
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
        if rotating_score < 10.0:
            selected_videos[video_path] = score
        # else:
        #     vide_dirs = video_path.split("/")
        #     tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
        #     os.makedirs(tmp_low_folder, exist_ok=True)
        #     shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + f"/selected_videos_by_camera_rotating_{date}.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(f"Selected videos have been saved to {save_path}")
    return save_json_path


def select_video_by_multi_vars(file_path, save_path, flag):
    """只保存optical_flow<40, scale:[1280, 720],motion:0的数据
        merge所有获得的数据，包括pose array

    Args:
        file_path (_type_): _description_
        save_path (_type_): _description_
        threshold_low (_type_): _description_
    """
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)
    print("length of data", len(data))

    # 创建保存视频片段的文件夹
    low_folder = (
        save_path + f"/filted_by_multi_vars"
    )  # 替换为低阈值视频保存文件夹的路径
    os.makedirs(low_folder, exist_ok=True)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        optical_flow = score["optical_flow"]
        scale = score["resolution"]
        motion = score["motion_score"]

        pose_arr_path = video_path.replace("cascade_cut", "dwpose_array")
        pose_arr_path = pose_arr_path.replace("mp4", "npy")

        audio_path = video_path.replace("cascade_cut", "audio_files")
        audio_path = audio_path.replace(".mp4", "_denoised.wav")

        if scale == [1280, 720] and motion == 0:
            if os.path.exists(pose_arr_path) and os.path.exists(audio_path):
                score["pose"] = pose_arr_path
                score["audio_path"] = audio_path

                selected_videos[video_path] = score
        # else:
        #     vide_dirs = video_path.split("/")
        #     tmp_low_folder = low_folder + "/" + vide_dirs[-3] + "/" + vide_dirs[-2]
        #     os.makedirs(tmp_low_folder, exist_ok=True)
        #     shutil.copy(video_path, tmp_low_folder)

    save_json_path = save_path + f"/{flag}_selected_videos_by_multi_vars.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(len(selected_videos))
    print(f"Selected videos have been saved to {save_path}")
    return save_json_path


def select_video_by_length(file_path, save_path, flag):
    """只保存optical_flow<40, scale:[1280, 720],motion:0的数据

    Args:
        file_path (_type_): _description_
        save_path (_type_): _description_
        threshold_low (_type_): _description_
    """
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        frame_length = score["frame_length"]

        if 600 > frame_length > 60:
            selected_videos[video_path] = score

    save_json_path = save_path + f"/{flag}_selected_videos_by_length_60.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(len(selected_videos))

    print(f"Selected videos have been saved to {save_path}")
    return save_json_path


def is_bbox_valid(bbox, width, height):
    x1, y1, x2, y2 = bbox
    # 检查坐标是否在图像范围内
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return False
    return True


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


def select_video_by_check_bbox(file_path, save_path, flag):
    """只保存optical_flow<40, scale:[1280, 720],motion:0的数据

    Args:
        file_path (_type_): _description_
        save_path (_type_): _description_
        threshold_low (_type_): _description_
    """
    # 读取保存得分的JSON文件
    with open(file_path, "r") as f:
        data = json.load(f)

    print(len(data))

    # 选取得分大于阈值的视频
    selected_videos = {}
    for video_path, score in data.items():
        bbox = score["xyxy_bbox"]
        check_result = is_bbox_valid(bbox, 1280, 720)

        cap = cv2.VideoCapture(video_path)  # 替换为你的视频路径
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = total_frames // 2  # 选择中间帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        frame = draw_bbox(frame, bbox)

        if check_result:
            selected_videos[video_path] = score
            cv2.imwrite(
                f"tmp/bbox_7_04_correct/{video_path.split('/')[-1].split('.')[0]}.jpg",
                frame,
            )
        else:
            cv2.imwrite(
                f"tmp/bbox_6_04/{video_path.split('/')[-1].split('.')[0]}.jpg", frame
            )
            print(video_path, bbox)

    save_json_path = save_path + f"/selected_videos_by_bbox.json"
    save_motion_amplitudes_to_json(selected_videos, save_json_path)

    print(len(selected_videos))
    print(f"Selected videos have been saved to {save_path}")
    return save_json_path


if __name__ == "__main__":

    # flag = "TED_videos"
    # flag="TEDxTalks"
    # flag="Youtube_batch10"
    flag_list = [
        "TED_videos",
        "TEDxTalks",
        "Youtube_batch10",
        "Youtube_batch01",
        "Youtube_batch02",
        "Youtube_batch03",
        "Youtube_batch04",
        "Youtube_batch05",
        "Youtube_batch06",
        "Youtube_batch07",
        "Youtube_batch08",
        "Youtube_batch09",
        "Youtube_batch11",
    ]

    # select_video_by_check_bbox(json_file, save_path, "Youtube_batch01")

    # plot_flow_scores(score_files, img_save_path, title="detected text score distribution")

    save_path = "Select_json_for_a2p_0704_v1"

    for flag in flag_list:
        json_file = (
            f"Select_json_for_a2p_0704_v1/{flag}_selected_videos_0704_v1_cutting_filtered.json"
        )
        select_video_by_multi_vars(json_file, save_path, flag)

    # json_file2 = f"Select_json/{flag}_selected_videos_by_multi_vars.json"
    # select_video_by_length(json_file2, save_path, flag)
