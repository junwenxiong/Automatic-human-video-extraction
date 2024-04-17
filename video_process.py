import os
import torch
import json
import os
from video_process.video_cutting import cutting_videos_in_directory
from video_process.video_filter import (
    plot_flow_scores,
    plot_text_scores,
    select_video_by_av_consistency,
    select_video_by_text_detection,
    select_video_by_optical_flow,
    select_video_by_pose_detection,
    select_video_by_body_pose_text_detection,
    select_video_by_camera_rotating,
)
from video_process.video_info import (
    count_num,
    normalize_video_files,
    normalize_video_files_by_json,
    get_video_duration_by_json,
    get_video_duration,
    get_video_duration_mp,
    summarize_info,
    get_video_info,
)
from video_process.video_motion import (
    compute_motion_amplitudes,
    mp_camera_rotating_detect_process,
)
from video_process.video_pose_extraction import mp_pose_detection_process
from video_process.video_text_detection import mp_text_detect_process
from video_process.video_av_consistency import mp_av_consistency_detect_process

is_camera_rotating = False

is_av_consistency = False
is_text_detection = False

is_flow_computing = False

is_normlize_files = False
is_cutting_videos = False


def split_json(input_json_path, output_dir, batch_size=10000):
    # 读取原始的JSON文件
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # 确定要拆分的批次数量
    num_batches = (len(data) + batch_size - 1) // batch_size
    os.makedirs(output_dir, exist_ok=True)

    # 拆分JSON数据并保存为多个文件
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = {k: data[k] for k in list(data.keys())[start_idx:end_idx]}
        output_json_path = os.path.join(output_dir, f"videos_{i + 1}.json")
        with open(output_json_path, "w") as f:
            json.dump(batch_data, f, indent=4)

        print(f"Saved batch {i + 1} to {output_json_path}")


def print_info(json_path, video_root):
    # get_video_duration(video_root)
    get_video_duration_by_json(
        json_path, save_path=os.path.join(video_root, "av_consistency_video_time.png")
    )


if __name__ == "__main__":

    thread_num = 4
    video_root = (
        "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/TED/TED_videos"
    )
    prefix_str = "TED_videos"

    info_json_path = os.path.join(video_root, "selected_videos_by_camera_rotating.json")

    selected_flow_json_path = None
    selected_text_json_path = None
    selected_av_json_path = None

    video_dir_path = os.path.join(video_root, "post")
    if is_normlize_files:
        print("normalizing files, please wait...")
        norm_files = normalize_video_files(video_root)
        video_nums = count_num(video_root)
        # get_video_duration_mp(video_root, 16)

    if is_cutting_videos:
        print("cutting videos, please wait...")
        cutting_videos_in_directory(video_dir_path, 32)

    cutting_video_root = os.path.join(video_root, "cascade_cut")

    save_path = video_root

    if is_flow_computing:
        print("computing motion amplitudes, please wait...")
        flow_json_path = compute_motion_amplitudes(cutting_video_root, video_root)

        save_flow_fig_path = os.path.join(video_root, "flow_score.png")
        plot_flow_scores(
            flow_json_path, save_flow_fig_path, title="motion score distribution"
        )

        selected_flow_json_path = select_video_by_optical_flow(
            flow_json_path, save_path, threshold_low=10
        )

    # 完成单人检测-姿态检测-文本重叠
    if is_text_detection:
        print("detecting text, please wait...")
        if selected_flow_json_path is None:
            selected_flow_json_path = os.path.join(
                video_root, "selected_videos_by_optical_flow.json"
            )
        detected_text_json_path = mp_text_detect_process(
            selected_flow_json_path,
            save_path,
            threads=thread_num,
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        detected_text_json_path = os.path.join(video_root, "text_detection.json")
        save_text_fig_path = os.path.join(video_root, "text_area.png")
        plot_text_scores(
            detected_text_json_path,
            save_text_fig_path,
            title="detected text score distribution",
        )
        selected_text_json_path = select_video_by_body_pose_text_detection(
            detected_text_json_path, save_path, text_threshold=0.10, pose_threshold=1
        )

    if is_av_consistency:
        if selected_text_json_path is None:
            selected_text_json_path = os.path.join(
                video_root, "selected_videos_by_body_pose_text_detection.json"
            )

        print(f"detecting av consistency, please wait... {selected_text_json_path}")

        av_consistency_json_file = mp_av_consistency_detect_process(
            selected_text_json_path,
            save_path,
            prefix_str=prefix_str,
            threads=thread_num,
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        selected_av_json_path = select_video_by_av_consistency(
            av_consistency_json_file, save_path
        )

    if is_camera_rotating:
        if selected_av_json_path is None:
            selected_av_json_path = os.path.join(
                video_root, "selected_videos_by_av_consistency.json"
            )

        print(f"detecting camera rotating, please wait... {selected_av_json_path}")

        camera_rotating_json_file = mp_camera_rotating_detect_process(
            selected_av_json_path,
            save_path,
            threads=thread_num,
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        camera_rotating_json_file = os.path.join(
            video_root, "camera_rotating_detection.json"
        )
        selected_rotating_json_path = select_video_by_camera_rotating(
            camera_rotating_json_file, save_path
        )

    # split_json(info_json_path, os.path.join(video_root, "body_pose_text_jsons"))

    # print_info(info_json_path, video_dir_path)

    # pose_info_path = "TED/TEDxTalks/selected_videos_by_body_pose_text_detection.json"
    # consistency_info_path = "TED/TEDxTalks/selected_videos_by_av_consistency.json"
    # motion_info_path = "TED/TEDxTalks/selected_videos_by_camera_rotating.json"
    # # select_info_path = "TED/TEDxTalks/selected_videos.json"
    # select_info_path = "TED/TEDxTalks/selected_videos_info.json"

    # save_path = "TED/TEDxTalks"

    # summarize_info(
    #     pose_info_path,
    #     consistency_info_path,
    #     motion_info_path,
    #     select_info_path,
    #     save_path,
    # )

    # get_video_info(select_info_path, save_path)
