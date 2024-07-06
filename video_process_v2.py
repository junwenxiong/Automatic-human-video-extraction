import os
import argparse
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
from video_process.video_text_detection import mp_text_detect_process
from video_process.video_av_consistency import mp_av_consistency_detect_process


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="TED")
    parser.add_argument("--dataset_name", type=str, default="batch_08")
    parser.add_argument("--date", type=str, default="0702")
    parser.add_argument("--is_text_detection", type=int, default=0)
    parser.add_argument("--is_av_consistency", type=int, default=0)
    parser.add_argument("--is_camera_rotating", type=int, default=0)
    args = parser.parse_args()

    # prefix_str = "batch_11"
    thread_num = 4

    is_camera_rotating = args.is_camera_rotating
    is_av_consistency = args.is_av_consistency
    is_text_detection = args.is_text_detection

    is_flow_computing = False
    is_normlize_files = False
    is_cutting_videos = False

    prefix_str = args.dataset_name
    date = args.date

    if args.dataset == "TED":
        video_root = (
            f"/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/TED/{prefix_str}"
        )
    else:
        video_root = f"/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/Youtube_videos/{prefix_str}"

    info_json_path = os.path.join(video_root, "selected_videos_by_camera_rotating.json")

    selected_flow_json_path = None
    selected_text_json_path = None
    selected_av_json_path = None
    selected_video_json_path = None
    flow_json_path = os.path.join(video_root, "motion_amplitudes.json")

    # 一定要记得加post文件夹
    video_dir_path = os.path.join(video_root, "post")
    # 正则化视频名称
    if is_normlize_files:
        print("normalizing files, please wait...")
        norm_files = normalize_video_files(video_root)
        video_nums = count_num(video_root)
        # get_video_duration_mp(video_root, 16)

    # 剪切视频
    if is_cutting_videos:
        print("cutting videos, please wait...")
        cutting_videos_in_directory(video_dir_path, 32)

    cutting_video_root = os.path.join(video_root, "cascade_cut")

    save_path = video_root

    # 计算光流
    if is_flow_computing:
        print("computing motion amplitudes, please wait...")
        flow_json_path = compute_motion_amplitudes(cutting_video_root, video_root)

        save_flow_fig_path = os.path.join(video_root, "flow_score.png")
        plot_flow_scores(
            flow_json_path, save_flow_fig_path, title="motion score distribution"
        )

        selected_flow_json_path = select_video_by_optical_flow(
            flow_json_path,
            save_path,
            threshold_low=7,
            threshold_high=30,
            date=date,
        )

    # 开始调用GPU
    # 完成单人检测-姿态检测-文本重叠
    if is_text_detection == 1:
        print("detecting text, please wait...")
        if selected_flow_json_path is None:
            selected_flow_json_path = os.path.join(
                video_root, "selected_videos_by_optical_flow_{}.json".format(date)
            )
            print(f"selected_video_by_optical_flow: {selected_flow_json_path}")

        detected_text_json_path = mp_text_detect_process(
            selected_flow_json_path,
            save_path,
            threads=thread_num,
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            date=date,
        )

        detected_text_json_path = os.path.join(
            video_root, "text_detection_{}.json".format(date)
        )
        selected_text_json_path = select_video_by_body_pose_text_detection(
            detected_text_json_path,
            save_path,
            text_threshold=0.10,
            pose_threshold=1,
            date=date,
        )

    # 音视同步性分析
    if is_av_consistency == 1:
        if selected_text_json_path is None:
            selected_text_json_path = os.path.join(
                video_root,
                "selected_videos_by_body_pose_text_detection_{}.json".format(date),
            )

        print(f"detecting av consistency, please wait... {selected_text_json_path}")

        av_consistency_json_file = mp_av_consistency_detect_process(
            selected_text_json_path,
            save_path,
            prefix_str=prefix_str,
            threads=thread_num,
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            date=date,
        )

        selected_av_json_path = select_video_by_av_consistency(
            av_consistency_json_file,
            save_path,
            date=date,
        )

    # 视频背景运动计算
    if is_camera_rotating == 1:
        if selected_av_json_path is None:
            selected_av_json_path = os.path.join(
                video_root, "selected_videos_by_av_consistency_{}.json".format(date)
            )

        if selected_text_json_path is None:
            selected_text_json_path = os.path.join(
                video_root,
                "selected_videos_by_body_pose_text_detection_{}.json".format(date),
            )

        print(f"detecting camera rotating, please wait... {selected_av_json_path}")

        camera_rotating_json_file = mp_camera_rotating_detect_process(
            selected_av_json_path,
            selected_text_json_path,
            save_path,
            threads=thread_num,
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            date=date,
        )
        camera_rotating_json_file = os.path.join(
            video_root, "camera_rotating_detection_{}.json".format(date)
        )
        selected_rotating_json_path = select_video_by_camera_rotating(
            camera_rotating_json_file, save_path, date=date
        )

    # # 通用操作
    video_motion_path = f"{video_root}/selected_videos_by_optical_flow_{date}.json"
    pose_info_path = (
        f"{video_root}/selected_videos_by_body_pose_text_detection_{date}.json"
    )
    consistency_info_path = (
        f"{video_root}/selected_videos_by_av_consistency_{date}.json"
    )
    camera_motion_info_path = (
        f"{video_root}/selected_videos_by_camera_rotating_{date}.json"
    )

    summarize_info(
        video_motion_path,
        pose_info_path,
        consistency_info_path,
        camera_motion_info_path,
        video_root,
        date=date,
    )
