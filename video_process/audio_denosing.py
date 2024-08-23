# -*- coding: utf-8 -*-
import random
import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import torch.nn as nn
import numpy as np
import json
import torch.multiprocessing as mp
from PIL import Image
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import soundfile as sf
import librosa


def audio_denoising(arg):
    thread_id, gpu_id, video_lists = arg
    # 设置线程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # 加载音频去噪模型
    audio_denoising_model = pretrained.dns64().to(device)

    print("Loading audio denoising model")

    text_results = {}
    for video_path, audio_path in video_lists:

        output_path = audio_path.replace(".wav", "_denoised.wav")

        if not os.path.exists(output_path):
            wav, sr = torchaudio.load(audio_path)
            wav = convert_audio(
                wav.to(device),
                sr,
                audio_denoising_model.sample_rate,
                audio_denoising_model.chin,
            )
            with torch.no_grad():
                denoised = audio_denoising_model(wav[None])[0]
            denoised = denoised.squeeze(0)

            sf.write(
                output_path,
                denoised.data.cpu().numpy(),
                audio_denoising_model.sample_rate,
            )

        result_dict = {
            "audio_denoising": output_path,
        }
        text_results[video_path] = result_dict
        print("{}: {}".format(video_path, result_dict))

    return text_results


def mp_audio_denoising_process(
    file_json,
    save_path,
    threads=2,
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    date=0,
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

        audio_path = video_path.replace("cascade_cut", "audio_files")
        audio_path = audio_path.replace(".mp4", ".wav")
        video_paths.append([video_path, audio_path])

    print("All videos loaded. {} videos in total.".format(len(video_paths)))

    # 单线程测试
    # audio_denoising((0, 0, video_paths[:1000]))

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
            audio_denoising,
            zip(range(num_threads), gpu_ids[:num_threads], video_list),
        )

    results_dict = {}
    for p in results:
        results_dict.update(p)

    print("All threads completed.")

    save_json_path = save_path + "/audio_denoising_{}.json".format(date)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")
    return save_json_path


if __name__ == "__main__":
    json_file = (
        "Select_json_for_a2p_0704_v1/Youtube_batch01_selected_videos_0704_v1.json"
    )
    save_path = "Select_json_audio_denoising"

    mp_audio_denoising_process(json_file, save_path)
