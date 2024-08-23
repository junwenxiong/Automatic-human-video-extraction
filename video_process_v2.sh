export PYTHONPATH=/cpfs/user/xiongjunwen/workspace/Scraper/VideoProcess/video_process:$PYTHONPATH

day_time=0702_refined

# CUDA_VISIBLE_DEVICES=5,6,7,1 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
# --dataset TED \
# --dataset_name TEDxTalks \
# --date $day_time \
# --is_text_detection 0 \
# --is_av_consistency 0 \
# --is_camera_rotating 0 \
# --is_extract_audio 1 \
# --is_audio_denoising 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
# --dataset TED \
# --dataset_name TED_videos \
# --date $day_time \
# --is_text_detection 0 \
# --is_av_consistency 0 \
# --is_camera_rotating 0 \
# --is_extract_audio 1 \
# --extract_audio_path Select_json_for_a2p_0702_refined/TED_videos.json \
# --is_audio_denoising 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
# --dataset TED \
# --dataset_name TEDxTalks \
# --date $day_time \
# --is_text_detection 0 \
# --is_av_consistency 0 \
# --is_camera_rotating 0 \
# --is_extract_audio 1 \
# --extract_audio_path Select_json_for_a2p_0702_refined/TEDxTalks.json \
# --is_audio_denoising 1

for i in $(seq -f "%02g" 3 3)
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
    --dataset Youtube \
    --dataset_name batch_$i \
    --date $day_time \
    --is_text_detection 0 \
    --is_av_consistency 0 \
    --is_camera_rotating 0 \
    --is_extract_audio 1 \
    --extract_audio_path Select_json_for_a2p_0702_refined/Youtube_batch_$i.json \
    --is_audio_denoising 1
done

# for i in $(seq -f "%02g" 5 8)
# do
#     CUDA_VISIBLE_DEVICES=6,7,0,1 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
#     --dataset Youtube \
#     --dataset_name batch_$i \
#     --date $day_time \
#     --is_text_detection 1 \
#     --is_av_consistency 1 \
#     --is_camera_rotating 1
# done

# CUDA_VISIBLE_DEVICES=2,3,5,6 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
# --dataset Youtube \
# --dataset_name batch_01 \
# --date $day_time \
# --is_text_detection 0 \
# --is_av_consistency 0 \
# --is_camera_rotating 0 \
# --is_extract_audio 1

# CUDA_VISIBLE_DEVICES=0,1,6,7 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
# --dataset Youtube \
# --dataset_name batch_10 \
# --date $day_time \
# --is_text_detection 0 \
# --is_av_consistency 1 \
# --is_camera_rotating 1



# for i in $(seq -f "%02g" 11 11)
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_v2.py \
#     --dataset Youtube \
#     --dataset_name batch_$i \
#     --date $day_time \
#     --thread_num 8 \
#     --is_text_detection 0 \
#     --is_av_consistency 0 \
#     --is_camera_rotating 0 \
#     --is_extract_audio 0 \
#     --is_audio_denoising 1
# done

# CUDA_VISIBLE_DEVICES=4,5,6,7 /cpfs/user/lianyixin/anaconda3/envs/video_retalking/bin/python video_process_more_fine.py