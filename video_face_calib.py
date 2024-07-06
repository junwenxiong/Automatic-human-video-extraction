import cv2
import dlib
import cv2
import torch
from torchvision import transforms
import numpy as np
from facenet_pytorch import MTCNN

def merge_ranges(data):
    merged_ranges = []
    start_index = None
    end_index = None

    for i, value in enumerate(data):
        if value:
            if start_index is None:
                start_index = i
            end_index = i
        elif start_index is not None:
            if end_index - start_index < 5:
                end_index = i - 1
            merged_ranges.append([start_index, end_index])
            start_index = None
            end_index = None

    if start_index is not None:
        merged_ranges.append([start_index, end_index])

    return merged_ranges

def detect_frontal_faces(video_path):
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化MTCNN
    mtcnn = MTCNN(device=device)

    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    results = []
    i = 0
    while True:
        # 读取视频帧
        ret, frame = video.read()
        if not ret:
            break

        i += 1

        # 转换为RGB图像
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # 进行人脸检测和关键点定位
            boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)

        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                # 提取左眼和右眼的关键点坐标
                left_eye = landmark[0]
                right_eye = landmark[1]

                # 计算眼睛的斜率
                eye_slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])

                # 判断人脸是否面对镜头
                slope_threshold = 0.2  # 根据实际情况调整
                is_frontal = abs(eye_slope) < slope_threshold

                if is_frontal:
                    print(f"{i}, 人脸正对镜头")
                else:
                    print(f"{i}, 人脸非正对镜头")

                results.append(is_frontal)

                # 绘制人脸框和判断结果
                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Frontal: {}".format(is_frontal),
                    (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # 显示帧
        cv2.imwrite(f"tmp/Video_{i}.jpg", frame)

    # 释放资源
    video.release()

    # TODO: 如何解决返回结果的问题？
    results = merge_ranges(results)

    return results


def is_frontal_face_dlib(image):
    # 初始化dlib的人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测人脸
    faces = detector(gray)

    # 如果检测到多个人脸，只考虑第一个人脸
    if len(faces) > 0:
        face = faces[0]

        # 使用关键点检测器检测人脸关键点
        predictor = dlib.shape_predictor(
            "video_process/dlib_weight/shape_predictor_68_face_landmarks.dat"
        )
        landmarks = predictor(gray, face)

        # 获取左右眼的坐标
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # 计算眼睛的斜率
        eye_slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])

        # 如果斜率接近于0，则认为人脸是正对镜头的
        if abs(eye_slope) < 0.1:
            return True

    return False


def is_frontal_face(keypoints):
    # 检查关键点位置，判断是否正脸
    # 这里可以根据具体需求进行自定义规则，比如判断眼睛位置、正脸角度等

    # 简单示例：判断两眼的水平位置差异是否较小
    eye_distance = abs(keypoints["left_eye"][0] - keypoints["right_eye"][0])
    if eye_distance < 15:
        return True
    else:
        return False


# 打开视频文件
video_path = "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/TED/TED_videos/cascade_cut/eZj5n8ScTkI/Scene-041.mp4"  # 替换为您的视频文件路径
res = detect_frontal_faces(video_path)
print(res)

# cap = cv2.VideoCapture(video_path)

# i = 0
# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         break

#     i += 1
#     # 检测人脸是否正对镜头
#     is_frontal = detect_frontal_faces(frame)

#     if is_frontal:
#         print(f"{i}, 人脸正对镜头")
#     else:
#         print(f"{i}, 人脸非正对镜头")

# cap.release()
