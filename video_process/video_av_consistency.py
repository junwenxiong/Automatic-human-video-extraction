import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
import json

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect import open_video
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from .av_consistency_model.model.faceDetector.s3fd import S3FD
from .av_consistency_model.talkNet import talkNet

import torch.multiprocessing as mp

warnings.filterwarnings("ignore")


class VideoInfo(object):
    def __init__(
        self,
        args,
        videoPath,
        savePath,
    ) -> None:

        self.videoPath = videoPath
        self.savePath = savePath

        pyaviPath = os.path.join(savePath, "pyavi")
        pyframesPath = os.path.join(savePath, "pyframes")
        pyworkPath = os.path.join(savePath, "pywork")
        pycropPath = os.path.join(savePath, "pycrop")
        videoFilePath = os.path.join(pyaviPath, "video.avi")

        self.videoFilePath = videoFilePath
        self.pyaviPath = pyaviPath
        self.pyframesPath = pyframesPath
        self.pyworkPath = pyworkPath
        self.pycropPath = pycropPath

        # if os.path.exists(savePath):
        # rmtree(savePath)

        os.makedirs(
            pyaviPath, exist_ok=True
        )  # The path for the input video, input audio, output video
        os.makedirs(pyframesPath, exist_ok=True)  # Save all the video frames
        os.makedirs(
            pyworkPath, exist_ok=True
        )  # Save the results in this process by the pckl method
        os.makedirs(
            pycropPath, exist_ok=True
        )  # Save the detected face clips (audio+video) in this process

        # Extract video
        # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
        if args.duration == 0:
            # command = (
            #     "ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic"
            #     % (videoPath, args.nDataLoaderThread, videoFilePath)
            # )
            command = "ffmpeg -y -i {} -qscale:v 2 -threads {} -async 1 -r 25 {} -loglevel panic".format(
                os.path.abspath(videoPath),
                args.nDataLoaderThread,
                os.path.abspath(videoFilePath),
            )
        else:
            # command = (
            #     "ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic"
            #     % (
            #         videoPath,
            #         args.nDataLoaderThread,
            #         args.start,
            #         args.start + args.duration,
            #         videoFilePath,
            #     )
            # )
            command = "ffmpeg -y -i {videoPath} -qscale:v 2 -threads {nDataLoaderThread} -ss {start} -to {duration} -async 1 -r 25 {videoFilePath} -loglevel panic".format(
                videoPath=videoPath,
                nDataLoaderThread=args.nDataLoaderThread,
                start=args.start,
                duration=args.duration,
                videoFilePath=videoFilePath,
            )
        print(command)
        subprocess.call(command, shell=True, stdout=None)

        # Extract audio
        self.audioFilePath = os.path.join(pyaviPath, "audio.wav")
        # command = (
        #     "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic"
        #     % (videoFilePath, args.nDataLoaderThread, self.audioFilePath)
        # )
        command = "ffmpeg -y -i {} -qscale:a 0 -ac 1 -vn -threads {} -ar 16000 {} -loglevel panic".format(
            videoFilePath,
            args.nDataLoaderThread,
            self.audioFilePath,
        )
        subprocess.call(command, shell=True, stdout=None)

        # Extract the video frames
        # command = (
        #     "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic"
        #     % (
        #         videoFilePath,
        #         args.nDataLoaderThread,
        #         os.path.join(pyframesPath, "%06d.jpg"),
        #     )
        # )
        command = (
            "ffmpeg -y -i {} -qscale:v 2 -threads {} -f image2 {} -loglevel panic"
            .format(
                videoFilePath,
                args.nDataLoaderThread,
                os.path.join(pyframesPath, "%06d.jpg"),
            )
        )
        subprocess.call(command, shell=True, stdout=None)


def prepare_args():
    parser = argparse.ArgumentParser(
        description="TalkNet Demo or Columnbia ASD Evaluation"
    )
    parser.add_argument(
        "--pretrainModel",
        type=str,
        default="pretrain_TalkSet.model",
        help="Path for the pretrained TalkNet model",
    )

    parser.add_argument(
        "--nDataLoaderThread", type=int, default=10, help="Number of workers"
    )
    parser.add_argument(
        "--facedetScale",
        type=float,
        default=0.25,
        help="Scale factor for face detection, the frames will be scale to 0.25 orig",
    )
    parser.add_argument(
        "--minTrack", type=int, default=10, help="Number of min frames for each shot"
    )
    parser.add_argument(
        "--numFailedDet",
        type=int,
        default=10,
        help="Number of missed detections allowed before tracking is stopped",
    )
    parser.add_argument(
        "--minFaceSize", type=int, default=1, help="Minimum face size in pixels"
    )
    parser.add_argument(
        "--cropScale", type=float, default=0.40, help="Scale bounding box"
    )

    parser.add_argument(
        "--start", type=int, default=0, help="The start time of the video"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="The duration of the video, when set as 0, will extract the whole video",
    )

    parser.add_argument(
        "--evalCol",
        dest="evalCol",
        action="store_true",
        help="Evaluate on Columnbia dataset",
    )
    args = parser.parse_args()
    return args


def scene_detect(videoInfo):
    # CPU: Scene detection, output is the list of each shot's time duration
    video = open_video(videoInfo.videoFilePath)

    # videoManager = VideoManager([videoInfo.videoFilePath])
    # video.base_timecode
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    # baseTimecode = videoManager.get_base_timecode()
    # videoManager.set_downscale_factor()
    # videoManager.start()
    start_time = video.base_timecode
    end_time = video.duration

    sceneManager.detect_scenes(frame_source=video)
    sceneList = sceneManager.get_scene_list()
    savePath = os.path.join(videoInfo.pyworkPath, "scene.pckl")
    if sceneList == []:
        sceneList = [
            (start_time, end_time)
        ]
    with open(savePath, "wb") as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write(
            "%s - scenes detected %d\n" % (videoInfo.videoFilePath, len(sceneList))
        )
    return sceneList


def inference_video(args, detector, videoInfo):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    flist = glob.glob(os.path.join(videoInfo.pyframesPath, "*.jpg"))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(
            imageNumpy, conf_th=0.9, scales=[args.facedetScale]
        )
        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
            )  # dets has the frames info, bbox info, conf info
        # 多个人头的情况下，退出
        if len(dets[-1]) >= 2:
            return dets, False

    return dets, True


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f["frame"] for f in track])
            bboxes = numpy.array([numpy.array(f["bbox"]) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if (
                max(
                    numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                    numpy.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                )
                > args.minFaceSize
            ):
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks


def crop_video(args, track, cropFile, videoInfo):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(videoInfo.pyframesPath, "*.jpg"))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )  # Write video
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:  # Read the tracks
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = args.cropScale
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = numpy.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    # command = (
    #     "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
    #     % (
    #         videoInfo.audioFilePath,
    #         args.nDataLoaderThread,
    #         audioStart,
    #         audioEnd,
    #         audioTmp,
    #     )
    # )
    command = (
        "ffmpeg -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads {} -ss {} -to {} {} -loglevel panic"
        .format(
            videoInfo.audioFilePath,
            args.nDataLoaderThread,
            audioStart,
            audioEnd,
            audioTmp,
        )
    )
    subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    # command = (
    #     "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
    #     % (cropFile, audioTmp, args.nDataLoaderThread, cropFile)
    # )  # Combine audio and video file

    command = (
        "ffmpeg -y -i {}t.avi -i {} -threads {} -c:v copy -c:a copy {}.avi -loglevel panic"
        .format(cropFile, audioTmp, args.nDataLoaderThread, cropFile)
    )  # Combine audio and video file

    subprocess.call(command, shell=True, stdout=None)

    if os.path.exists(cropFile + "t.avi"):
        os.remove(cropFile + "t.avi")

    return {"track": track, "proc_track": dets}


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)  # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split("/")[-1].replace(".wav", ".npy"))
    numpy.save(featuresPath, mfcc)


def evaluate_network(files, talknet, videoInfo):
    # GPU: active speaker detection by pretrained TalkNet
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        4,
        5,
        6,
    }  # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split("/")[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(videoInfo.pycropPath, fileName + ".wav"))
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )
        video = cv2.VideoCapture(os.path.join(videoInfo.pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25,
        )
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]
        allScore = []  # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = (
                        torch.FloatTensor(
                            audioFeature[
                                i * duration * 100 : (i + 1) * duration * 100, :
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )
                    inputV = (
                        torch.FloatTensor(
                            videoFeature[
                                i * duration * 25 : (i + 1) * duration * 25, :, :
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )
                    embedA = talknet.model.forward_audio_frontend(inputA)
                    embedV = talknet.model.forward_visual_frontend(inputV)
                    embedA, embedV = talknet.model.forward_cross_attention(
                        embedA, embedV
                    )
                    out = talknet.model.forward_audio_visual_backend(embedA, embedV)
                    score = talknet.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(
            float
        )
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores, args, videoInfo):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(videoInfo.pyframesPath, "*.jpg"))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[
                max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
            ]  # average smoothing
            s = numpy.mean(s)
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(videoInfo.pyaviPath, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        vOut.write(image)
    vOut.release()
    command = (
        "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic"
        % (
            os.path.join(videoInfo.pyaviPath, "video_only.avi"),
            os.path.join(videoInfo.pyaviPath, "audio.wav"),
            args.nDataLoaderThread,
            os.path.join(videoInfo.pyaviPath, "video_out.avi"),
        )
    )
    output = subprocess.call(command, shell=True, stdout=None)


# Main function
def process(arg):

    thread_id, gpu_id, video_lists, prefix_str = arg

    args = prepare_args()

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    DET = S3FD(device=device)

    talknet = talkNet()
    pretrain_path = "./video_process/av_consistency_model/pretrained_weights/pretrain_TalkSet.model"
    talknet.loadParameters(pretrain_path)
    print("thread_id: ", thread_id, "gpu_id: ", gpu_id)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % pretrain_path)
    talknet.eval()

    results = {}
    for video_path in video_lists:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        save_path = video_dir.replace(f"{prefix_str}", f"{prefix_str}/Temp_dir2")
        save_path = os.path.join(save_path, video_name)
        print(prefix_str, save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        videoInfo = VideoInfo(args, videoPath=video_path, savePath=save_path)
        # Scene detection for the video frames
        scene = scene_detect(videoInfo)
        # Face detection for the video frames
        faces, one_person = inference_video(args, DET, videoInfo)

        if one_person and faces != []:
            # Face tracking
            allTracks, vidTracks = [], []
            for shot in scene:
                if (
                    shot[1].frame_num - shot[0].frame_num >= args.minTrack
                ):  # Discard the shot frames less than minTrack frames
                    allTracks.extend(
                        track_shot(args, faces[shot[0].frame_num : shot[1].frame_num])
                    )  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
            sys.stderr.write(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + " Face track and detected %d tracks \r\n" % len(allTracks)
            )

            # Face clips cropping
            for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
                vidTracks.append(
                    crop_video(
                        args,
                        track,
                        os.path.join(videoInfo.pycropPath, "%05d" % ii),
                        videoInfo,
                    )
                )

            # Active Speaker Detection by TalkNet
            files = glob.glob("%s/*.avi" % videoInfo.pycropPath)
            files.sort()
            scores = evaluate_network(files, talknet, videoInfo)
            flist = glob.glob(os.path.join(videoInfo.pyframesPath, "*.jpg"))
            flist.sort()
            faces = [[] for i in range(len(flist))]
            for tidx, track in enumerate(vidTracks):
                try:
                    score = scores[tidx]
                except IndexError as e:
                    print(e)
                    faces = []
                    break

                for fidx, frame in enumerate(track["track"]["frame"].tolist()):
                    s = score[
                        max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
                    ]  # average smoothing
                    s = numpy.mean(s)
                    faces[frame].append([int(float(s) >= 0)])

            if all(not sublist for sublist in faces):
                faces = []

            visualization(vidTracks, scores, args, videoInfo)
        else:
            faces = []

        print(faces)
        results[videoInfo.videoPath] = faces

    return results


def mp_av_consistency_detect_process(
    file_json, save_path, prefix_str, threads=2, gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    save_flag=0,

):
    mp.set_start_method("spawn", force=True)

    with open(
        file_json,
        "r",
    ) as f:
        data = json.load(f)

    video_paths = []
    for i, (video_path, score) in enumerate(data.items()):
        video_paths.append(video_path)

    print("All videos loaded. {} videos in total.".format(len(video_paths)))
    video_list = []
    num_threads = threads
    batch_size = len(video_paths) // num_threads
    for i in range(num_threads):
        if i == num_threads - 1:
            video_list.append(video_paths[i * batch_size :])
        else:
            video_list.append(video_paths[i * batch_size : (i + 1) * batch_size])

    prefix_str_list = [prefix_str] * num_threads

    with mp.Pool(num_threads) as pool:
        results = pool.map(
            process,
            zip(range(num_threads), gpu_ids[:num_threads], video_list, prefix_str_list),
        )
    results_dict = {}
    for p in results:
        results_dict.update(p)

    print("All threads completed.")

    if save_flag == 0:
        save_json_path = save_path + "/av_consistency.json"
    else:
        save_json_path = save_path + f"/av_consistency_{save_flag}.json"

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Detected results saved to {save_json_path}")

    return save_json_path
