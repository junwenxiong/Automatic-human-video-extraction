import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import glob
import random
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model = YOLO("./video_process/yolo_weights/yolov8x.pt")
model.to(torch.device("cuda"))


# from PIL
# im1 = Image.open("/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir/random_images/0_origin.jpg")
im2 = Image.open("/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir/random_images/11_origin.jpg")

im3 = Image.open("/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir/random_images_detect_3_14/534_origin.jpg")

img_list = glob.glob(
    "/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir/random_images/*_origin.jpg"
)

im_list = [Image.open(img) for img in img_list]
print(img_list)

# results = model('/cpfs/user/xiongjunwen/workspace/Scraper/TikTokDownload/Temp_dir/random_images/0_origin.jpg', save=True)
# results = model(img_list)  # save plotted images
results = model.predict(source=im3)  # save plotted images
import ipdb; ipdb.set_trace()
# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    classes = result.boxes.cls.tolist()  # Class labels
    boxes_list = boxes.xywh.tolist()  # List of bounding boxes

    cls_count = Counter(classes)
    if 0.0 in cls_count.keys() and cls_count[0.0] > 1:
        print("person: ", cls_count[0.0])
        print(classes)

    rand_num = random.randint(0, 100)
    result.save(
        filename=f"{rand_num}_result.jpg"
    )
