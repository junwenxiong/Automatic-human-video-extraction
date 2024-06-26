#  Automatic-human-video-extraction (自动人体视频提取)

1. 按镜头切割后得到的视频片段
2. 按运动信息过滤静态/低于阈值（低于10）的视频
3. 按人体信息以及文字所占图像面积（高于2.5%）过滤视频
   1. 一个视频片段只取3~7帧进行进检测识别
   2. Yolov8模型检测是否包含人体，并过滤含有多人场景以及单人所占区域面积小于20%的视频
   3. Dwpose模型检测人体是否完整出现，按预测姿态点数量分为[0,1,2,3,4] 5个等级
      1. 0 表示上半身都没有
      2. 1 表示上半身有，下半身没有    
      3. 2 表示上半身有，下半身有，但膝盖没有    
      4. 3 表示上半身有，下半身有，膝盖有，但脚踝没有 
      5. 4 表示上半身有，下半身有，膝盖有，脚踝有
   4. 人体分割模型与文本检测模型都同时执行，过滤人体分割图与文本框重叠区域占比大于10%的视频
4. 按音视同步信息是否同步进行过滤，检测唇动和语音相匹配的帧序列

### 需要下载的权重

1. download dwpose/yolox_l.onnx and dw-ll_ucoco_384.onnx, put them into ```video_process/pose_detection_model/weights``` dir.
2. download yolo_weights/yolov8x.pt and put it into ```video_process/yolo_weights``` dir.