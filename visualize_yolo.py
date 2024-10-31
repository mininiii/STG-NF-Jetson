from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("yolov8n-pose.pt")
results = model.track(source="/workspace/dataset/aihub_data_upload/video_data_val/C021_A19_SY32_P01_S05_01DBS.mp4", conf=0.1, iou=0.6, save=True)