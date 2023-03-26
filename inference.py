from ultralytics import YOLO

# inferencing pretrained YOLOv8 directly on the video
pretrained_model = YOLO('yolov8n.pt')
results = pretrained_model.track(source="data/DatasetVideo.mp4", show=True, tracker="tracker.yaml", classes = 0) 
