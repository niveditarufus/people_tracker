from ultralytics import YOLO

# inferencing pretrained YOLOv8 directly on the video
pretrained_model = YOLO('runs/detect/train2/weights/best.pt')
results = pretrained_model.track(source="https://www.youtube.com/watch?v=__eLCXUKtec", show=True, tracker="tracker.yaml") 