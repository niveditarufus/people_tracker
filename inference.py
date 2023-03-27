from ultralytics import YOLO
import cv2
from collections import defaultdict
import pandas as pd

def get_time_per_person(counter_dict, ouput_path):
    time = []
    for id, frames in counter_dict.items():
        time.append([id, frames["frames"]/30.0])
    df = pd.DataFrame(time, columns = ['Person_Id', 'Time spent in video'])
    df.to_csv(ouput_path, index=0)
    print("Total number of people: ", len(df))
    return df

def inference_tracking(model_path, data_source, tracker_config="tracker.yaml", class_ids=0):
    pretrained_model = YOLO(model_path)
    results = pretrained_model.track(source=data_source, show=True, tracker=tracker_config, classes = class_ids) 
    return results

def main():
    # inferencing pretrained YOLOv8 directly on the video
    results = inference_tracking(model_path='yolov8n.pt', data_source="data/DatasetVideo.mp4")
    counter_dict = {}

    for i,r in enumerate(results):
        image = r.orig_img 
        H,W,_ = image.shape

        for det in r:
            box = det.boxes.xyxy 
            tlbr = box.cpu().numpy().astype(int).reshape((4,))
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
            
            if(det.boxes.id and patch is not None):
                id = int(det.boxes.id)
                if id in counter_dict.keys():
                    frame = counter_dict[id]["frames"] + 1
                else:
                    frame = 1
                counter_dict[id] = {"frames":frame}
                try: 
                    cv2.imwrite("./post_process/images/Person_"+str(id)+"_frame_"+str(i)+".jpg", patch)
                except:
                    pass

        cv2.imwrite("output/detection3/"+str(i)+"_.jpg", r.plot())
        # cv2.waitKey(0)


    get_time_per_person(counter_dict, "yolo_ir_trained_with_reid.csv")

if __name__ == '__main__':
    main()
