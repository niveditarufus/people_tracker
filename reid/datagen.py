import json
import os
import cv2

def return_class(category_id):
    if category_id in [1,2,3,4,5]:
        return category_id-1
    if category_id in [11]:
        return 5
    if category_id in [14]:
        return 6
    if category_id in [21]:
        return 7
    if category_id in [22]:
        return 8

with open('/home/luffy/workspace/TII/Data/Daytime/daytime.json') as f:
    train_data = json.load(f)
os.makedirs('/home/luffy/workspace/TII/reid/images/train/')
ctr_train = 0
for image_name in train_data['images']:
    print(image_name['file_name'])
    image = cv2.imread('/home/luffy/workspace/TII/Data/Daytime/IR/'+image_name['file_name'])
    for ann in train_data['annotations']:
        if image_name['id']==ann['image_id'] and ann['category_id']in [1,2,3,4,5,11,14,21,22]:
            x, y, w, h = int(ann['bbox'][0]), int(ann['bbox'][1]), int(ann['bbox'][2]), int(ann['bbox'][3])
            patch = image[y:y+h,x:x+w,:]
            patch = cv2.resize(patch, (224, 224))
            cv2.imwrite('/home/luffy/workspace/TII/reid/images/train/'+str(ctr_train)+'.jpg', patch)
            f = open("/home/luffy/workspace/TII/reid/images/annotations.csv", "a")
            print('reid/images/train/'+str(ctr_train)+'.jpg',',' ,return_class(ann['category_id']), file=f)
            f.close()
            ctr_train+=1