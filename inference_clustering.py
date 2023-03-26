from ultralytics import YOLO
import cv2
import torch
from torchvision import models, transforms
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


pretrained_model = YOLO('runs/detect/train2/weights/best.pt')

results = pretrained_model(source="data/DatasetVideo.mp4", stream=True)  # generator of Results objects

batch_size = 16

def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features
batch_patches = []
for r in results:
    boxes = r.boxes.xyxy  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segmenation masks outputs
    probs = r.probs  # Class probabilities for classification outputs
    image = r.orig_img 
    patches = []
    

    for i,box in enumerate(boxes):
        tlbr = box.cpu().numpy().astype(int)
        patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
        patch = cv2.resize(patch, (224,224), interpolation=cv2.INTER_LINEAR)
        patch = torch.as_tensor(patch.transpose(2, 0, 1))
        patch = patch.cuda().half()
        patches.append(patch)

        if (i + 1) % batch_size == 0:
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)
            patches = []

    if len(patches):
        patches = torch.stack(patches, dim=0)
        batch_patches.append(patches)

features = np.zeros((0, 1000))
model = models.resnet50(pretrained=True)
state_dict_model = torch.load("model/custom_model_infrared/epoch=9-step=6140.ckpt")['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict_model.items():
    if "model.model." in k:
        name = k.replace("model.model.", "", 1)
        new_state_dict[name] = v
model.load_state_dict(new_state_dict)
        
model.eval().cuda().half()
print()
for batch in batch_patches:

    emb = model(batch)
    feat = postprocess(emb)
    
    features = np.vstack((features, feat))

print(features.shape)
db = DBSCAN(eps=0.009, min_samples=30).fit(features)
print(len(db.labels_))

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
  
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
unique_labels = set(labels)
colors = ['y', 'b', 'g', 'r']
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
  
    class_member_mask = (labels == k)
  
    xy = features[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)
  
    xy = features[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)
  
plt.title('number of clusters: %d' % n_clusters_)
plt.savefig('test.png')
  

