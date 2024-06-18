import os
import torch
import sys
os.chdir('/CLIP-Driven-Universal-Model')
if '/CLIP-Driven-Universal-Model' not in sys.path:
    sys.path.append('/CLIP-Driven-Universal-Model')
from utils.utils import get_key, dice_score, TEMPLATE, ORGAN_NAME, NUM_CLASS
data_dir = '/data/clip/result/unet/prediction'
# only list the files 
def list_files(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.pt'):
                path = os.path.join(root, file)
                name = path[len(data_dir)+1:]
                yield os.path.join(root, file),name
img_name = '01_Multi-Atlas_Labeling/label/label0010.pt'
# def get_organs(img_name):
#     return TEMPLATE[get_key(img_name)]

def update_distance_centroids_to_weight(img_name,result, distance_centroids_to_weight, counts_centroids_to_weight,device):
    organs = TEMPLATE[get_key(img_name)]
    for i in range(len(result['labels'])):
        xs = result['xs'][i][0][-1].squeeze(1)[organs, :, :, :]
        weights = result['weights'][i][0][-1].squeeze()[organs, :]
        biases = result['biases'][i][0][-1][organs]
        labels = result['labels'][i].squeeze(0)[organs, :, :, :]
        xs = xs.to(device)
        weights = weights.to(device)
        biases = biases.to(device)
        labels = labels.to(device)
        positive_points_by_organs = []
        for index in range(len(labels)):
            mask = labels[index].bool()
            filtered_tensor = xs[index][:,mask]
            if filtered_tensor.shape[0] == 0:
                continue
            # centroid = torch.mean(filtered_tensor, dim=0)
            # get distance from weight to centroids
            # distance = torch.norm(weights[index] - centroid, dim=0)
            positive_points_by_organs.append(filtered_tensor)
        centroids = []
        for positive_points_by_organ in positive_points_by_organs:
            # print(positive_points_by_organ.shape)
            if positive_points_by_organ.shape[1] <= 30:
                
                centroids.append(None)
            else:
                centroid = torch.mean(positive_points_by_organ, dim=1).to(device)
                centroids.append(centroid)
        # count = 0
        # for i in range(len(centroids)):
        #     if centroids[i] is not None:
        #         count += 1
        # print(count)
        for i in range(len(centroids)):
            if centroids[i] is None:
                continue
            for j in range(len(centroids)):
                if centroids[j] is not None:
                    
                    distance_centroids_to_weight[organs[i]][organs[j]] += torch.norm(centroids[i] - weights[j])
                    counts_centroids_to_weight[organs[i]][organs[j]] += 1

distance_centroids_to_weight = [[0 for _ in range(NUM_CLASS)] for _ in range(NUM_CLASS)]
counts_centroids_to_weight = [[0 for _ in range(NUM_CLASS)] for _ in range(NUM_CLASS)]
device = "cuda:0"
from tqdm import tqdm
for path, img_name in tqdm(list_files(data_dir)):
    result = torch.load(path)
    update_distance_centroids_to_weight(img_name,result, distance_centroids_to_weight, counts_centroids_to_weight,device)
    # store distance_centroids_to_weight and counts_centroids_to_weight
    torch.save({'distance_centroids_to_weight': distance_centroids_to_weight, 'counts_centroids_to_weight': counts_centroids_to_weight}, '/data/clip/result/unet/distance_centroids_to_weight.pt')
