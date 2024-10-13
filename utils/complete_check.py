import numpy as np
from plyfile import PlyData
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

labels_color = ([255, 255, 255],#0
                [193,221,223],#1
                [227,141,97],#2
                [224,212,147],#3
                [152,162,70],#4
                [244,213,110],#5
                [182,202,190],#6
                [216,233,248],#7
                [241,240,180],#8
                [201,90,85],#9
                [64,74,117],#10
                [95,146,140],#11
                [209,137,110],#12
                [194,155,60],#13
                [133,148,156],#14
                [162,139,135],#15
                [176,138,76])#16

def readplyFile(path):
    row_data = PlyData.read(path)  # read ply file
    R = row_data.elements[1].data['red'].reshape(-1, 1)
    G = row_data.elements[1].data['green'].reshape(-1, 1)
    B = row_data.elements[1].data['blue'].reshape(-1, 1)
    rgbs = np.concatenate((R, G, B), axis=1)
    gts = []
    for rgb in rgbs:
        label = labels_color.index(list(rgb))
        gts.append(label)
    gts = np.array(gts)
    return gts

import pymeshlab
def readplyFile2(path):
    ms = pymeshlab.MeshSet()
    ms.clear()

    # load mesh
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    # get elements
    points_ori = mesh.vertex_matrix()
    # z score
    # # move to center
    center = points_ori.min(0)
    points = points_ori - center
    # # normalize
    # max_len = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    # points /= max_len

    # 4. 获取gt
    gts=[]
    rgbs = mesh.vertex_color_matrix()*255
    for rgb in rgbs:
        label = labels_color.index(list(rgb[:3]))
        gts.append(label)
    gt_sem = np.array(gts)[:,np.newaxis]
    gt_ins = np.array(gts)[:,np.newaxis]
    pc = np.concatenate([points, gt_sem, gt_ins], axis=1)
    return pc


gt_path = '/home/lipengcheng/Codes/ThisNet/3D-BoNet/log_split2/test_res/3dbonet_test/gt/'
pred_path = '/home/lipengcheng/Codes/ThisNet/3D-BoNet/log_split2/test_res/3dbonet_test/pred/'

cnt = 0
numer_of_teeth = 0
for item in tqdm(os.listdir(pred_path)):
    if item.endswith('.ply'):
        # print('processing ', item)
        pred = readplyFile2(os.path.join(pred_path, item))[:,-1]
        gt = readplyFile2(os.path.join(gt_path, item))[:,-1]
        numer_of_teeth += len(np.unique(gt) - 1)
        # print('number of teeth:', numer_of_teeth)
        for cls in np.unique(gt):
            if cls != 0 :
                gt_mask = np.where(gt==cls,1,0)
                pred_mask = np.where(pred==cls,1,0)
                intersection = (gt_mask * pred_mask).sum(-1)
                union = gt_mask.sum(-1) + pred_mask.sum(-1) - intersection
                score = intersection / (union + 1e-6)
                # print('iou in class{} is{}'.format(cls, score))
                if score>0.90:
                    cnt+=1
                    # print('complete teeth:',cnt)

file = open('/home/lipengcheng/Codes/ThisNet/3D-BoNet/log_split2/test_res/3dbonet_test/complete_80_2_90.txt','a')
print("Complete 90 ratio in {} is: {}".format(pred_path.split('/')[-3], (cnt/numer_of_teeth)*100), file=file)
print('complete teeth:',cnt, file=file)
print('all teeth:', numer_of_teeth, file=file)
file.close()