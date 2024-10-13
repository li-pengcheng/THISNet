import random
# from vedo import *
import json
import os
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
from tqdm import tqdm
import torch
from plyfile import PlyData
from scipy.spatial import distance_matrix
from random import sample
import cv2
import pymeshlab
import scipy.sparse.linalg
import scipy

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
labels_color2 = ([255, 255, 255],#0
                [74, 173, 146],#1
                [73,175,114],#2
                [118,171,71],#3
                [152,162,70],#4
                [176,157,72],#5
                [202,152,76],#6
                [230,149,118],#7
                [234,150,163],#8
                [75,171,164],#9
                [78,171,182],#10
                [83,172,205],#11
                [136,174,230],#12
                [182,168,235],#13
                [212,151,232],#14
                [231,140,215],#15
                [233,146,190])#16

def readplyFile(path):
    ms = pymeshlab.MeshSet()
    ms.clear()

    # load mesh
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    # get elements
    points = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    points_normal = mesh.vertex_normal_matrix()
    faces_center_normal = mesh.face_normal_matrix()
    # move to center
    center = points.mean(0)
    points -= center
    # normalize
    max_len = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points /= max_len

    n_face = faces.shape[0]
    x1, y1, z1 = points[faces[:, 0]][:, 0], points[faces[:, 0]][:, 1], points[faces[:, 0]][:, 2]
    x2, y2, z2 = points[faces[:, 1]][:, 0], points[faces[:, 1]][:, 1], points[faces[:, 1]][:, 2]
    x3, y3, z3 = points[faces[:, 2]][:, 0], points[faces[:, 2]][:, 1], points[faces[:, 2]][:, 2]
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    faces_points = np.concatenate((points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]), axis=1)
    faces_center_points = np.concatenate(
        (x_centre.reshape(n_face, 1), y_centre.reshape(n_face, 1), z_centre.reshape(n_face, 1)), axis=1)
    faces_normals = np.concatenate((points_normal[faces[:, 0]], points_normal[faces[:, 1]], points_normal[faces[:, 2]]),
                                   axis=1)
    # 4. get ground truth
    row_data = PlyData.read(path)  # read ply file
    R = row_data.elements[1].data['red'].reshape(-1, 1)
    G = row_data.elements[1].data['green'].reshape(-1, 1)
    B = row_data.elements[1].data['blue'].reshape(-1, 1)
    rgbs = np.concatenate((R, G, B), axis=1)
    gts = []
    gt_faces = []
    for rgb in rgbs:
        label = labels_color.index(list(rgb))
        gts.append(label)
    gts = np.array(gts)
    gts[gts==8]=0
    gts[gts==16]=0
    gt_faces = gts
    return faces, faces_points, faces_center_points, faces_normals, faces_center_normal, gt_faces

def readplyFile_testcase(path):
    ms = pymeshlab.MeshSet()
    ms.clear()

    # load mesh
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    # get elements
    points = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    points_normal = mesh.vertex_normal_matrix()
    faces_center_normal = mesh.face_normal_matrix()
    # move to center
    center = points.mean(0)
    points -= center
    # normalize
    max_len = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points /= max_len

    n_face = faces.shape[0]
    x1, y1, z1 = points[faces[:, 0]][:, 0], points[faces[:, 0]][:, 1], points[faces[:, 0]][:, 2]
    x2, y2, z2 = points[faces[:, 1]][:, 0], points[faces[:, 1]][:, 1], points[faces[:, 1]][:, 2]
    x3, y3, z3 = points[faces[:, 2]][:, 0], points[faces[:, 2]][:, 1], points[faces[:, 2]][:, 2]
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    faces_points = np.concatenate((points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]), axis=1)
    faces_center_points = np.concatenate(
        (x_centre.reshape(n_face, 1), y_centre.reshape(n_face, 1), z_centre.reshape(n_face, 1)), axis=1)
    faces_normals = np.concatenate((points_normal[faces[:, 0]], points_normal[faces[:, 1]], points_normal[faces[:, 2]]),
                                   axis=1)

    return faces, faces_points, faces_center_points, faces_normals, faces_center_normal

def generate_plyfile(index_face, point_face, normal_face, label_face, path= " "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        point_face: 3 points coordinate in a face [N, 9]
        normal_face: 3 points normal in a face [N, 9]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """
    unique_index = np.unique(index_face.flatten())  # get unique points index
    flag = np.zeros([unique_index.max()+1, 2]).astype('uint64')
    order = 0
    with open(path, "a") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(unique_index.shape[0]) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("element face " + str(index_face.shape[0]) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for i, index in enumerate(index_face):
            for j, data in enumerate(index):
                if flag[data, 0] == 0:  # if this point has not been wrote
                    xyz = point_face[i, 3*j:3*(j+1)]  # Get coordinate
                    xyz_nor = normal_face[i, 3*j:3*(j+1)]
                    f.write(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + " " + str(xyz_nor[0]) + " "
                            + str(xyz_nor[1]) + " " + str(xyz_nor[2]) + "\n")
                    flag[data, 0] = 1  # this point has been wrote
                    flag[data, 1] = order  # give point a new index
                    order = order + 1  # index add 1 for next point

        for i, data in enumerate(index_face):  # write new point index for every face
            if label_face.ndim==1:
                label_face = label_face[:,np.newaxis]
            RGB = labels_color[label_face[i, 0]]  # Get RGB value according to face label
            f.write(str(3) + " " + str(int(flag[data[0], 1])) + " " + str(int(flag[data[1], 1])) + " "
                    + str(int(flag[data[2], 1])) + " " + str(RGB[0]) + " " + str(RGB[1]) + " "
                    + str(RGB[2]) + " " + str(255) + "\n")
        f.close()


class PlyDataSet(Dataset):
    def __init__(self, plyPath):
        self.plyPath = plyPath
        self.file_list = os.listdir(plyPath)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name = self.file_list[item]
        if ".pth" in name:
            faces, faces_points, faces_center_points, faces_normals, faces_center_normal, gt_faces, name = \
                torch.load(os.path.join(self.plyPath, name))
        else:
            plyReadPath = os.path.join(self.plyPath, self.file_list[item])
            faces, faces_points, faces_center_points, faces_normals, faces_center_normal, gt_faces = readplyFile(plyReadPath)

        return faces, faces_points, faces_center_points, faces_normals, faces_center_normal, gt_faces, name

class PlyDataSet_test(Dataset):
    def __init__(self, plyPath):
        self.plyPath = plyPath
        self.file_list = os.listdir(plyPath)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name = self.file_list[item]
        if ".pth" in name:
            faces, faces_points, faces_center_points, faces_normals, faces_center_normal, gt_faces, name = \
                torch.load(os.path.join(self.plyPath, name))
        else:
            plyReadPath = os.path.join(self.plyPath, self.file_list[item])
            faces, faces_points, faces_center_points, faces_normals, faces_center_normal = readplyFile_testcase(plyReadPath)

        return faces, faces_points, faces_center_points, faces_normals, faces_center_normal, name