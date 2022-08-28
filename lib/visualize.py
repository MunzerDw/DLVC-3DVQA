"""
Modified from https://github.com/daveredrum/ScanRefer/blob/master/scripts/visualize.py
"""

import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile
from plyfile import PlyData, PlyElement

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from utils.pc_utils import write_ply_rgb, write_oriented_bbox
from utils.box_util import get_3d_box, box3d_iou
from data.scannet.model_util_scannet import ScannetDatasetConfig
#from model.ap_helper import APCalculator, parse_predictions, parse_groundtruths
#from lib.loss_helper import get_loss
#from lib.eval_helper import get_eval
from data.config import CONF 

# data
# SCANNET_ROOT = "/mnt/canis/Datasets/ScanNet/public/v2/scans/" # TODO point this to your scannet data
# SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply") # scene_id, scene_id 
# SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt") # scene_id, scene_id 
# SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
# SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DC = ScannetDatasetConfig()

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_bbox(bbox, mode, output_file, write=True):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string

    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    def get_bbox_corners(bbox):
        centers, lengths = bbox[:3], bbox[3:6]
        xmin, xmax = centers[0] - lengths[0] / 2, centers[0] + lengths[0] / 2
        ymin, ymax = centers[1] - lengths[1] / 2, centers[1] + lengths[1] / 2
        zmin, zmax = centers[2] - lengths[2] / 2, centers[2] + lengths[2] / 2
        corners = []
        corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
        corners = np.concatenate(corners, axis=0) # 8 x 3

        return corners

    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(bbox)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    palette = {
        0: [0, 255, 0], # gt
        1: [0, 0, 255]  # pred
    }
    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)
    if write:
        write_ply(verts, colors, indices, output_file)
    return verts, colors

def read_mesh(filename):
    """ read XYZ for each vertex.
    """

    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

    return vertices, plydata['face']

def export_mesh(vertices, faces):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append(
            (
                vertices[i][0],
                vertices[i][1],
                vertices[i][2],
                vertices[i][3],
                vertices[i][4],
                vertices[i][5],
            )
        )

    vertices = np.array(
        new_vertices,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    vertices = PlyElement.describe(vertices, "vertex")
    
    return PlyData([vertices, faces])

# def align_mesh(scene_id):
#     vertices, faces = read_mesh(SCANNET_MESH.format(scene_id, scene_id))
#     for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
#         if 'axisAlignment' in line:
#             axis_align_matrix = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
#             break
    
#     # align
#     pts = np.ones((vertices.shape[0], 4))
#     pts[:, :3] = vertices[:, :3]
#     pts = np.dot(pts, axis_align_matrix.T)
#     vertices[:, :3] = pts[:, :3]

#     mesh = export_mesh(vertices, faces)

#     return mesh

def dump_results(args, data, config, gt_bbox, gt_obb, pred_bbox, pred_obb, iou):
    dump_dir = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis")
    os.makedirs(dump_dir, exist_ok=True)

    # from inputs
    ids = data['scan_idx'].detach().cpu().numpy()
    point_clouds = data['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    pcl_color = data["pcl_color"].detach().cpu().numpy()
    if args.use_color:
        pcl_color = (pcl_color * 256 + MEAN_COLOR_RGB).astype(np.int64)
    
    for i in range(batch_size):
        # basic info
        idx = ids[i]
        scene_id = data["scene_id"][idx]
        object_id = data["object_id"][idx]
        object_name = data["object_name"][idx]
        ann_id = data["ann_id"][idx]
    
        # scene_output
        scene_dump_dir = os.path.join(dump_dir, scene_id)
        if not os.path.exists(scene_dump_dir):
            os.mkdir(scene_dump_dir)

            # # Dump the original scene point clouds
            #mesh = align_mesh(scene_id)
            #mesh.write(os.path.join(scene_dump_dir, 'mesh.ply'))

            write_ply_rgb(point_clouds[i], pcl_color[i], os.path.join(scene_dump_dir, 'pc.ply'))

        #  # filter out the valid ground truth reference box
        # assert gt_ref_labels[i].shape[0] == gt_center[i].shape[0]
        # gt_ref_idx = np.argmax(gt_ref_labels[i], 0)

        # visualize the gt reference box
        # NOTE: for each object there should be only one gt reference box
        object_dump_dir = os.path.join(dump_dir, scene_id, "gt_{}_{}.ply".format(object_id, object_name))

        if not os.path.exists(object_dump_dir):
            gt_verts = write_bbox(gt_obb, 0, os.path.join(scene_dump_dir, 'gt_{}_{}.ply'.format(object_id, object_name)))

        pred_verts = write_bbox(pred_obb, 1, os.path.join(scene_dump_dir, 'pred_{}_{}_{}_{:.5f}.ply'.format(object_id, object_name, ann_id, iou)))
