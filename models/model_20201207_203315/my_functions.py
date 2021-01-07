import torch
import numpy as np
import cv2

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def generate_octagon_torch(output_pts):
    repeated_indices = [0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,8,9,8,9,10,11,10,11,12,13,12,13,14,15,14,15]
    output_pts_repeated = output_pts[:,repeated_indices]
    add_indices = [8,9,4,5,10,11,6,7,0,1,12,13,2,3,14,15,0,1,12,13,2,3,14,15,8,9,4,5,10,11,6,7]
    add_vector = output_pts[:, add_indices]
    return (output_pts_repeated * 3 + add_vector) / 4

def generate_octagon(key_points,with_center=False):
    numpy_key_pts = key_points.detach().numpy()
    # numpy_key_pts = key_points.numpy()
    # numpy_key_pts = key_points
    if with_center:
        bbox_pts = numpy_key_pts[:,1:] 
    else:
        bbox_pts = numpy_key_pts
    octagon = make_octagon(bbox_pts).astype(int)
    center_coords = numpy_key_pts[:,:1]
    if with_center:
        result = np.hstack((center_coords,octagon))
    else:
        result = octagon
    return torch.from_numpy(result.reshape(numpy_key_pts.shape[0],-1)).float()
    # return result.reshape(numpy_key_pts.shape[0],-1)

def generate_symmetries(key_points,with_center=False):
    numpy_key_pts = key_points.detach().numpy()
    # numpy_key_pts = key_points.numpy()
    # numpy_key_pts = key_points
    if with_center:
        bbox_pts = numpy_key_pts[:,1:] 
    else:
        bbox_pts = numpy_key_pts
    octagon = make_octagon(bbox_pts).astype(int)
    symmetries = make_octagon_symmetries(octagon)
    center_coords = np.repeat(numpy_key_pts[:,:1],8,axis=0).squeeze()
    if with_center:
        result = np.hstack((center_coords,symmetries))
    else:
        result = symmetries
    return torch.from_numpy(result.reshape(numpy_key_pts.shape[0],8,-1)).float()
    # return result.reshape(numpy_key_pts.shape[0],8,-1)

def make_repeated(key_points):
    repeated = np.repeat(key_points,2,axis=1)
    repeated = repeated.reshape(repeated.shape[0],-1)
    return repeated

def make_add_vec(key_points):
    add_indices = [4,2,5,3,0,6,1,7,0,6,1,7,4,2,5,3]
    add_vec = key_points[:,add_indices].reshape(key_points.shape[0],-1)
    return add_vec

def make_octagon(key_points):
    numpy_key_pts = key_points
    bbox_pts = numpy_key_pts
    repeated = make_repeated(bbox_pts)
    add_vec = make_add_vec(bbox_pts)
    return ((repeated * 3) + add_vec) / 4

def make_octagon_symmetries(octagon):
    rot_1 = octagon

    indices_2 = [16,17,0,1,20,21,4,5,2,3,8,9,6,7,12,13,18,19,24,25,22,23,28,29,26,27,10,11,30,31,14,15]
    rot_2 = octagon[:,indices_2]

    indices_3 = [18,19,16,17,22,23,20,21,0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31,10,11,8,9,14,15,12,13]
    rot_3 = octagon[:,indices_3]

    indices_4 = [24,25,18,19,28,29,22,23,16,17,0,1,20,21,4,5,26,27,10,11,30,31,14,15,8,9,2,3,12,13,6,7]
    rot_4 = octagon[:,indices_4]

    indices_5 = [26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21,10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5]
    rot_5 = octagon[:,indices_5]

    indices_6 = [10,11,26,27,14,15,30,31,24,25,18,19,28,29,22,23,8,9,2,3,12,13,6,7,0,1,16,17,4,5,20,21]
    rot_6 = octagon[:,indices_6]

    indices_7 = [8,9,10,11,12,13,14,15,26,27,24,25,30,31,28,29,2,3,0,1,6,7,4,5,16,17,18,19,20,21,22,23]
    rot_7 = octagon[:,indices_7]

    indices_8 = [2,3,8,9,6,7,12,13,10,11,26,27,14,15,30,31,0,1,16,17,4,5,20,21,18,19,24,25,22,23,28,29]
    rot_8 = octagon[:,indices_8]

    stacked = np.vstack((rot_1,rot_2,rot_3,rot_4,rot_5,rot_6,rot_7,rot_8))

    indices = np.arange(0,8*octagon.shape[0],octagon.shape[0])
    indices = np.tile(indices,(octagon.shape[0],1))
    constant = np.arange(octagon.shape[0]).reshape(-1,1)
    indices = indices + constant
    stacked = stacked[indices.ravel()]
    return stacked

def draw(image, points):
    bbox_3d_viz = image.copy()

    plot_list = [
        (0, 1, (0, 0, 255),1), 
        (2, 3, (0, 0, 255),3),
        (4, 5, (0, 255, 255),1),
        (6, 7, (0, 255, 255),3),
        (13, 12, (255, 0, 255),1),
        (15, 14, (255, 0, 255),3),
        (9, 8, (255, 255, 0),1),
        (11, 10, (255, 255, 0),3),

        (0, 2, (255, 0, 0),3),
        (1, 3, (255, 0, 0),3),
        (4, 6, (255, 0, 0),3),
        (5, 7, (255, 0, 0),3),
        (13, 15, (255, 0, 0),3),
        (12, 14, (255, 0, 0),3),
        (9, 11, (255, 0, 0),3),
        (8, 10, (255, 0, 0),3),

        (1, 4, (0, 255, 0),3),
        (3, 6, (0, 255, 0),3),
        (5, 13, (0, 255, 0),3),
        (7, 15, (0, 255, 0),3),
        (12, 9, (0, 255, 0),3),
        (14, 11, (0, 255, 0),3),
        (8, 0, (0, 255, 0),3),
        (10, 2, (0, 255, 0),3)]

    for to_plot in plot_list:
        p1 = (int(points[to_plot[0],0]), int(points[to_plot[0],1]))
        p2 = (int(points[to_plot[1],0]), int(points[to_plot[1],1]))
        bbox_3d_viz = cv2.line(bbox_3d_viz, p1, p2, color=to_plot[2], thickness=to_plot[3])
    return bbox_3d_viz