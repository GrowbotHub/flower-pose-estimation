import torch
import numpy as np
import cv2
from my_constants import IMAGE_HEIGHT, IMAGE_WIDTH, CAMERA_YFOV, CAMERA_ZNEAR
import math
from scipy.spatial.transform import Rotation as R

def generate_octagon(output_pts):

    """Efficiently generates octagon bounding boxes from a set
    of quadrilateral bounding boxes. 

    Parameters
    ----------
    key_points : (B,16) float `torch.Tensor` or `numpy.ndarray`
        Tensor or numpy array of pixel values representing a 
        quadrilateral bounding box. B is the number of bounding 
        boxes in order to allow for several octagon bounding boxes
        to be generated for different quadrilateral bounding
        boxes all input at the same time.

    Returns
    -------
    octagon : (B,32) float `torch.Tensor` or `numpy.ndarray`
        The octagon bounding boxes of all input quadrilateral 
        bounding boxes. The first Tensor (or numpy array) is 
        the octagon bounding box of the first input quadrilateral 
        bounding box, the next is the octagon bounding box of 
        the second input bounding box, and so on. 
    """

    # if isinstance(output_pts, np.ndarray):
    #     output_pts = output_pts.copy()
    # elif isinstance(output_pts, torch.Tensor):
    #     output_pts = output_pts.detach().clone()
    # else:
    #     assert False, "key_points must be either numpy array or Tensor!"

    repeated_indices = [0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,\
                        8,9,8,9,10,11,10,11,12,13,12,13,14,15,14,15]
    output_pts_repeated = output_pts[:,repeated_indices]
    add_indices = [8,9,4,5,10,11,6,7,0,1,12,13,2,3,14,15,\
                   0,1,12,13,2,3,14,15,8,9,4,5,10,11,6,7]
    add_vector = output_pts[:, add_indices]
    return (output_pts_repeated * 3 + add_vector) / 4

def generate_symmetries(key_points):

    """Generates octagon bounding box symmetries from a set
    of quadrilateral bounding boxes.

    Parameters
    ----------
    key_points : (B,16) float `torch.Tensor` or `numpy.ndarray`
        Tensor of pixel values representing a quadrilateral
        bounding box. B is the number of bounding boxes 
        in order to allow for several octagon bounding box 
        symmetries to be generated for different quadrilateral bounding
        boxes all input at the same time.

    Returns
    -------
    symmetries : (B,8,32) float `torch.Tensor` or `numpy.ndarray`
        All symmetries of all input quadrilateral bounding boxes.
        The first Tensor or array contains the 8 symmetries of the
        first input bounding box, the next Tensor or array contains
        the 8 symmetries of the second input bounding box, and so on. 
    """

    if isinstance(key_points, np.ndarray):
        is_tensor = False
        numpy_key_pts = key_points
    elif isinstance(key_points, torch.Tensor):
        is_tensor = True
        numpy_key_pts = key_points.detach().numpy()
    else:
        assert False, "key_points must be either numpy array or Tensor!"

    octagon = generate_octagon(numpy_key_pts)
    symmetries = make_octagon_symmetries(octagon)

    if is_tensor:
        return torch.from_numpy(symmetries).float()
    else:
        return symmetries

def generate_symmetries_quad(key_points):

    """Generates quadrilateral bounding box 
    symmetries from a set of quadrilateral bounding boxes.

    Parameters
    ----------
    key_points : (B,16) float `torch.Tensor` or `numpy.ndarray`
        Tensor of pixel values representing a quadrilateral
        bounding box. B is the number of bounding boxes 
        in order to allow for several quadrilateral bounding box 
        symmetries to be generated for different quadrilateral bounding
        boxes all input at the same time.

    Returns
    -------
    symmetries : (B,4,16) float `torch.Tensor` or `numpy.ndarray`
        All symmetries of all input quadrilateral bounding boxes.
        The first Tensor or array contains the 4 symmetries of the
        first input bounding box, the next Tensor or array contains 
        the 4 symmetries of the second input bounding box, and so on. 
    """

    if isinstance(key_points, np.ndarray):
        is_tensor = False
        numpy_key_pts = key_points
    elif isinstance(key_points, torch.Tensor):
        is_tensor = True
        numpy_key_pts = key_points.detach().numpy()
    else:
        assert False, "key_points must be either numpy array or Tensor!"

    symmetries = make_quad_symmetries(numpy_key_pts)

    if is_tensor:
        return torch.from_numpy(symmetries).float()
    else:
        return symmetries

def make_octagon_symmetries(octagon):

    """Generates octagon bounding box symmetries from a set
    of octagon bounding boxes. An octagon bounding box is
    considered to have 8 rotational symmetries around
    its vertical axis. The first rotation is simply the 
    original octagon bounding box itself.

    Parameters
    ----------
    octagon : (B,32) float or int `numpy.ndarray`
        Array of pixel values representing an octagon
        bounding box. B is the number of bounding boxes 
        in order to allow for several sets of symmetries 
        to be generated for different bounding boxes all 
        input at the same time.

    Returns
    -------
    stacked : (B,8,32) float `numpy.ndarray`
        All symmetries of all input quadrilateral bounding boxes.
        The first array contains the 8 symmetries of the
        first input bounding box, the next array contains the 8 
        symmetries of the second input bounding box, and so on. 
    """

    rot_1 = octagon

    indices_2 = [16,17,0,1,20,21,4,5,2,3,8,9,6,7,12,13,\
                 18,19,24,25,22,23,28,29,26,27,10,11,30,31,14,15]
    rot_2 = octagon[:,indices_2]

    indices_3 = [18,19,16,17,22,23,20,21,0,1,2,3,4,5,6,7,\
                 24,25,26,27,28,29,30,31,10,11,8,9,14,15,12,13]
    rot_3 = octagon[:,indices_3]

    indices_4 = [24,25,18,19,28,29,22,23,16,17,0,1,20,21,4,5,\
                 26,27,10,11,30,31,14,15,8,9,2,3,12,13,6,7]
    rot_4 = octagon[:,indices_4]

    indices_5 = [26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21,\
                 10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5]
    rot_5 = octagon[:,indices_5]

    indices_6 = [10,11,26,27,14,15,30,31,24,25,18,19,28,29,22,23,\
                 8,9,2,3,12,13,6,7,0,1,16,17,4,5,20,21]
    rot_6 = octagon[:,indices_6]

    indices_7 = [8,9,10,11,12,13,14,15,26,27,24,25,30,31,28,29,\
                 2,3,0,1,6,7,4,5,16,17,18,19,20,21,22,23]
    rot_7 = octagon[:,indices_7]

    indices_8 = [2,3,8,9,6,7,12,13,10,11,26,27,14,15,30,31,\
                 0,1,16,17,4,5,20,21,18,19,24,25,22,23,28,29]
    rot_8 = octagon[:,indices_8]

    stacked = np.vstack((rot_1,rot_2,rot_3,rot_4,
                         rot_5,rot_6,rot_7,rot_8))

    indices = np.arange(0,8*octagon.shape[0],octagon.shape[0])
    indices = np.tile(indices,(octagon.shape[0],1))
    constant = np.arange(octagon.shape[0]).reshape(-1,1)
    indices = indices + constant
    stacked = stacked[indices.ravel()]
    return stacked.reshape(octagon.shape[0],8,32)

def make_quad_symmetries(quadrilateral):

    """Generates quadrilateral bounding box symmetries 
    from a set of quadrilateral bounding boxes. A 
    quadrilateral bounding box is considered to have 4
    rotational symmetries around its vertical axis. 
    The first rotation is simply the original 
    quadrilateral bounding box itself.

    Parameters
    ----------
    octagon : (B,16) float or int `numpy.ndarray`
        Array of pixel values representing a quadrilateral
        bounding box. B is the number of bounding boxes 
        in order to allow for several sets of symmetries 
        to be generated for different bounding boxes all 
        input at the same time.

    Returns
    -------
    stacked : (B,4,16) float `numpy.ndarray`
        All symmetries of all input quadrilateral bounding boxes.
        The first array contains the 4 symmetries of the
        first input bounding box, the next array contains the 4 
        symmetries of the second input bounding box, and so on. 
    """

    rot_1 = quadrilateral

    indices_2 = [8,9,10,11,0,1,2,3,12,13,14,15,4,5,6,7]
    rot_2 = quadrilateral[:,indices_2]

    indices_3 = [12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3]
    rot_3 = quadrilateral[:,indices_3]

    indices_4 = [4,5,6,7,12,13,14,15,0,1,2,3,8,9,10,11]
    rot_4 = quadrilateral[:,indices_4]

    stacked = np.vstack((rot_1,rot_2,rot_3,rot_4,))

    indices = np.arange(0,4*quadrilateral.shape[0],quadrilateral.shape[0])
    indices = np.tile(indices,(quadrilateral.shape[0],1))
    constant = np.arange(quadrilateral.shape[0]).reshape(-1,1)
    indices = indices + constant
    stacked = stacked[indices.ravel()]
    return stacked.reshape(quadrilateral.shape[0],4,16)

def draw(image, points,black=False,ID=False,fname=None):

    """Draws a quadrilateral or octagon bounding box
    on a copy of the input image.

    Parameters
    ----------
    image : (H,W,3) float `numpy.ndarray`
        Image as an array with height H and width W. The three 
        color channels are expected to be in BGR format (as opposed
        to RGB). See cv2.cvtColor()
    points : (N,2) float or int `numpy.ndarray`
        Array of points to draw. N is the number of points.
        N = 16 for quadrilateral or N = 32 for octagon.
    black : bool
        If 'True' a different color variant is chosen.
    id : bool
        If 'True' each pixel will be numbered on the image.
    id : string
        If filename is specified it will be written in the bottom
        left corner.

    Returns
    -------
    image : (H,W,3) float `numpy.ndarray`
        A copy of the image with the bounding box drawn on it.
    """

    bbox_3d_viz = image.copy()

    plot_list_quad = [
        (0, 1, (255, 0, 0),3), 
        (0, 2, (0, 0, 255),2),
        (0, 4, (0, 255, 0),2), 
        (1, 3, (0, 0, 255),3),
        (1, 5, (0, 255, 0),3),
        (2, 3, (255, 0, 0),3),
        (2, 6, (0, 255, 0),2),
        (3, 7, (0, 255, 0),3), 
        (4, 5, (255, 0, 0),3),
        (4, 6, (0, 0, 255),2),
        (5, 7, (0, 0, 255),3), 
        (6, 7, (255, 0, 0),3)]

    plot_list_oct = [
        # (0, 1, (0, 0, 255),1), 
        # (2, 3, (0, 0, 255),3),
        # (4, 5, (0, 255, 255),1),
        # (6, 7, (0, 255, 255),3),
        # (13, 12, (255, 0, 255),1),
        # (15, 14, (255, 0, 255),3),
        # (9, 8, (255, 255, 0),1),
        # (11, 10, (255, 255, 0),3),

        (0, 1, (0, 0, 255),1), 
        (2, 3, (0, 0, 255),3),
        (4, 5, (0, 0, 255),1),
        (6, 7, (0, 0, 255),3),
        (13, 12, (0, 0, 255),1),
        (15, 14, (0, 0, 255),3),
        (9, 8, (0, 0, 0),1),
        (11, 10, (0, 0, 0),3),

        (0, 2, (255, 0, 0),3),
        (1, 3, (255, 0, 0),3),
        (4, 6, (255, 0, 0),3),
        (5, 7, (255, 0, 0),3),
        (13, 15, (255, 0, 0),3),
        (12, 14, (255, 0, 0),3),
        (9, 11, (255, 0, 0),3),
        (8, 10, (255, 0, 0),3),

        (1, 4, (0, 255, 0),1),
        (3, 6, (0, 255, 0),3),
        (5, 13, (0, 255, 0),1),
        (7, 15, (0, 255, 0),3),
        (12, 9, (0, 255, 0),1),
        (14, 11, (0, 255, 0),3),
        (8, 0, (0, 255, 0),1),
        (10, 2, (0, 255, 0),3)]

    plot_list_quad_black = [
        (0, 1, (203,192,255),3), 
        (0, 2, (255, 255, 0),2),
        (0, 4, (255, 255, 0),2), 
        (1, 3, (255, 255, 0),3),
        (1, 5, (255, 255, 0),3),
        (2, 3, (203,192,255),3),
        (2, 6, (255, 255, 0),2),
        (3, 7, (255, 255, 0),3), 
        (4, 5, (203,192,255),3),
        (4, 6, (255, 255, 0),2),
        (5, 7, (255, 255, 0),3), 
        (6, 7, (203,192,255),3)]

    plot_list_oct_black = [
        (0, 1, (255, 255, 0),1), 
        (2, 3, (255, 255, 0),3),
        (4, 5, (255, 255, 0),1),
        (6, 7, (255, 255, 0),3),
        (13, 12, (255, 255, 0),1),
        (15, 14, (255, 255, 0),3),
        (9, 8, (255, 255, 0),1),
        (11, 10, (255, 255, 0),3),

        (0, 2, (203,192,255),3),
        (1, 3, (203,192,255),3),
        (4, 6, (203,192,255),3),
        (5, 7, (203,192,255),3),
        (13, 15, (203,192,255),3),
        (12, 14, (203,192,255),3),
        (9, 11, (203,192,255),3),
        (8, 10, (203,192,255),3),

        (1, 4, (255, 255, 0),1),
        (3, 6, (255, 255, 0),3),
        (5, 13, (255, 255, 0),1),
        (7, 15, (255, 255, 0),3),
        (12, 9, (255, 255, 0),1),
        (14, 11, (255, 255, 0),3),
        (8, 0, (255, 255, 0),1),
        (10, 2, (255, 255, 0),3)]
    
    if points.shape[0] == 8:
        if black:
            plot_list = plot_list_quad_black
        else:
            plot_list = plot_list_quad

    else:
        if black:
            plot_list = plot_list_oct_black
        else:
            plot_list = plot_list_oct

    for to_plot in plot_list:
        p1 = (int(points[to_plot[0],0]), 
              int(points[to_plot[0],1]))
        p2 = (int(points[to_plot[1],0]), 
              int(points[to_plot[1],1]))
        bbox_3d_viz = cv2.line(bbox_3d_viz, 
                               p1, p2, 
                               color=to_plot[2], 
                               thickness=to_plot[3])

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if ID:
        for i,point in enumerate(points):
            bbox_3d_viz = cv2.putText(bbox_3d_viz,str(i),
                                    (int(point[0]),int(point[1])), 
                                    font, 0.5,(150,150,150),2)

    if fname:
        bbox_3d_viz = cv2.putText(bbox_3d_viz,fname,
                                    (0,599), 
                                    font, 0.5,(150,150,150),2)


    return bbox_3d_viz

def rotate_box(box,rotations):

    """Rotates a single set of pixel points counter-clockwise. 

    Parameters
    ----------
    box : (N,2) `numpy.ndarray`
        Array of points to rotate. N is the number of points.
    rotations : int 
        Number of times to rotate the points. The points are 
        rotated by rotations * 90 degrees. rotations must be
        in the set {0,1,2,3}.

    Returns
    -------
    box : (N,2) `numpy.ndarray`
        Array of points afer rotation. N is the number of points.
    """

    assert rotations in [0,1,2,3]
    if rotations == 1:
        box[:,[0,1]] = box[:,[1,0]]
        box[:,1] = (IMAGE_WIDTH-1)-box[:,1]
    elif rotations == 2:
        maxes = np.array([IMAGE_WIDTH-1,IMAGE_HEIGHT-1])
        box = maxes - box
    elif rotations == 3:
        box[:,[0,1]] = box[:,[1,0]]
        box[:,0] = (IMAGE_HEIGHT-1)-box[:,0]
    else:
        box = box
    return box

def sigmoid_wide_interval(input):

    """Applies a custom sigmoid function function element-wise
    so that the output lies in the interval [-0.1, 1.1].

    Parameters
    ----------
    input : `torch.Tensor`
        Tensor to apply the function to.

    Returns
    -------
    out : `torch.Tensor`
        Tensor with the function applied to it.
    """

    return ((1.2 / (1 + torch.exp(-input))) - 0.1)

def sigmoid_regular_interval(input):

    """Applies the regular sigmoid function function element-wise
    so that the output lies in the interval [0.0, 1.0].
    
    Parameters
    ----------
    input : `torch.Tensor`
        Tensor to apply the function to.

    Returns
    -------
    out : `torch.Tensor`
        Tensor with the function applied to it.
    """
    
    return (torch.sigmoid(input))

def identity(input):

    """Returns the input without modifying it.
    
    Parameters
    ----------
    input : `torch.Tensor`
        Tensor to apply the identity function to.

    Returns
    -------
    out : `torch.Tensor`
        Tensor with the identity function applied to it.
    """
    
    return input

def S_Loss(output_pts, symmetries, metric):

    """Computes the S_Loss between a predicted bounding box 
    and the ground truth bounding box's symmetries. 
    The S_Loss is defined as the minimum pixel distance 
    between the predicted bounding box and one of the 
    symmetries. The distance is computed usingthe specified 
    metric (either MSE, RMSE, MAE).

    If bounding boxes are passed in batches then the mean
    accross the minimum for each bounding box is returned.

    Parameters
    ----------
    output_pts : (B,1,N) `torch.Tensor` or `numpy.ndarray`
        Predicted bounding box. B is the number of bounding
        boxes in a batch and N/2 is the number of points 
        defining the bounding box. Ex: N = 16 for quadrilateral
        bounding box and N = 32 for octagon bounding box.
    symmetries : (B,8,N) `torch.Tensor` or `numpy.ndarray`
        Symmetries of the ground truth bounding box. 
    metric : str
        Selects metric. Either 'MSE', 'RMSE' or 'MAE'.

    Returns
    -------
    symmetry_loss : int
        S_Loss
    """

    if (isinstance(output_pts, torch.Tensor) 
        and isinstance(symmetries, torch.Tensor)):
        isTensor = True
    elif (isinstance(output_pts, np.ndarray) 
          and isinstance(symmetries, np.ndarray)):
        isTensor = False
    else:
        assert False, "Inputs to loss functions must match \
                       and be either Tensors or numpy arrays"          

    diff = output_pts - symmetries

    if metric == 'MSE' or metric == 'RMSE':
        symmetry_loss = diff**2
    elif metric == 'MAE':
        if isTensor:
            symmetry_loss = torch.abs(diff)
        else: 
            symmetry_loss = np.abs(diff)

    if isTensor:

        symmetry_loss = torch.mean(symmetry_loss,dim=2)
        if metric == 'RMSE':
            symmetry_loss = torch.sqrt(symmetry_loss)
        symmetry_loss = torch.min(symmetry_loss, dim=1).values
        symmetry_loss = torch.mean(symmetry_loss)

    else:

        symmetry_loss = np.mean(symmetry_loss,axis=2)
        if metric == 'RMSE':
            symmetry_loss = np.sqrt(symmetry_loss)
        symmetry_loss = np.min(symmetry_loss, axis=1)
        symmetry_loss = np.mean(symmetry_loss)  
 
    return symmetry_loss  

def S_Loss_MSE(output_pts, symmetries):

    """Alias for S_Loss(output_pts, symmetries, 'MSE')
    used during training for speedup.

    Computes the MeanSquaredError (MSE) between each bounding
    box and each of its respective symmetries. The resulting 
    minimums for each bounding box are then averaged.

    Parameters
    ----------
    output_pts : (B,1,N) `torch.Tensor`
        Predicted bounding box. B is the number of bounding
        boxes in a batch and N/2 is the number of points 
        defining the bounding box. Ex: N = 16 for quadrilateral
        bounding box and N = 32 for octagon bounding box.
    symmetries : (B,8,N) `torch.Tensor`
        Symmetries of the ground truth bounding box. 

    Returns
    -------
    symmetry_loss : int
        MeanSquaredError (MSE) variant of S_Loss
    """

    symmetry_loss = (output_pts - symmetries)**2
    symmetry_loss = torch.mean(symmetry_loss,dim=2)
    symmetry_loss = torch.min(symmetry_loss, dim=1).values
    symmetry_loss = torch.mean(symmetry_loss)

    return symmetry_loss

def S_Loss_RMSE(output_pts, symmetries):

    """Alias for S_Loss(output_pts, symmetries, 'RMSE')
    used during training for speedup.

    Computes the RootMeanSquaredError (RMSE) between each bounding
    box and each of its respective symmetries. The resulting 
    minimums for each bounding box are then averaged.

    Parameters
    ----------
    output_pts : (B,1,N) `torch.Tensor`
        Predicted bounding box. B is the number of bounding
        boxes in a batch and N/2 is the number of points 
        defining the bounding box. Ex: N = 16 for quadrilateral
        bounding box and N = 32 for octagon bounding box.
    symmetries : (B,8,N) `torch.Tensor`
        Symmetries of the ground truth bounding box. 

    Returns
    -------
    symmetry_loss : int
        RootMeanSquaredError (RMSE) variant of S_Loss
    """

    symmetry_loss = (output_pts - symmetries)**2
    symmetry_loss = torch.mean(symmetry_loss,dim=2)
    symmetry_loss = torch.sqrt(symmetry_loss)
    symmetry_loss = torch.min(symmetry_loss, dim=1).values
    symmetry_loss = torch.mean(symmetry_loss)

    return symmetry_loss

def S_Loss_MAE(output_pts, symmetries):

    """Alias for S_Loss(output_pts, symmetries, 'MAE')
    used during training for speedup.

    Computes the MeanAbsoluteError (MAE) between each bounding
    box and each of its respective symmetries. The resulting 
    minimums for each bounding box are then averaged.

    Parameters
    ----------
    output_pts : (B,1,N) `torch.Tensor`
        Predicted bounding box. B is the number of bounding
        boxes in a batch and N/2 is the number of points 
        defining the bounding box. Ex: N = 16 for quadrilateral
        bounding box and N = 32 for octagon bounding box.
    symmetries : (B,8,N) `torch.Tensor`
        Symmetries of the ground truth bounding box. 

    Returns
    -------
    symmetry_loss : int
        MeanAbsoluteError (MAE) variant of S_Loss
    """

    symmetry_loss = symmetry_loss = torch.abs(output_pts - symmetries)
    symmetry_loss = torch.mean(symmetry_loss,dim=2)
    symmetry_loss = torch.min(symmetry_loss, dim=1).values
    symmetry_loss = torch.mean(symmetry_loss)

    return symmetry_loss

def get_quadrilateral_from_octagon(octagon,closest_to=None):

    """Computes the quadrilateral bounding box
    from which the given octagon bounding box 
    could have been generated.

    Parameters
    ----------
    octagon : (32,) float or int `torch.Tensor` or `numpy.ndarray`
        Tensor or array of pixel values representing an octagon
        bounding box.

    Returns
    -------
    quadrilateral : (16,) float or int `torch.Tensor` or `numpy.ndarray`
        Tensor or array of pixel values representing the quadrilateral
        bounding box that the input octagon bounding box could have 
        been generated from.
    """

    if isinstance(octagon, np.ndarray):
        is_tensor = False
        numpy_octagon = octagon
    elif isinstance(octagon, torch.Tensor):
        is_tensor = True
        numpy_octagon = octagon.detach().numpy()
    else:
        assert False, "octagon must be either numpy array or Tensor!"

    a = (numpy_octagon[0], numpy_octagon[1])
    b = (numpy_octagon[2], numpy_octagon[3])
    A = (numpy_octagon[4], numpy_octagon[5])
    B = (numpy_octagon[6], numpy_octagon[7])

    c = (numpy_octagon[8], numpy_octagon[9])
    d = (numpy_octagon[10], numpy_octagon[11])
    C = (numpy_octagon[12], numpy_octagon[13])
    D = (numpy_octagon[14], numpy_octagon[15])

    e = (numpy_octagon[16], numpy_octagon[17])
    f = (numpy_octagon[18], numpy_octagon[19])
    E = (numpy_octagon[20], numpy_octagon[21])
    F = (numpy_octagon[22], numpy_octagon[23])

    g = (numpy_octagon[24], numpy_octagon[25])
    h = (numpy_octagon[26], numpy_octagon[27])
    G = (numpy_octagon[28], numpy_octagon[29])
    H = (numpy_octagon[30], numpy_octagon[31])

    ae = get_line(a,e)
    cb = get_line(c,b) 
    dh = get_line(d,h)
    gf = get_line(g,f)

    AE = get_line(A,E)
    CB = get_line(C,B)    
    DH = get_line(D,H)
    GF = get_line(G,F)

    x1 = get_intersection(ae,cb)
    x2 = get_intersection(cb,dh)
    x3 = get_intersection(ae,gf)
    x4 = get_intersection(dh,gf)

    X1 = get_intersection(AE,CB)
    X2 = get_intersection(CB,DH)
    X3 = get_intersection(AE,GF)
    X4 = get_intersection(DH,GF)

    quadrilateral_1 = np.array([[
        x1[0],x1[1],X1[0],X1[1],
        x2[0],x2[1],X2[0],X2[1],
        x3[0],x3[1],X3[0],X3[1],
        x4[0],x4[1],X4[0],X4[1]
    ]]).astype('float')

    ab = get_line(a,b)
    cd = get_line(c,d) 
    hg = get_line(h,g)
    fe = get_line(f,e)

    AB = get_line(A,B)
    CD = get_line(C,D)    
    HG = get_line(H,G)
    FE = get_line(F,E)

    x1 = get_intersection(ab,cd)
    x2 = get_intersection(cd,hg)
    x3 = get_intersection(fe,ab)
    x4 = get_intersection(hg,fe)

    X1 = get_intersection(AB,CD)
    X2 = get_intersection(CD,HG)
    X3 = get_intersection(FE,AB)
    X4 = get_intersection(HG,FE)

    quadrilateral_2 = np.array([[
        x1[0],x1[1],X1[0],X1[1],
        x2[0],x2[1],X2[0],X2[1],
        x3[0],x3[1],X3[0],X3[1],
        x4[0],x4[1],X4[0],X4[1]
    ]]).astype('float')

    if closest_to is not None:
        loss_1 = S_Loss(quadrilateral_1, closest_to, 'MSE')
        loss_2 = S_Loss(quadrilateral_2, closest_to, 'MSE')
        quadrilateral = quadrilateral_1 if loss_1 < loss_2 else quadrilateral_2
    else: 
        quadrilateral = quadrilateral_1

    quadrilateral = quadrilateral.ravel()

    if is_tensor:
        return torch.from_numpy(quadrilateral).float()
    else:
        return quadrilateral

def get_line(p1,p2):

    """Computes the equation of the line
    that passes through two points. 
    Returns the slope and intercept as a tuple
    such that y = slope * x + intercept is
    the line equation.

    Parameters
    ----------
    p1 : 2-tuple of int
        Pair of (x,y) of the first point
    p2 : 2-tuple of int
        Pair of (x,y) of the second point 

    Returns
    -------
    (slope,intercept) : 2-tuple
        Slope and intercept defining the line.
    """

    if math.isclose(p2[0],p1[0]):
        slope = 'v'
        intercept = p1[0]
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p2[1] - (slope * p2[0])
    return (slope, intercept)

def get_intersection(l1,l2):

    """Computes the intersection of two lines.
    If the lines are (almost) parallel an exception
    is raised.

    Parameters
    ----------
    l1 : 2-tuple of int
        Pair of (slope,intercept) of the first line
    l2 : 2-tuple of int
        Pair of (slope,intercept) of the second line

    Returns
    -------
    (x,y) : 2-tuple
        Intersection of the two lines.
    """

    if ( (isinstance(l1[0], str) and isinstance(l2[0], str)) and (l1[0] == 'v' and l2[0] == 'v') ) or ( (not isinstance(l1[0], str)) and (not isinstance(l2[0], str)) and math.isclose(l1[0], l2[0]) ):
        raise Exception('No intersection between parallel lines!')

    # if l1[0] == 'v' and l2[0] == 'v' or (not isinstance(l1[0], str) and not isinstance(l2[0], str) and math.isclose(l1[0], l2[0])):
    #     raise Exception('No intersection between parallel lines!')

    elif l1[0] == 'v':
        x = l1[1]
        y = l2[0] * x + l2[1]

    elif l2[0] == 'v':
        x = l2[1]
        y = l1[0] * x + l1[1]

    else:
        x = (l2[1] - l1[1]) / (l1[0] - l2[0])
        y = l1[0] * x + l1[1]

    return x, y

def get_campose_from_rotation_and_translation(rvec,tvec):
    rmat, _ = cv2.Rodrigues(rvec)
    intermediate = np.hstack((rmat,tvec))
    last_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    return np.vstack((intermediate,last_row))

def get_3D_from_2D(rvec,tvec,point2D,points3D,znear=CAMERA_ZNEAR,yfov=CAMERA_YFOV):
    assert points3D.shape == (3,3)
    campose = get_campose_from_rotation_and_translation(rvec,tvec)
    inv_campose = np.linalg.inv(campose)
    x, y = point2D
    a, b, c, d = inv_campose
    t = np.tan(yfov / 2.0)
    n = znear
    o, p, q, r = get_plane_equation_from_points(points3D) 
    tH = t * (2.0*y / IMAGE_HEIGHT) - 1 
    tW = t * (2.0*x / IMAGE_WIDTH) - 1
    tHcb1 = tH * c[0] - b[0]
    tHcb2 = tH * c[1] - b[1]
    tHcb3 = tH * c[2] - b[2]
    tWca1 = tW * c[0] + a[0]
    tWca2 = tW * c[1] + a[1]
    tWca3 = tW * c[2] + a[2]


    # Y's numerator
    Ynumpart1 = -a[3] - tW * (c[3] + 2*n) 

    Ynumpart2_num = (b[3] - (tH * (c[3] + 2*n)) - r * tHcb3) * (tWca1 + (o/q) * tWca3)
    Ynumpart2_denom = tHcb1 + ((o/q) * tHcb3)

    Ynumpart2 = Ynumpart2_num / Ynumpart2_denom

    Ynum = Ynumpart1 - Ynumpart2

    # Y's denominator
    Ydenompart1 = tWca2 + ((p/q) * tWca3)

    Ydenompart2_num = (tHcb2 + ((p/q) * tHcb3)) * (tWca1 + ((o/q) * tWca3))
    Ydenompart2_denom = tHcb1 + ((o/q) * tHcb3)

    Ydenompart2 = Ydenompart2_num / Ydenompart2_denom
    
    Ydenom = Ydenompart1 - Ydenompart2

    # Y
    Y = Ynum / Ydenom

    # X's numerator
    numX = b[3] - (tH * (c[3] + 2*n)) - (r * tHcb3) - (Y * (tHcb2 + (p/q) * tHcb3))

    # X's denominator
    denomX = tHcb1 + ((o/q) * tHcb3)

    # X
    X = numX / denomX

    # Z
    Z = (o * X + p * Y + r) / q

    return np.array([[X,Y,Z]])

def get_plane_equation_from_points(points):
    # assert points.shape[1] == 3
    # if points.shape[0] > 3:
    #     points_cpy = points[:3,:]

    assert points.shape == (3,3)

    p1, p2, p3 = points
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3

    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 

    return a, b, c, d


def get_points_from_bbox3D(bbox_3d):

    bbox_3d_copy = bbox_3d.copy()
    if bbox_3d_copy.shape == (8,3):
        bbox_3d_copy = bbox_3d_copy.ravel()

    bottom_1 = bbox_3d_copy[:3]
    bottom_2 = bbox_3d_copy[6:9]
    bottom_3 = bbox_3d_copy[12:15]

    upper_1 = bbox_3d_copy[3:6]
    upper_2 = bbox_3d_copy[9:12]
    upper_3 = bbox_3d_copy[15:18]

    bottom = np.vstack((bottom_1,bottom_2,bottom_3))
    upper = np.vstack((upper_1, upper_2, upper_3))

    return bottom, upper

def get_rotation_axis(bottom,upper):
    center_top = np.mean(upper, axis=0)
    center_bottom = np.mean(bottom, axis=0)
    rotation_axis = center_top - center_bottom
    print("top: {}".format(center_top))
    print("bottom: {}".format(center_bottom))
    print("axis: {}".format(rotation_axis.ravel()))
    return rotation_axis.ravel()

def rodrigues_rotation_one_point(point,theta,k):
    
    K = np.array(
        [[0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(theta)*K + (1-np.cos(theta))*(np.matmul(K,K))

    print("R:{}".format(R))
    print("Point: {}".format(point.reshape(-1,1)))
    rotated = np.matmul(R,point.reshape(-1,1))
    return rotated.ravel()

def rodrigues_rotation(points,theta,bbox_3d):
    rotated_points = np.zeros(points.shape)
    bottom, upper = get_points_from_bbox3D(bbox_3d)
    k = get_rotation_axis(bottom, upper)
    for idx,point in enumerate(points):
        rotated_points[idx] = rodrigues_rotation_one_point(point,theta,k)
    return rotated_points

def rot(points,theta,bbox_3d):
    bottom, upper = get_points_from_bbox3D(bbox_3d)
    axis = get_rotation_axis(bottom, upper)
    r = R.from_rotvec(theta * axis)
    return r.apply(points)

def rotate_points(points,theta,bbox_3d):
    bottom, upper = get_points_from_bbox3D(bbox_3d)
    axis = get_rotation_axis(bottom, upper)

    centroid = np.mean(bbox_3d, axis=0)
    z_axis = np.array([0,0,1])

    shifted_points = points - centroid

    normalized_v = axis / np.sqrt(np.sum(axis**2))

    k = (normalized_v + z_axis) / 2

    rotate_to_original = R.from_rotvec(k * np.pi)

    # r2 = rotate_to_original.inv()

    original_box = rotate_to_original.apply(shifted_points)

    print("Shifted: {}".format(shifted_points))

    print("Original: {}".format(original_box))
    centroid_center = np.mean(original_box, axis=0)
    print(centroid_center)

    z_axis = np.array([0,0,1])

    rotate = R.from_rotvec(z_axis * theta)
    rotated_original = rotate.apply(original_box)

    # rotated_original = original_box

    r2 = R.from_rotvec(k * np.pi)
    r2 = r2.inv()

    rotated_back = r2.apply(rotated_original)

    print("Rotated back: {}".format(rotated_back))

    shifted_back = rotated_back + centroid

    return shifted_back


def rotate_points_good(points,theta,bbox_3d):
    bottom, upper = get_points_from_bbox3D(bbox_3d)
    axis = get_rotation_axis(bottom, upper)

    centroid = np.mean(bbox_3d, axis=0)
    z_axis = np.array([0,0,1])

    shifted_points = points - centroid

    normalized_v = axis / np.sqrt(np.sum(axis**2))

    k = (normalized_v + z_axis) / 2

    rotate_to_original = R.from_rotvec(k * np.pi)
    r2 = rotate_to_original.inv()
    original_box = r2.apply(shifted_points)

    print("Shifted: {}".format(shifted_points))

    print("Original: {}".format(original_box))
    centroid_center = np.mean(original_box, axis=0)
    print(centroid_center)

    z_axis = np.array([0,0,1])

    rotate = R.from_rotvec(z_axis * theta)
    rotated_original = rotate.apply(original_box)

    # rotated_original = original_box

    rotated_back = rotate_to_original.apply(rotated_original)

    print("Rotated back: {}".format(rotated_back))

    shifted_back = rotated_back + centroid

    return shifted_back










