import unittest
from my_functions import generate_octagon,\
                     generate_symmetries,\
                     generate_symmetries_quad,\
                     make_octagon_symmetries,\
                     rotate_box,\
                     S_Loss,\
                     S_Loss_MSE,\
                     S_Loss_RMSE,\
                     S_Loss_MAE,\
                     get_line,\
                     get_intersection,\
                     get_quadrilateral_from_octagon,\
                     is_valid_prediction
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

def make_symmetries_helper(ax,ay,bx,by,Ax,Ay,Bx,By,cx,cy,dx,dy,Cx,Cy,Dx,Dy,
                    ex,ey,fx,fy,Ex,Ey,Fx,Fy,gx,gy,hx,hy,Gx,Gy,Hx,Hy):

    symmetry_pts_grdth = np.array([

        ### Rot 1

        [ax,ay, bx,by, Ax,Ay, Bx,By,
        cx,cy, dx,dy, Cx,Cy, Dx,Dy,
        ex,ey, fx,fy, Ex,Ey, Fx,Fy,
        gx,gy, hx,hy, Gx,Gy, Hx,Hy],

        ### Rot 2

        [ex,ey, ax,ay, Ex,Ey, Ax,Ay,
        bx,by, cx,cy, Bx,By, Cx,Cy,
        fx,fy, gx,gy, Fx,Fy, Gx,Gy,
        hx,hy, dx,dy, Hx,Hy, Dx,Dy],

        ### Rot 3

        [fx,fy, ex,ey, Fx,Fy, Ex,Ey,
        ax,ay, bx,by, Ax,Ay, Bx,By,
        gx,gy, hx,hy, Gx,Gy, Hx,Hy,
        dx,dy, cx,cy, Dx,Dy, Cx,Cy],

        ### Rot 4

        [gx,gy, fx,fy, Gx,Gy, Fx,Fy,
        ex,ey, ax,ay, Ex,Ey, Ax,Ay,
        hx,hy, dx,dy, Hx,Hy, Dx,Dy,
        cx,cy, bx,by, Cx,Cy, Bx,By],

        ### Rot 5

        [hx,hy, gx,gy, Hx,Hy, Gx,Gy,
        fx,fy, ex,ey, Fx,Fy, Ex,Ey,
        dx,dy, cx,cy, Dx,Dy, Cx,Cy,
        bx,by, ax,ay, Bx,By, Ax,Ay],

        ### Rot 6

        [dx,dy, hx,hy, Dx,Dy, Hx,Hy,
        gx,gy, fx,fy, Gx,Gy, Fx,Fy,
        cx,cy, bx,by, Cx,Cy, Bx,By,
        ax,ay, ex,ey, Ax,Ay, Ex,Ey],

        ### Rot 7

        [cx,cy, dx,dy, Cx,Cy, Dx,Dy,
        hx,hy, gx,gy, Hx,Hy, Gx,Gy,
        bx,by, ax,ay, Bx,By, Ax,Ay,
        ex,ey, fx,fy, Ex,Ey, Fx,Fy],

        ### Rot 8

        [bx,by, cx,cy, Bx,By, Cx,Cy,
        dx,dy, hx,hy, Dx,Dy, Hx,Hy,
        ax,ay, ex,ey, Ax,Ay, Ex,Ey,
        fx,fy, gx,gy, Fx,Fy,Gx,Gy],
    ])

    return symmetry_pts_grdth

def make_symmetries_quad_helper(ax,ay,Ax,Ay,bx,by,Bx,By,cx,cy,Cx,Cy,dx,dy,Dx,Dy):

    symmetry_pts_grdth = np.array([

        ### Rot 1

        [ax,ay,Ax,Ay,bx,by,Bx,By,
        cx,cy,Cx,Cy,dx,dy,Dx,Dy],

        ### Rot 2

        [cx,cy,Cx,Cy,ax,ay,Ax,Ay,
        dx,dy,Dx,Dy,bx,by,Bx,By],

        ### Rot 3

        [dx,dy,Dx,Dy,cx,cy,Cx,Cy,
        bx,by,Bx,By,ax,ay,Ax,Ay],

        ### Rot 4

        [bx,by,Bx,By,dx,dy,Dx,Dy,
        ax,ay,Ax,Ay,cx,cy,Cx,Cy],

    ])

    return symmetry_pts_grdth

def metric_scikit(points_numpy,symmetries_numpy,metric,disp=False):

    """Manually compute distance between points and symmetries
    using scikit tools.
    """

    predictions = points_numpy.shape[0]
    nb_symmetries = symmetries_numpy.shape[1]
    dists = np.zeros((predictions,nb_symmetries))

    if disp: print("START --- "+str(metric))
    for i in range(predictions):
        for j,symmetry in enumerate(symmetries_numpy[i]):
            mse = metric(symmetry, points_numpy[i].squeeze())
            if disp: print("dist between: {} // {} is {}".format(points_numpy[i].squeeze(), symmetry, mse))
            dists[i,j] = mse
    if disp:
        print("RESULT")
        print(dists)
    return dists

def find_min(loss):
    return np.min(loss,axis=1)

    

class TestFunctions(unittest.TestCase):

    def test_generate_symmetries_one_input(self):

        octagon = np.array([['ax', 'ay', 'bx', 'by', 'Ax', 'Ay', 'Bx', 'By', 'cx', 'cy', 'dx',
        'dy', 'Cx', 'Cy', 'Dx', 'Dy', 'ex', 'ey', 'fx', 'fy', 'Ex', 'Ey',
        'Fx', 'Fy', 'gx', 'gy', 'hx', 'hy', 'Gx', 'Gy', 'Hx', 'Hy']])
        symmetries = make_octagon_symmetries(octagon)

        symmetries_grdth = np.array([
            [['ax', 'ay', 'bx', 'by', 'Ax', 'Ay', 'Bx', 'By', 'cx', 'cy', 'dx',
            'dy', 'Cx', 'Cy', 'Dx', 'Dy', 'ex', 'ey', 'fx', 'fy', 'Ex', 'Ey',
            'Fx', 'Fy', 'gx', 'gy', 'hx', 'hy', 'Gx', 'Gy', 'Hx', 'Hy'],
            ['ex', 'ey', 'ax', 'ay', 'Ex', 'Ey', 'Ax', 'Ay', 'bx', 'by', 'cx',
            'cy', 'Bx', 'By', 'Cx', 'Cy', 'fx', 'fy', 'gx', 'gy', 'Fx', 'Fy',
            'Gx', 'Gy', 'hx', 'hy', 'dx', 'dy', 'Hx', 'Hy', 'Dx', 'Dy'],
            ['fx', 'fy', 'ex', 'ey', 'Fx', 'Fy', 'Ex', 'Ey', 'ax', 'ay', 'bx',
            'by', 'Ax', 'Ay', 'Bx', 'By', 'gx', 'gy', 'hx', 'hy', 'Gx', 'Gy',
            'Hx', 'Hy', 'dx', 'dy', 'cx', 'cy', 'Dx', 'Dy', 'Cx', 'Cy'],
            ['gx', 'gy', 'fx', 'fy', 'Gx', 'Gy', 'Fx', 'Fy', 'ex', 'ey', 'ax',
            'ay', 'Ex', 'Ey', 'Ax', 'Ay', 'hx', 'hy', 'dx', 'dy', 'Hx', 'Hy',
            'Dx', 'Dy', 'cx', 'cy', 'bx', 'by', 'Cx', 'Cy', 'Bx', 'By'],
            ['hx', 'hy', 'gx', 'gy', 'Hx', 'Hy', 'Gx', 'Gy', 'fx', 'fy', 'ex',
            'ey', 'Fx', 'Fy', 'Ex', 'Ey', 'dx', 'dy', 'cx', 'cy', 'Dx', 'Dy',
            'Cx', 'Cy', 'bx', 'by', 'ax', 'ay', 'Bx', 'By', 'Ax', 'Ay'],
            ['dx', 'dy', 'hx', 'hy', 'Dx', 'Dy', 'Hx', 'Hy', 'gx', 'gy', 'fx',
            'fy', 'Gx', 'Gy', 'Fx', 'Fy', 'cx', 'cy', 'bx', 'by', 'Cx', 'Cy',
            'Bx', 'By', 'ax', 'ay', 'ex', 'ey', 'Ax', 'Ay', 'Ex', 'Ey'],
            ['cx', 'cy', 'dx', 'dy', 'Cx', 'Cy', 'Dx', 'Dy', 'hx', 'hy', 'gx',
            'gy', 'Hx', 'Hy', 'Gx', 'Gy', 'bx', 'by', 'ax', 'ay', 'Bx', 'By',
            'Ax', 'Ay', 'ex', 'ey', 'fx', 'fy', 'Ex', 'Ey', 'Fx', 'Fy'],
            ['bx', 'by', 'cx', 'cy', 'Bx', 'By', 'Cx', 'Cy', 'dx', 'dy', 'hx',
            'hy', 'Dx', 'Dy', 'Hx', 'Hy', 'ax', 'ay', 'ex', 'ey', 'Ax', 'Ay',
            'Ex', 'Ey', 'fx', 'fy', 'gx', 'gy', 'Fx', 'Fy', 'Gx', 'Gy']]
        ])

        self.assertEqual(symmetries.tolist(), symmetries_grdth.tolist(), "")

    def test_generate_symmetries_three_inputs(self):

        octagon = np.array([
            ['a1x', 'a1y', 'b1x', 'b1y', 'A1x', 'A1y', 'B1x', 'B1y', 'c1x', 'c1y', 'd1x',
            'd1y', 'C1x', 'C1y', 'D1x', 'D1y', 'e1x', 'e1y', 'f1x', 'f1y', 'E1x', 'E1y',
            'F1x', 'F1y', 'g1x', 'g1y', 'h1x', 'h1y', 'G1x', 'G1y', 'H1x', 'H1y'],

            ['a2x', 'a2y', 'b2x', 'b2y', 'A2x', 'A2y', 'B2x', 'B2y', 'c2x', 'c2y', 'd2x',
            'd2y', 'C2x', 'C2y', 'D2x', 'D2y', 'e2x', 'e2y', 'f2x', 'f2y', 'E2x', 'E2y',
            'F2x', 'F2y', 'g2x', 'g2y', 'h2x', 'h2y', 'G2x', 'G2y', 'H2x', 'H2y'],
            
            ['a3x', 'a3y', 'b3x', 'b3y', 'A3x', 'A3y', 'B3x', 'B3y', 'c3x', 'c3y', 'd3x',
            'd3y', 'C3x', 'C3y', 'D3x', 'D3y', 'e3x', 'e3y', 'f3x', 'f3y', 'E3x', 'E3y',
            'F3x', 'F3y', 'g3x', 'g3y', 'h3x', 'h3y', 'G3x', 'G3y', 'H3x', 'H3y']
        ])
        symmetries = make_octagon_symmetries(octagon)

        symmetries_grdth = np.array([
            [['a1x', 'a1y', 'b1x', 'b1y', 'A1x', 'A1y', 'B1x', 'B1y', 'c1x', 'c1y', 'd1x',
            'd1y', 'C1x', 'C1y', 'D1x', 'D1y', 'e1x', 'e1y', 'f1x', 'f1y', 'E1x', 'E1y',
            'F1x', 'F1y', 'g1x', 'g1y', 'h1x', 'h1y', 'G1x', 'G1y', 'H1x', 'H1y'],
            ['e1x', 'e1y', 'a1x', 'a1y', 'E1x', 'E1y', 'A1x', 'A1y', 'b1x', 'b1y', 'c1x',
            'c1y', 'B1x', 'B1y', 'C1x', 'C1y', 'f1x', 'f1y', 'g1x', 'g1y', 'F1x', 'F1y',
            'G1x', 'G1y', 'h1x', 'h1y', 'd1x', 'd1y', 'H1x', 'H1y', 'D1x', 'D1y'],
            ['f1x', 'f1y', 'e1x', 'e1y', 'F1x', 'F1y', 'E1x', 'E1y', 'a1x', 'a1y', 'b1x',
            'b1y', 'A1x', 'A1y', 'B1x', 'B1y', 'g1x', 'g1y', 'h1x', 'h1y', 'G1x', 'G1y',
            'H1x', 'H1y', 'd1x', 'd1y', 'c1x', 'c1y', 'D1x', 'D1y', 'C1x', 'C1y'],
            ['g1x', 'g1y', 'f1x', 'f1y', 'G1x', 'G1y', 'F1x', 'F1y', 'e1x', 'e1y', 'a1x',
            'a1y', 'E1x', 'E1y', 'A1x', 'A1y', 'h1x', 'h1y', 'd1x', 'd1y', 'H1x', 'H1y',
            'D1x', 'D1y', 'c1x', 'c1y', 'b1x', 'b1y', 'C1x', 'C1y', 'B1x', 'B1y'],
            ['h1x', 'h1y', 'g1x', 'g1y', 'H1x', 'H1y', 'G1x', 'G1y', 'f1x', 'f1y', 'e1x',
            'e1y', 'F1x', 'F1y', 'E1x', 'E1y', 'd1x', 'd1y', 'c1x', 'c1y', 'D1x', 'D1y',
            'C1x', 'C1y', 'b1x', 'b1y', 'a1x', 'a1y', 'B1x', 'B1y', 'A1x', 'A1y'],
            ['d1x', 'd1y', 'h1x', 'h1y', 'D1x', 'D1y', 'H1x', 'H1y', 'g1x', 'g1y', 'f1x',
            'f1y', 'G1x', 'G1y', 'F1x', 'F1y', 'c1x', 'c1y', 'b1x', 'b1y', 'C1x', 'C1y',
            'B1x', 'B1y', 'a1x', 'a1y', 'e1x', 'e1y', 'A1x', 'A1y', 'E1x', 'E1y'],
            ['c1x', 'c1y', 'd1x', 'd1y', 'C1x', 'C1y', 'D1x', 'D1y', 'h1x', 'h1y', 'g1x',
            'g1y', 'H1x', 'H1y', 'G1x', 'G1y', 'b1x', 'b1y', 'a1x', 'a1y', 'B1x', 'B1y',
            'A1x', 'A1y', 'e1x', 'e1y', 'f1x', 'f1y', 'E1x', 'E1y', 'F1x', 'F1y'],
            ['b1x', 'b1y', 'c1x', 'c1y', 'B1x', 'B1y', 'C1x', 'C1y', 'd1x', 'd1y', 'h1x',
            'h1y', 'D1x', 'D1y', 'H1x', 'H1y', 'a1x', 'a1y', 'e1x', 'e1y', 'A1x', 'A1y',
            'E1x', 'E1y', 'f1x', 'f1y', 'g1x', 'g1y', 'F1x', 'F1y', 'G1x', 'G1y']],

            [['a2x', 'a2y', 'b2x', 'b2y', 'A2x', 'A2y', 'B2x', 'B2y', 'c2x', 'c2y', 'd2x',
            'd2y', 'C2x', 'C2y', 'D2x', 'D2y', 'e2x', 'e2y', 'f2x', 'f2y', 'E2x', 'E2y',
            'F2x', 'F2y', 'g2x', 'g2y', 'h2x', 'h2y', 'G2x', 'G2y', 'H2x', 'H2y'],
            ['e2x', 'e2y', 'a2x', 'a2y', 'E2x', 'E2y', 'A2x', 'A2y', 'b2x', 'b2y', 'c2x',
            'c2y', 'B2x', 'B2y', 'C2x', 'C2y', 'f2x', 'f2y', 'g2x', 'g2y', 'F2x', 'F2y',
            'G2x', 'G2y', 'h2x', 'h2y', 'd2x', 'd2y', 'H2x', 'H2y', 'D2x', 'D2y'],
            ['f2x', 'f2y', 'e2x', 'e2y', 'F2x', 'F2y', 'E2x', 'E2y', 'a2x', 'a2y', 'b2x',
            'b2y', 'A2x', 'A2y', 'B2x', 'B2y', 'g2x', 'g2y', 'h2x', 'h2y', 'G2x', 'G2y',
            'H2x', 'H2y', 'd2x', 'd2y', 'c2x', 'c2y', 'D2x', 'D2y', 'C2x', 'C2y'],
            ['g2x', 'g2y', 'f2x', 'f2y', 'G2x', 'G2y', 'F2x', 'F2y', 'e2x', 'e2y', 'a2x',
            'a2y', 'E2x', 'E2y', 'A2x', 'A2y', 'h2x', 'h2y', 'd2x', 'd2y', 'H2x', 'H2y',
            'D2x', 'D2y', 'c2x', 'c2y', 'b2x', 'b2y', 'C2x', 'C2y', 'B2x', 'B2y'],
            ['h2x', 'h2y', 'g2x', 'g2y', 'H2x', 'H2y', 'G2x', 'G2y', 'f2x', 'f2y', 'e2x',
            'e2y', 'F2x', 'F2y', 'E2x', 'E2y', 'd2x', 'd2y', 'c2x', 'c2y', 'D2x', 'D2y',
            'C2x', 'C2y', 'b2x', 'b2y', 'a2x', 'a2y', 'B2x', 'B2y', 'A2x', 'A2y'],
            ['d2x', 'd2y', 'h2x', 'h2y', 'D2x', 'D2y', 'H2x', 'H2y', 'g2x', 'g2y', 'f2x',
            'f2y', 'G2x', 'G2y', 'F2x', 'F2y', 'c2x', 'c2y', 'b2x', 'b2y', 'C2x', 'C2y',
            'B2x', 'B2y', 'a2x', 'a2y', 'e2x', 'e2y', 'A2x', 'A2y', 'E2x', 'E2y'],
            ['c2x', 'c2y', 'd2x', 'd2y', 'C2x', 'C2y', 'D2x', 'D2y', 'h2x', 'h2y', 'g2x',
            'g2y', 'H2x', 'H2y', 'G2x', 'G2y', 'b2x', 'b2y', 'a2x', 'a2y', 'B2x', 'B2y',
            'A2x', 'A2y', 'e2x', 'e2y', 'f2x', 'f2y', 'E2x', 'E2y', 'F2x', 'F2y'],
            ['b2x', 'b2y', 'c2x', 'c2y', 'B2x', 'B2y', 'C2x', 'C2y', 'd2x', 'd2y', 'h2x',
            'h2y', 'D2x', 'D2y', 'H2x', 'H2y', 'a2x', 'a2y', 'e2x', 'e2y', 'A2x', 'A2y',
            'E2x', 'E2y', 'f2x', 'f2y', 'g2x', 'g2y', 'F2x', 'F2y', 'G2x', 'G2y']],

            [['a3x', 'a3y', 'b3x', 'b3y', 'A3x', 'A3y', 'B3x', 'B3y', 'c3x', 'c3y', 'd3x',
            'd3y', 'C3x', 'C3y', 'D3x', 'D3y', 'e3x', 'e3y', 'f3x', 'f3y', 'E3x', 'E3y',
            'F3x', 'F3y', 'g3x', 'g3y', 'h3x', 'h3y', 'G3x', 'G3y', 'H3x', 'H3y'],
            ['e3x', 'e3y', 'a3x', 'a3y', 'E3x', 'E3y', 'A3x', 'A3y', 'b3x', 'b3y', 'c3x',
            'c3y', 'B3x', 'B3y', 'C3x', 'C3y', 'f3x', 'f3y', 'g3x', 'g3y', 'F3x', 'F3y',
            'G3x', 'G3y', 'h3x', 'h3y', 'd3x', 'd3y', 'H3x', 'H3y', 'D3x', 'D3y'],
            ['f3x', 'f3y', 'e3x', 'e3y', 'F3x', 'F3y', 'E3x', 'E3y', 'a3x', 'a3y', 'b3x',
            'b3y', 'A3x', 'A3y', 'B3x', 'B3y', 'g3x', 'g3y', 'h3x', 'h3y', 'G3x', 'G3y',
            'H3x', 'H3y', 'd3x', 'd3y', 'c3x', 'c3y', 'D3x', 'D3y', 'C3x', 'C3y'],
            ['g3x', 'g3y', 'f3x', 'f3y', 'G3x', 'G3y', 'F3x', 'F3y', 'e3x', 'e3y', 'a3x',
            'a3y', 'E3x', 'E3y', 'A3x', 'A3y', 'h3x', 'h3y', 'd3x', 'd3y', 'H3x', 'H3y',
            'D3x', 'D3y', 'c3x', 'c3y', 'b3x', 'b3y', 'C3x', 'C3y', 'B3x', 'B3y'],
            ['h3x', 'h3y', 'g3x', 'g3y', 'H3x', 'H3y', 'G3x', 'G3y', 'f3x', 'f3y', 'e3x',
            'e3y', 'F3x', 'F3y', 'E3x', 'E3y', 'd3x', 'd3y', 'c3x', 'c3y', 'D3x', 'D3y',
            'C3x', 'C3y', 'b3x', 'b3y', 'a3x', 'a3y', 'B3x', 'B3y', 'A3x', 'A3y'],
            ['d3x', 'd3y', 'h3x', 'h3y', 'D3x', 'D3y', 'H3x', 'H3y', 'g3x', 'g3y', 'f3x',
            'f3y', 'G3x', 'G3y', 'F3x', 'F3y', 'c3x', 'c3y', 'b3x', 'b3y', 'C3x', 'C3y',
            'B3x', 'B3y', 'a3x', 'a3y', 'e3x', 'e3y', 'A3x', 'A3y', 'E3x', 'E3y'],
            ['c3x', 'c3y', 'd3x', 'd3y', 'C3x', 'C3y', 'D3x', 'D3y', 'h3x', 'h3y', 'g3x',
            'g3y', 'H3x', 'H3y', 'G3x', 'G3y', 'b3x', 'b3y', 'a3x', 'a3y', 'B3x', 'B3y',
            'A3x', 'A3y', 'e3x', 'e3y', 'f3x', 'f3y', 'E3x', 'E3y', 'F3x', 'F3y'],
            ['b3x', 'b3y', 'c3x', 'c3y', 'B3x', 'B3y', 'C3x', 'C3y', 'd3x', 'd3y', 'h3x',
            'h3y', 'D3x', 'D3y', 'H3x', 'H3y', 'a3x', 'a3y', 'e3x', 'e3y', 'A3x', 'A3y',
            'E3x', 'E3y', 'f3x', 'f3y', 'g3x', 'g3y', 'F3x', 'F3y', 'G3x', 'G3y']]
        ])
        
        self.assertEqual(symmetries.tolist(), symmetries_grdth.tolist(), "")

    def test_generate_octagon_with_one_input(self):

        quad_pts_numpy = np.array([
            [100,700,0,800,100,100,0,0,700,700,800,800,700,100,800,0]
        ])
        quad_pts_torch = quad_pts_numpy.reshape((1,-1))

        oct_pts_torch = generate_octagon(quad_pts_torch)
        oct_pts_numpy = generate_octagon(quad_pts_numpy)

        oct_pts_grdth = np.array([[
            250,700,
            100,550,
            200,800,
            0,600,

            100,250,
            250,100,
            0,200,
            200,0,

            550,700,
            700,550,
            600,800,
            800,600,

            700,250,
            550,100,
            800,200,
            600,0
        ]])

        self.assertEqual(oct_pts_torch.tolist(), oct_pts_grdth.tolist(), "")
        self.assertEqual(oct_pts_numpy.tolist(), oct_pts_grdth.tolist(), "")

    
    def test_generate_symmetries_with_two_inputs(self):

        number_of_octagons = 2

        quad_pts_numpy = np.array([

            ### First quadrilateral

            [100,700,0,800,100,100,0,0,700,700,800,800,700,100,800,0],

            ### Second quadrilateral

            [200,600,0,600,600,0,400,0,800,800,600,800,1200,200,1000,200]

        ]).astype('float')

        quad_pts_torch = torch.from_numpy(quad_pts_numpy)

        ### Value to test
        symmetry_pts_numpy = generate_symmetries(quad_pts_numpy)
        symmetry_pts_torch = generate_symmetries(quad_pts_torch)

        ### First octagon

        ax,ay = 250,700
        bx,by = 100,550
        Ax,Ay = 200,800  
        Bx,By = 0,600

        cx,cy = 100,250
        dx,dy = 250,100  
        Cx,Cy = 0,200
        Dx,Dy = 200,0

        ex,ey = 550,700
        fx,fy = 700,550
        Ex,Ey = 600,800
        Fx,Fy = 800,600

        gx,gy = 700,250
        hx,hy = 550,100
        Gx,Gy = 800,200
        Hx,Hy = 600,0

        symmetry_pts_grdth_1 = make_symmetries_helper(
                                    ax,ay,bx,by,Ax,Ay,Bx,By,
                                    cx,cy,dx,dy,Cx,Cy,Dx,Dy,
                                    ex,ey,fx,fy,Ex,Ey,Fx,Fy,
                                    gx,gy,hx,hy,Gx,Gy,Hx,Hy
                                    )

        ### Second octagon
             
        ax, ay = 350,650
        bx, by = 300,450
        Ax, Ay = 150,650
        Bx, By = 100,450

        cx, cy = 500,150
        dx, dy = 750,50
        Cx, Cy = 300,150
        Dx, Dy = 550,50

        ex, ey = 650,750
        fx, fy = 900,650
        Ex, Ey = 450,750
        Fx, Fy = 700,650

        gx, gy = 1100,350
        hx, hy = 1050,150
        Gx, Gy = 900,350
        Hx, Hy = 850,150

        symmetry_pts_grdth_2 = make_symmetries_helper(
                                    ax,ay,bx,by,Ax,Ay,Bx,By,
                                    cx,cy,dx,dy,Cx,Cy,Dx,Dy,
                                    ex,ey,fx,fy,Ex,Ey,Fx,Fy,
                                    gx,gy,hx,hy,Gx,Gy,Hx,Hy
                                    )

        symmetry_pts_grdth = np.vstack((symmetry_pts_grdth_1,
                                        symmetry_pts_grdth_2)
                                       ).reshape(number_of_octagons,8,-1)

        self.assertEqual(symmetry_pts_numpy.tolist(), symmetry_pts_grdth.tolist(), "")
        self.assertEqual(symmetry_pts_torch.tolist(), symmetry_pts_grdth.tolist(), "")

    def test_generate_octagon_with_two_inputs(self):

        quad_pts_numpy = np.array([
            [100,700,0,800,100,100,0,0,700,700,800,800,700,100,800,0],
            [200,600,0,600,600,0,400,0,800,800,600,800,1200,200,1000,200]
        ])
        quad_pts_torch = torch.from_numpy(quad_pts_numpy).float()

        oct_pts_torch = generate_octagon(quad_pts_torch)
        oct_pts_numpy = generate_octagon(quad_pts_numpy)

        oct_pts_grdth = np.array([
            [250,700,
            100,550,
            200,800,
            0,600,

            100,250,
            250,100,
            0,200,
            200,0,

            550,700,
            700,550,
            600,800,
            800,600,

            700,250,
            550,100,
            800,200,
            600,0],
             
            [350,650,
            300,450,
            150,650,
            100,450,

            500,150,
            750,50,
            300,150,
            550,50,

            650,750,
            900,650,
            450,750,
            700,650,

            1100,350,
            1050,150,
            900,350,
            850,150]
        ])

        self.assertEqual(oct_pts_torch.tolist(), oct_pts_grdth.tolist(), "")
        self.assertEqual(oct_pts_numpy.tolist(), oct_pts_grdth.tolist(), "")

    def test_rotate_box_0(self):

        box = np.array([
            [0,0],
            [-2,-2],
            [605,800],
            [300,300],
            [100,250],
            [-100,30]
        ])

        rotated_box = rotate_box(box,0)

        rotated_box_grdth = box

        self.assertEqual(rotated_box.tolist(), rotated_box_grdth.tolist(), "")

    def test_rotate_box_90(self):

        box = np.array([
            [0,0],
            [-2,-2],
            [605,800],
            [300,300],
            [100,250],
            [-100,30]
        ])

        rotated_box = rotate_box(box,1)

        rotated_box_grdth = np.array([
            [0,599],
            [-2,601],
            [800,-6],
            [300,299],
            [250,499],
            [30,699]
        ])

        self.assertEqual(rotated_box.tolist(), rotated_box_grdth.tolist(), "")

    def test_rotate_box_180(self):

        box = np.array([
            [0,0],
            [-2,-2],
            [605,800],
            [300,300],
            [100,250],
            [-100,30]
        ])

        rotated_box = rotate_box(box,2)

        rotated_box_grdth = np.array([
            [599,599],
            [601,601],
            [-6,-201],
            [299,299],
            [499,349],
            [699,569]
        ])

        self.assertEqual(rotated_box.tolist(), rotated_box_grdth.tolist(), "")

    def test_rotate_box_270(self):

        box = np.array([
            [0,0],
            [-2,-2],
            [605,800],
            [300,300],
            [100,250],
            [-100,30]
        ])

        rotated_box = rotate_box(box,3)

        rotated_box_grdth = np.array([
            [599,0],
            [601,-2],
            [-201,605],
            [299,300],
            [349,100],
            [569,-100]
        ])

        self.assertEqual(rotated_box.tolist(), rotated_box_grdth.tolist(), "")

    def test_S_Loss_MSE(self):

        octagon_numpy = np.array([
            [[0,1,2,3,4,5,6,7,8,9]],
            [[5,5,5,5,5,5,5,5,5,5]],
            [[234,2345,8765,24,56,-2,-90,123,54378,993]],
        ])

        symmetries_numpy = np.array([
            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]],

            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]],

            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]]
        ])

        octagon_torch = torch.from_numpy(octagon_numpy).float()
        symmetries_torch = torch.from_numpy(symmetries_numpy).float()

        MSEs = metric_scikit(octagon_numpy,symmetries_numpy,metric=mean_squared_error, disp=False)
        mins = find_min(MSEs)
        avg = np.mean(mins)

        # print("--------MSE--------")
        # print(MSEs)
        # print("Mins: {}".format(mins))
        # print("Avg: {}".format(avg))

        symmetry_loss_numpy = S_Loss(octagon_numpy, symmetries_numpy, 'MSE')
        symmetry_loss_torch = S_Loss(octagon_torch, symmetries_torch, 'MSE')
        symmetry_loss_torch_speed = S_Loss_MSE(octagon_torch, symmetries_torch)

        self.assertEqual(symmetry_loss_numpy, avg, "")
        self.assertEqual(symmetry_loss_torch, avg, "")
        self.assertEqual(symmetry_loss_torch_speed, avg, "")

    def test_S_Loss_RMSE(self):

        octagon_numpy = np.array([
            [[0,1,2,3,4,5,6,7,8,9]],
            [[5,5,5,5,5,5,5,5,5,5]],
            [[234,245,85,24,56,-2,-90,123,54,99]],
        ])

        symmetries_numpy = np.array([
            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]],

            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]],

            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]]
        ])

        octagon_torch = torch.from_numpy(octagon_numpy).float()
        symmetries_torch = torch.from_numpy(symmetries_numpy).float()

        RMSEs = metric_scikit(octagon_numpy,symmetries_numpy,metric=lambda x,y: sqrt(mean_squared_error(x,y)), disp=False)
        mins = find_min(RMSEs)
        avg = np.mean(mins)

        # print("--------RMSE-------")
        # print(RMSEs)
        # print("Mins: {}".format(mins))
        # print("Avg: {}".format(avg))

        symmetry_loss_numpy = S_Loss(octagon_numpy, symmetries_numpy, 'RMSE')
        symmetry_loss_torch = S_Loss(octagon_torch, symmetries_torch, 'RMSE')
        symmetry_loss_torch_speed = S_Loss_RMSE(octagon_torch, symmetries_torch)

        self.assertEqual(symmetry_loss_numpy, avg, "")
        self.assertEqual(symmetry_loss_torch, avg, "")
        self.assertEqual(symmetry_loss_torch_speed, avg, "")


    def test_S_Loss_MAE(self):

        octagon_numpy = np.array([
            [[0,1,2,3,4,5,6,7,8,9]],
            [[5,5,5,5,5,5,5,5,5,5]],
            [[234,2345,8765,24,56,-2,-90,123,54378,993]],
        ])

        symmetries_numpy = np.array([
            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]],

            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]],

            [[0,1,2,3,4,5,6,7,8,9],
            [9,8,7,6,5,4,3,2,1,0],
            [2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3]]
        ])

        octagon_torch = torch.from_numpy(octagon_numpy).float()
        symmetries_torch = torch.from_numpy(symmetries_numpy).float()

        MAEs = metric_scikit(octagon_numpy,symmetries_numpy,metric=mean_absolute_error)
        mins = find_min(MAEs)
        avg = np.mean(mins)

        # print("--------MAE--------")
        # print(MAEs)
        # print("Mins: {}".format(mins))
        # print("Avg: {}".format(avg))

        symmetry_loss_numpy = S_Loss(octagon_numpy, symmetries_numpy, 'MAE')
        symmetry_loss_torch = S_Loss(octagon_torch, symmetries_torch, 'MAE')
        symmetry_loss_torch_speed = S_Loss_MAE(octagon_torch, symmetries_torch)

        self.assertEqual(symmetry_loss_numpy, avg, "")
        self.assertEqual(symmetry_loss_torch, avg, "")
        self.assertEqual(symmetry_loss_torch_speed, avg, "")

    def test_generate_quad_symmetries(self):

        number_of_quadrilaterals = 2

        quad_pts_numpy = np.array([

            ### First quadrilateral

            [100,700,0,800,100,100,0,0,700,700,800,800,700,100,800,0],

            ### Second quadrilateral

            [200,600,0,600,600,0,400,0,800,800,600,800,1200,200,1000,200]

        ]).astype('float')

        quad_pts_torch = torch.from_numpy(quad_pts_numpy)

        ### Value to test
        symmetry_pts_numpy = generate_symmetries_quad(quad_pts_numpy)
        symmetry_pts_torch = generate_symmetries_quad(quad_pts_torch)

        ### First quadrilateral

        ax, ay = 100,700
        Ax, Ay = 0,800
        bx, by = 100,100
        Bx, By = 0,0

        cx, cy = 700,700
        Cx, Cy = 800,800
        dx, dy = 700,100
        Dx, Dy = 800,0
        
        symmetry_pts_grdth_1 = make_symmetries_quad_helper(
                                    ax,ay,Ax,Ay,bx,by,Bx,By,
                                    cx,cy,Cx,Cy,dx,dy,Dx,Dy,
                                    )

        ### Second quadrilateral

        ax, ay = 200,600
        Ax, Ay = 0,600
        bx, by = 600,0
        Bx, By = 400,0

        cx, cy = 800,800
        Cx, Cy = 600,800
        dx, dy = 1200,200
        Dx, Dy = 1000,200

        symmetry_pts_grdth_2 = make_symmetries_quad_helper(
                                    ax,ay,Ax,Ay,bx,by,Bx,By,
                                    cx,cy,Cx,Cy,dx,dy,Dx,Dy,
                                    )

        symmetry_pts_grdth = np.vstack((symmetry_pts_grdth_1,
                                        symmetry_pts_grdth_2)
                                       ).reshape(number_of_quadrilaterals,4,-1).astype('float')

        self.assertEqual(symmetry_pts_numpy.tolist(), symmetry_pts_grdth.tolist(), "")
        self.assertEqual(symmetry_pts_torch.tolist(), symmetry_pts_grdth.tolist(), "")

    def test_get_line_from_2_points(self):

        p1, p2 = (10,2), (8,4)
        slope_grdth, intercept_grdth = -1, 12

        (slope, intercept) = get_line(p1,p2)

        self.assertEqual(slope,slope_grdth, "Should be -1")
        self.assertEqual(intercept,intercept_grdth, "Should be 12")

        p1, p2 = (-5,8), (4,2)
        slope_grdth, intercept_grdth = -2/3, 14/3

        (slope, intercept) = get_line(p1,p2)

        self.assertAlmostEqual(slope,slope_grdth, msg="Should be -2/3")
        self.assertAlmostEqual(intercept,intercept_grdth, msg="Should be 14/3")

        p1, p2 = (5,0), (5,20)
        slope_grdth, intercept_grdth = 'v', 5

        (slope, intercept) = get_line(p1,p2)

        self.assertAlmostEqual(slope,slope_grdth, msg="Should 'v'")
        self.assertAlmostEqual(intercept,intercept_grdth, msg="Should be 5")

        self.assertRaises
    
    def test_get_intersection_of_2_lines(self):

        l1, l2 = (2,3), (-0.5,7)
        x_grdth, y_grdth = 1.6,6.2 

        (x,y) = get_intersection(l1,l2)

        self.assertEqual(x,x_grdth, "Should be 1.6")
        self.assertEqual(y,y_grdth, "Should be 6.2")

        l1, l2 = (8,25), (-2,18)
        x_grdth, y_grdth = -0.7, 19.4

        (x,y) = get_intersection(l1,l2)

        self.assertEqual(x,x_grdth, "Should be -0.7")
        self.assertEqual(y,y_grdth, "Should be 19.4")
        
        l1, l2 = ('v',0), (1,-5)
        x_grdth, y_grdth = 0.0, -5.0

        (x,y) = get_intersection(l1,l2)

        self.assertEqual(x,x_grdth, "Should be 0.0")
        self.assertEqual(y,y_grdth, "Should be -5.0")

        l1, l2 = (1,-5), ('v',0)
        x_grdth, y_grdth = 0.0, -5.0

        (x,y) = get_intersection(l1,l2)

        self.assertEqual(x,x_grdth, "Should be 0.0")
        self.assertEqual(y,y_grdth, "Should be -5.0")

        l1, l2 = ('v',25), ('v',18)

        with self.assertRaises(Exception):
            get_intersection(l1,l2)

        l1, l2 = (0.0,25.0), (0.0,18.0)

        with self.assertRaises(Exception):
            get_intersection(l1,l2)

        l1, l2 = (500.0,25.0), (500.0,18.0)

        with self.assertRaises(Exception):
            get_intersection(l1,l2)
    
    def test_get_quad_from_oct_1(self):

        oct_pts_numpy = np.array([
            250,700,
            100,550,
            200,800,
            0,600,

            100,250,
            250,100,
            0,200,
            200,0,

            550,700,
            700,550,
            600,800,
            800,600,

            700,250,
            550,100,
            800,200,
            600,0
        ])

        quad_pts_numpy = get_quadrilateral_from_octagon(oct_pts_numpy)

        quad_pts_grdth = np.array(
            [100,700,0,800,100,100,0,0,700,700,800,800,700,100,800,0
        ])

        self.assertEqual(quad_pts_numpy.tolist(), quad_pts_grdth.tolist(), "")

    def test_is_valid_pred(self):

        quad_pts_numpy = np.array([
            [100,700],
            [0,800],
            [100,100],
            [0,0],
            [700,700],
            [800,800],
            [700,100],
            [800,0]],
        ).astype('float')

        symmetry = np.array(
            [100,700,0,800,100,100,0,0,700,700,800,800,700,100,800,0]
            ).reshape((8,2))

        is_valid = is_valid_prediction(quad_pts_numpy, symmetry,dist=5.0)

        self.assertEqual(is_valid, 1, "Should be 1.")

        old_offset = np.random.randint(-5, 6, size=(8,1))
        zeros = np.zeros((8,1))

        offset = np.hstack((old_offset,zeros))

        new_quad = quad_pts_numpy + offset

        is_valid = is_valid_prediction(new_quad, symmetry,dist=5.0)

        self.assertEqual(is_valid, 1, "Should be 1.")

        old_offset = np.random.randint(-5, 6, size=(8,1))
        zeros = np.zeros((8,1))
        fives = zeros + 5

        offset = np.hstack((old_offset,fives))

        new_quad = quad_pts_numpy + offset

        is_valid = is_valid_prediction(new_quad, symmetry,dist=5.0)

        self.assertEqual(is_valid, 0, "Should be 0.")




if __name__ == '__main__':
    unittest.main()