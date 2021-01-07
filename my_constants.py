import numpy as np

IMAGE_HEIGHT = 600
IMAGE_WIDTH = 600
OCTAGON_BOX_PTS = 32
QUADRILATERAL_BOX_PTS = 16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GILNET_MEAN = [0.3395, 0.3457, 0.2613]
GILNET_STD = [0.2035, 0.2037, 0.1777]
CAMERA_YFOV = (np.pi / 3.0)
CAMERA_ZNEAR = 0.01
CAMERA_INTRINSIC_MATRIX = np.array([[100 * np.float_power(3.0,1.5),               0              , 300],
                                    [             0               , 100 * np.float_power(3.0,1.5), 300],
                                    [             0               ,               0              ,  1 ]])