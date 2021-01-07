import torch
import pandas as pd
from skimage import io, transform
import os
import numpy as np
import torch.nn as nn
from torchvision import models

class FlowerPoseDataset(torch.utils.data.Dataset):
    """Flower Pose dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.bounbing_box_points = pd.read_csv(csv_file)[['filename',
                                            # 'center_U','center_V',
                                            'xyz_U_flower','xyz_V_flower',
                                            'xyZ_U_flower','xyZ_V_flower',
                                            'xYz_U_flower','xYz_V_flower',
                                            'xYZ_U_flower','xYZ_V_flower',
                                            'Xyz_U_flower','Xyz_V_flower',
                                            'XyZ_U_flower','XyZ_V_flower',
                                            'XYz_U_flower','XYz_V_flower',
                                            'XYZ_U_flower','XYZ_V_flower',
                                            'xyz_X_flower','xyz_Y_flower','xyz_Z_flower',
                                            'xyZ_X_flower','xyZ_Y_flower','xyZ_Z_flower',
                                            'xYz_X_flower','xYz_Y_flower','xYz_Z_flower',
                                            'xYZ_X_flower','xYZ_Y_flower','xYZ_Z_flower',
                                            'Xyz_X_flower','Xyz_Y_flower','Xyz_Z_flower',
                                            'XyZ_X_flower','XyZ_Y_flower','XyZ_Z_flower',
                                            'XYz_X_flower','XYz_Y_flower','XYz_Z_flower',
                                            'XYZ_X_flower','XYZ_Y_flower','XYZ_Z_flower']]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.bounbing_box_points)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.bounbing_box_points.iloc[idx, 0])
        image = io.imread(img_name)

        bbox_2d = self.bounbing_box_points.iloc[idx, 1:17]
        bbox_2d = np.array([bbox_2d])
        bbox_2d = bbox_2d.astype('float').reshape(-1, 2)

        bbox_3d = self.bounbing_box_points.iloc[idx, 17:]
        bbox_3d = np.array([bbox_3d])
        bbox_3d = bbox_3d.astype('float').reshape(-1, 3)

        sample = {'image': image, 'bbox_2d': bbox_2d, 'bbox_3d': bbox_3d}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox_2d, bbox_3d = sample['image'], sample['bbox_2d'], sample['bbox_3d']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'bbox_2d': torch.from_numpy(bbox_2d).float(),
                'bbox_3d': torch.from_numpy(bbox_3d).float()}

def custom_sigmoid(input):
    '''
    Applies a custom sigmoid function function element-wise
    so that the output lies in the interval [-0.1, 1.1]

    '''
    # return (11 * torch.exp(input) - 1) / (10 * (torch.exp(input) + 1))
    # return 1 / (1 + torch.exp(-input))
    # return ((1.2 / (1 + torch.exp(-input))) - 0.1) * 600
    # return (torch.sigmoid(input))
    return input

# x = torch.tensor([60.])
# print(custom_sigmoid(x))

class GilSigmoid(torch.nn.Module):
    '''
    Applies the custom sigmoid function element-wise:
    
        custom_sigmoid(x) = (11 * np.exp(x) - 1) / (10 * (np.exp(x) + 1))

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    # def forward(self, input):
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        # return torch.sigmoid(input)
        return custom_sigmoid(input)

class GilNet(nn.Module):

    def __init__(self):
        super().__init__()

        model_ft = models.resnet34(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 16)

        self.res = model_ft

    def forward(self,x):
        # x = torch.sigmoid(self.res(x))
        x = custom_sigmoid(self.res(x))
        # x = self.res(x)
        return x



# import datetime as dt
# import re
# import shutil

# timestamp = str(dt.datetime.now())[:19]
# timestamp = re.sub(r'[\:-]','', timestamp)
# timestamp = re.sub(r'[\s]','_', timestamp)

# dir_prefix = 'model'
# formatted_dirname = ('{}_{}'.format(dir_prefix,timestamp))
# dirname = "./models/"+formatted_dirname
# fname = "my_classes.py"

# if not os.access(dirname, os.F_OK):
#     os.mkdir(dirname, 0o700)
# shutil.copy(fname, dirname)