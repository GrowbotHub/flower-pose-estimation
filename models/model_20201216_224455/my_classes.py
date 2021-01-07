import torch
import pandas as pd
from skimage import io, transform
import os
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from my_functions import rotate_box, identity
import random

class FlowerPoseDataset(torch.utils.data.Dataset):

    """Flower Pose dataset."""

    def __init__(self, csv_file, root_dir, transform=None):

        """ Init method.

        Parameters
        ----------
        csv_file : string 
            Path to the csv file with annotations.
        root_dir : string 
            Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """

        self.data = pd.read_csv(csv_file)[['filename',
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
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        image = io.imread(img_name)

        bbox_2d = self.data.iloc[idx, 1:17]
        bbox_2d = np.array(bbox_2d)
        bbox_2d = bbox_2d.astype('float')

        bbox_3d = self.data.iloc[idx, 17:]
        bbox_3d = np.array(bbox_3d)
        bbox_3d = bbox_3d.astype('float')

        sample = {'image': image, 'bbox_2d': bbox_2d, 'bbox_3d': bbox_3d}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MyTransformation(object):

    """Custom transformation class that transforms a 
    sample {numpy RGB image, bbox_2d, bbox_3d} into 
    its Tensor counterpart.
    """

    def __init__(self,image_transforms=[
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                                        ],
                      rotation=False):
                      
        """Init method.

        Parameters
        ----------
        image_transforms : list of `torch.transforms.Transform` objects
            List of transforms to compose and apply on the numpy array image
        rotation : bool
            If 'True' each image is rotated counter-clockwise by one of 0, 90, 
            180 or 270 degrees chosen at random. The 2D pixel boudning 
            box (bbox_2d) is rotated as well to take into account the change.
        """

        self.transform = transforms.Compose(image_transforms)
        self.rotation = rotation

    def __call__(self, sample):

        """Applies the transformations defined at the time of the 
        `MyTransformation` object's creation.
        """

        image, bbox_2d, bbox_3d = sample['image'], sample['bbox_2d'], sample['bbox_3d']
        image = self.transform(image)

        if self.rotation:
            rotations = random.randint(0,3)
            image = torch.rot90(image, rotations, [1,2]) 
            bbox_2d = rotate_box(box=bbox_2d.reshape((8,2)),rotations=rotations).reshape(-1)

        return {'image': image,
                'bbox_2d': torch.from_numpy(bbox_2d).float(),
                'bbox_3d': torch.from_numpy(bbox_3d).float()}

class UnNormalize(object):

    """Unnormalizes a Tensor given channel-wise 
    mean and satandard deviation lists.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Parameters
        ----------
        tensor : `torch.Tensor`
            Tensor image of size (C, H, W) to be normalized.

        Returns
        -------
        tensor : `torch.Tensor`
            Unnormalized Tensor image.
        """
        # This method produces dodgy result ???
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.mul_(s).add_(m)

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        tensor.mul_(std).add_(mean)
        return tensor

class GilNet(nn.Module):

    """Custom `nn.Module` class representing the network used
    during training.
    """

    def __init__(self, model='resnet18', pretrained=True, num_outputs=16, out_func=identity):

        """Init method.

        Parameters
        ----------
        model : string
            name of model architecture. Currently only 'resnet18'
            and 'resnet34' are supported.
        pretrained : bool
            If 'True' the specified model will be pre-trained.
        num_outputs : int
            The number of outputs of the final fully connected layer
            of the network. Currently only 16 (quadrilateral bounding
            box) and 32 (octagon bounding box) are supported.
        out_func : function
            The function applied to the outputs of the final fully
            connected layer. Default is identity function.
        """

        assert model in ['resnet18', 'resnet34'], "Incompatible model type!"
        super().__init__()
        self.out_func = out_func
        if model == 'resnet18':
            model_ft = models.resnet18(pretrained=pretrained) 
        elif model == 'resnet34':
            model_ft = models.resnet34(pretrained=pretrained)
        else:
            model_ft = None
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_outputs)
        self.res = model_ft

    def forward(self,x):
        x = self.out_func(self.res(x))
        return x

