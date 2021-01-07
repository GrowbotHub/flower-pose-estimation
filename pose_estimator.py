from my_classes import FlowerPoseDataset, MyTransformation, GilNet
from my_functions import generate_octagon,\
                         generate_symmetries,\
                         S_Loss_MSE,\
                         generate_symmetries_quad,\
                         identity
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils, datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import numpy as np
import copy
import time
import datetime as dt
import re
import shutil
import os
import sys

plt.ion()

timestamp = str(dt.datetime.now())[:19]
timestamp = re.sub(r'[\:-]','', timestamp)
timestamp = re.sub(r'[\s]','_', timestamp)

dir_prefix = 'model'
formatted_dirname = ('{}_{}'.format(dir_prefix,timestamp))
dirname = "./models/"+formatted_dirname

if not os.access(dirname, os.F_OK):
    os.mkdir(dirname, 0o700)

fname = "pose_estimator.py"
shutil.copy(fname, dirname)
fname = "my_classes.py"
shutil.copy(fname, dirname)
fname = "my_functions.py"
shutil.copy(fname, dirname)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
print("Training on: {}".format(device))

if cuda_available:
    root_dir = "../../cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/natural"
    csv_file = "../../cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/flowers_dataset.csv"
else:
    root_dir = "../3D/outputs/pose_testing/natural"
    csv_file = "../3D/outputs/pose_testing/flowers_dataset.csv"

transform = MyTransformation(rotation=False)
flower_dataset = FlowerPoseDataset(csv_file=csv_file,
                                   root_dir=root_dir,
                                   transform=transform)

#==============================================================================
# Split set into train, validation and test sets
#==============================================================================

# Define the split ratios. The test set is inferred from what remains
# after train and validation sets have been picked. Therefore, test_split 
# is never actually used.
train_split = 0.8
validation_split = 0.1
test_split = 0.1

# Given a seed (in order to reproduce the same split accross trainings),
# a random split according to the ratios is created.
seed = 123
np.random.seed(seed)
dataset_len = len(flower_dataset)
permutation = np.random.permutation(range(dataset_len))

train_end = int(train_split * dataset_len)
validation_end = int(validation_split * dataset_len) + train_end

train_idx = permutation[:train_end]
validation_idx = permutation[train_end:validation_end]
test_idx = permutation[validation_end:]

# Define the samplers for each phase based on the random indices:
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(flower_dataset, sampler=train_sampler, batch_size=19) # 52459 = 11 × 19 × 251
validation_loader = torch.utils.data.DataLoader(flower_dataset, sampler=validation_sampler) # 6557 = 79 * 83
data_loaders = {"train": train_loader, "val": validation_loader}
data_lengths = {"train": len(train_idx), "val": len(validation_idx)}

#==============================================================================
# Define train function
#==============================================================================

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    with open(dirname+'/out.txt', 'w') as f:
        since = time.time()
        prev_min_val = sys.float_info.max

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10, file=f)
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for data in data_loaders[phase]:

                    # get the input images and their corresponding labels
                    images = data['image']
                    key_pts = data['bbox_2d']

                    symmetries = generate_symmetries_quad(key_pts)

                    images = images.to(device)
                    symmetries = symmetries.to(device)

                    # forward pass to get outputs
                    output_pts = model(images)

                    # output_pts_octagon = generate_octagon(output_pts)
                    output_pts_quad = output_pts.view(output_pts.size(0), 1, -1)

                    loss = criterion(output_pts_quad, symmetries)

                    # zero the parameter (weight) gradients
                    optimizer.zero_grad()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # update the weights
                        optimizer.step()

                    # print loss statistics
                    running_loss += loss.item() * images.size(0)
            
                epoch_loss = running_loss / data_lengths[phase]

                # If we achieve small validation loss save the model
                if phase == 'val' and epoch_loss < prev_min_val:
                    prev_min_val = epoch_loss
                    torch.save(model.state_dict(), dirname+"/model_min_val_loss") 

                if epoch == 14 and phase == 'val':
                    torch.save(model.state_dict(), dirname+"/model_epoch_17") 

                # if epoch == 9 and phase == 'val':
                #     torch.save(model.state_dict(), dirname+"/model_epoch_10") 

                print('{} Loss: {:.4f}'.format(phase, epoch_loss), file=f)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                if phase == 'train':
                    scheduler.step()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), file=f) 
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)) 
        return model

### Train a model further
# fname = "model_20201218_140110"
# model = GilNet(model="resnet18",pretrained=False,num_outputs=16,out_func=identity)
# model_path = "./models/"+fname+"/model"
# model.load_state_dict(torch.load(model_path, map_location=device))

### Train a new model
model = GilNet(model='resnet18',pretrained=True,num_outputs=16,out_func=lambda x: x)

model = model.to(device)
criterion = S_Loss_MSE

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)

shutil.copy(fname, dirname)

torch.save(model.state_dict(), dirname+"/model")