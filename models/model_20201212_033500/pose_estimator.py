from my_classes import FlowerPoseDataset, ToTensor, GilSigmoid, GilNet, NormalizedTensor
from my_functions import generate_octagon_torch, generate_symmetries
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

# transform = ToTensor()
transform = NormalizedTensor(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
flower_dataset = FlowerPoseDataset(csv_file=csv_file,
                                   root_dir=root_dir,
                                   transform=transform)


# dataloader = torch.utils.data.DataLoader(flower_dataset, batch_size=3,
#                         shuffle=True, num_workers=0)

# validation_split = 0.2

# dataset_len = len(flower_dataset)
# indices = list(range(dataset_len))

# # Randomly splitting indices:
# val_len = int(np.floor(validation_split * dataset_len))
# validation_idx = np.random.choice(indices, size=val_len, replace=False)
# train_idx = list(set(indices) - set(validation_idx))

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
        # best_model_wts = copy.deepcopy(model.state_dict())
        # best_acc = 0.0
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

                    symmetries = generate_symmetries(key_pts)

                    # # flatten pts
                    # key_pts = key_pts.view(key_pts.size(0), -1)

                    images = images.to(device)
                    # key_pts = key_pts.to(device)
                    symmetries = symmetries.to(device)

                    # print(symmetries.size())

                    # # wrap them in a torch Variable
                    # images, key_pts = Variable(images), Variable(key_pts)

                    # # convert variables to floats for regression loss
                    # key_pts = key_pts.type(torch.FloatTensor)
                    # images = images.type(torch.FloatTensor)


                    # forward pass to get outputs
                    output_pts = model(images)

                    # print(output_pts.size())

                    # output_pts_octagon = output_pts

                    output_pts_octagon = generate_octagon_torch(output_pts)
                    output_pts_octagon = output_pts_octagon.view(output_pts_octagon.size(0), 1, -1)

                    # print(output_pts_octagon.size())



                    # print(output_pts)
                    # print('{} {}'.format(phase, output_pts), file=f)

                    # output_pts *= 600
                    # print(output_pts)
                    # print(output_pts_octagon)
                    # print(symmetries)


                    symmetry_loss = ((output_pts_octagon - symmetries)**2)
                    symmetry_loss = torch.mean(symmetry_loss,dim=2)
                    symmetry_loss = torch.min(symmetry_loss, dim=1).values
                    symmetry_loss = torch.mean(symmetry_loss)

                    # calculate the loss between predicted and target keypoints
                    # loss = criterion(output_pts, key_pts)

                    loss = symmetry_loss
                    # print(loss.item())
                    # print(running_loss)

                    # zero the parameter (weight) gradients
                    optimizer.zero_grad()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # update the weights
                        optimizer.step()
                    # print(loss.item(), file=f)

                    # print loss statistics
                    running_loss += loss.item() * images.size(0)
            
                epoch_loss = running_loss / data_lengths[phase]

                # If we achieve small validation loss save the model
                if phase == 'val' and epoch_loss < prev_min_val:
                    prev_min_val = epoch_loss
                    torch.save(model.state_dict(), dirname+"/model_min_val_loss") 

                print('{} Loss: {:.4f}'.format(phase, epoch_loss), file=f)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                if phase == 'train':
                    scheduler.step()

                # # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), file=f) 
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)) 
        return model

# model_ft = models.resnet18(pretrained=True)
# # print(model_ft)
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 18)

# model = nn.Sequential(
#     model_ft,
#     # nn.Sigmoid()
#     GilSigmoid()
# )


model = GilNet()

model = model.to(device)

criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.1)





model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=30)
# print(model)
# from torchsummary import summary

# # summary(model_ft)

# summary(model)

torch.save(model.state_dict(), dirname+"/model")