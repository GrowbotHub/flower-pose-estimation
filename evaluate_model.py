from my_classes import GilNet, FlowerPoseDataset, MyTransformation, UnNormalize
from my_functions import identity,\
                         draw,\
                         generate_symmetries_quad,\
                         S_Loss_MSE,\
                         generate_octagon,\
                         generate_symmetries,\
                         get_quadrilateral_from_octagon,\
                         is_valid_prediction,\
                         PLOSS,\
                         L2_dist
from my_constants import CAMERA_INTRINSIC_MATRIX,\
                         IMAGENET_MEAN, IMAGENET_STD,\
                         QUADRILATERAL_BOX_PTS,\
                         OCTAGON_BOX_PTS
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler,\
                                     SequentialSampler
import cv2
from torchvision import transforms
import torch.nn as nn
import pandas as pd
import sys

IMAGENET_norm = transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
unnorm = UnNormalize()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
print("Evaluating on: {}".format(device))

#==============================================================================
# Configure necessary dictionairies
#==============================================================================

# All model filenames
model_fnames = [
    "model_20201129_113012", 
    "model_20201130_223048", 
    "model_20201203_003241", 
    "model_20201205_230901", 
    "model_20201207_203315", 
    "model_20201207_203315_30",
    "model_20201210_162514", 
    "model_20201212_033500",
    "model_20201216_224455",
    "model_20201218_140110",
    "model_20201219_125500_10",
    "model_20201219_125500_17",
    "model_20201219_125500_30",
    "model_20201226_204651_40",
    "model_20201226_204651_50"
]

# Define cofigurations for each model filename
models_config = {
    "model_20201129_113012": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": nn.MSELoss(), 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201130_223048": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": nn.MSELoss(), 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201203_003241": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201205_230901": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201207_203315": {"model_name": "resnet34", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201207_203315_30": {"model_name": "resnet34", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201210_162514": {"model_name": "resnet18", 
                              "num_outputs": 32, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201212_033500": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(True,IMAGENET_norm),
                              "rotation":(False,"")},
    "model_20201216_224455": {"model_name": "resnet18",
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(True,identity),
                              "rotation":(False,"")},
    "model_20201218_140110": {"model_name": "resnet18",
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(True,identity),
                              "rotation":(False,"")},
    "model_20201219_125500_10": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201219_125500_17": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201219_125500_30": {"model_name": "resnet18", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(False,None),
                              "rotation":(False,"")},
    "model_20201226_204651_40": {"model_name": "resnet34", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(True,IMAGENET_norm),
                              "rotation":(False,"")},
    "model_20201226_204651_50": {"model_name": "resnet34", 
                              "num_outputs": 16, 
                              "out_func": identity, 
                              "citerion": S_Loss_MSE, 
                              "normalization":(True,IMAGENET_norm),
                              "rotation":(False,"")}
}


# Map model filenames to the model objects themselves
models = {}

for fname in model_fnames:
    model_config = models_config[fname]
    model_name = model_config["model_name"]
    num_outputs = model_config["num_outputs"]
    out_func = model_config["out_func"]
    model = GilNet(model=model_name,pretrained=False,num_outputs=num_outputs,out_func=out_func)
    model_path = "./models/"+fname+"/model"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.train(False)
    models[fname] = model

# Dictionairies to map every model filename to its metric value
dict_MSE_quad = {}
dict_MSE_oct = {}
dict_S_Loss_Quad = {}  
dict_S_Loss_Oct = {}  
dict_reproj_MSE_quad = {}
dict_reproj_MSE_oct = {}
dict_reproj_S_Loss_Quad = {}  
dict_reproj_S_Loss_Oct = {}
dict_correct_5_reproj = {}
dict_correct_10_reproj = {}
dict_correct_15_reproj = {}
dict_reproj_PLOSS_quad = {}
dict_reproj_PLOSS_oct = {}
dict_avg_dist = {}


#==============================================================================
# Configure where to load dataset from
#==============================================================================

path_to_models = "./models/"
model_variant = "model"


path_to_mount = "/path/to/mount/"

if cuda_available:
    root_dir = "../../cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/natural"
    csv_file = "../../cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/flowers_dataset.csv"
else:
    root_dir = path_to_mount+"/cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/natural"
    csv_file = path_to_mount+"/cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/flowers_dataset.csv"


transform = MyTransformation(image_transforms=[transforms.ToTensor()], rotation=False)

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

# Define the sampler
test_idx = permutation[validation_end:]
test_sampler = SubsetRandomSampler(test_idx)
test_loader = torch.utils.data.DataLoader(flower_dataset, sampler=test_sampler, batch_size=1)

data_loaders = {"test": test_loader}
data_lengths = {"test": len(test_idx)}

print("Number of images to test: {}".format(data_lengths["test"]))

percent = "%"

for idx,data in enumerate(data_loaders["test"]):
    print("\rImages %.3d%s done." %(int(((idx+1)/data_lengths["test"]) * 100),percent), end='')
    image = data['image']
    bbox_2d = data['bbox_2d']
    bbox_3d = data['bbox_3d']

    image = image.to(device)

    ### COMPUTE OCTAGON

    oct_from_grdth = generate_octagon(bbox_2d).to(device)

    ### COMPUTE QUADRILATERAL SYMMETRIES

    quad_symmetries_from_grdth = generate_symmetries_quad(bbox_2d).to(device)

    quad_symmetries_from_grdth_numpy = generate_symmetries_quad(bbox_2d).detach().numpy()

    ### COMPUTE OCTAGON SYMMETRIES

    oct_symmetries_from_grdth = generate_symmetries(bbox_2d).to(device)


    bbox_2d_numpy = bbox_2d.detach().numpy().ravel()
    bbox_3d_numpy = bbox_3d.detach().numpy().ravel()
    bbox_3d_numpy = bbox_3d_numpy.reshape((8,3))

    bbox_2d = bbox_2d.to(device)

    for fname in model_fnames:
        model = models[fname]
        image_ = image.detach().clone()
        (apply_norm,norm) = models_config[fname]["normalization"]
        if apply_norm:
            image_ = torch.unsqueeze(norm(image_.squeeze()),0)
        else: 
            image_.mul_(255)

        output_2d = model(image_)

        ### CASE 1: 16 outputs
        if output_2d.size(1) == QUADRILATERAL_BOX_PTS:
            quad_from_out = output_2d
            oct_from_out = generate_octagon(output_2d)

        ### CASE 2: 32 outputs
        elif output_2d.size(1) == OCTAGON_BOX_PTS:
            oct_from_out = output_2d
            output_2d_tmp = output_2d.detach().clone().cpu()
            quad_from_out = torch.unsqueeze(get_quadrilateral_from_octagon(output_2d_tmp.squeeze(),quad_symmetries_from_grdth_numpy),0).to(device)

        ### REGULAR MSE

        dict_MSE_quad[fname] = dict_MSE_quad.get(fname, 0.0) + PLOSS(bbox_2d.squeeze(), quad_from_out.squeeze()).item()
        dict_MSE_oct[fname] = dict_MSE_oct.get(fname, 0.0) + PLOSS(oct_from_grdth.squeeze(), oct_from_out.squeeze()).item()        
        
        ### S_LOSS_MSE FOR QUAD AND OCT

        quad_from_out_dim = quad_from_out.view(1, 1, QUADRILATERAL_BOX_PTS)
        dict_S_Loss_Quad[fname] = dict_S_Loss_Quad.get(fname, 0.0) + S_Loss_MSE(quad_from_out_dim, quad_symmetries_from_grdth).item()

        oct_from_out_dim = oct_from_out.view(1, 1, OCTAGON_BOX_PTS)
        dict_S_Loss_Oct[fname] = dict_S_Loss_Oct.get(fname, 0.0) + S_Loss_MSE(oct_from_out_dim, oct_symmetries_from_grdth).item()

        ### REPROJECTION PART AND CHECK VALIDITY PART
    
        quad_from_out_numpy = quad_from_out.cpu().detach().numpy()
        quad_symmetries_from_out_numpy = generate_symmetries_quad(quad_from_out_numpy).squeeze()
        quad_from_out_numpy = quad_from_out_numpy.reshape((QUADRILATERAL_BOX_PTS//2,2))

        ### IS VALID PREDICTION

        min_reproj_PLOSS = sys.float_info.max
        for symmetry_quad_numpy in quad_symmetries_from_out_numpy:
            symmetry_quad_numpy = symmetry_quad_numpy.reshape((8,2))
            _ , rvecs_try, tvecs_try = cv2.solvePnP(bbox_3d_numpy, symmetry_quad_numpy, CAMERA_INTRINSIC_MATRIX, None)
            reproj_quad_pts_try, _ = cv2.projectPoints(bbox_3d_numpy, rvecs_try, tvecs_try, CAMERA_INTRINSIC_MATRIX, None)
            reproj_quad_pts_try_ravel = reproj_quad_pts_try.ravel()
            current_reproj_PLOSS = PLOSS(reproj_quad_pts_try_ravel, bbox_2d_numpy)
            if current_reproj_PLOSS < min_reproj_PLOSS:
                min_reproj_PLOSS = current_reproj_PLOSS
                reproj_quad_pts = reproj_quad_pts_try

        dict_correct_5_reproj[fname] = dict_correct_5_reproj.get(fname, 0) + is_valid_prediction(reproj_quad_pts.reshape((8,2)), bbox_2d_numpy.reshape((8,2)), dist=5.0)
        dict_correct_10_reproj[fname] = dict_correct_10_reproj.get(fname, 0) + is_valid_prediction(reproj_quad_pts.reshape((8,2)), bbox_2d_numpy.reshape((8,2)), dist=10.0)
        dict_correct_15_reproj[fname] = dict_correct_15_reproj.get(fname, 0) + is_valid_prediction(reproj_quad_pts.reshape((8,2)), bbox_2d_numpy.reshape((8,2)), dist=15.0)

        dict_avg_dist[fname] = dict_avg_dist.get(fname, 0) + L2_dist(reproj_quad_pts.reshape((8,2)), bbox_2d_numpy.reshape((8,2)))

        reproj_quad_pts_torch = torch.from_numpy(reproj_quad_pts.ravel().reshape(1,QUADRILATERAL_BOX_PTS))
        reproj_oct_pts_torch = generate_octagon(reproj_quad_pts_torch).to(device)
        reproj_quad_pts_torch = reproj_quad_pts_torch.to(device)

        ### REPROJECTION REGULAR MSE

        dict_reproj_MSE_quad[fname] = dict_reproj_MSE_quad.get(fname, 0.0) + nn.MSELoss()(bbox_2d, reproj_quad_pts_torch).item()
        dict_reproj_MSE_oct[fname] = dict_reproj_MSE_oct.get(fname, 0.0) + nn.MSELoss()(oct_from_grdth, reproj_oct_pts_torch).item()

        ### REPROJECTION S_LOSS_MSE FOR QUAD AND OCT

        reproj_quad_pts_torch_dim = reproj_quad_pts_torch.view(1, 1, QUADRILATERAL_BOX_PTS)
        dict_reproj_S_Loss_Quad[fname] = dict_reproj_S_Loss_Quad.get(fname, 0.0) + S_Loss_MSE(reproj_quad_pts_torch_dim, quad_symmetries_from_grdth).item()

        reproj_oct_pts_torch_dim = reproj_oct_pts_torch.view(1, 1, OCTAGON_BOX_PTS)
        dict_reproj_S_Loss_Oct[fname] = dict_reproj_S_Loss_Oct.get(fname, 0.0) + S_Loss_MSE(reproj_oct_pts_torch_dim, oct_symmetries_from_grdth).item()

        ### FOR PLOSS

        reproj_oct_pts_torch_PLOSS = reproj_oct_pts_torch.squeeze()
        reproj_quad_pts_torch_PLOSS = reproj_quad_pts_torch.squeeze()
        oct_from_grdth_PLOSS = oct_from_grdth.squeeze()
        bbox_2d_PLOSS = bbox_2d.squeeze()

        dict_reproj_PLOSS_quad[fname] = dict_reproj_PLOSS_quad.get(fname, 0.0) + PLOSS(reproj_quad_pts_torch_PLOSS, bbox_2d_PLOSS).item()
        dict_reproj_PLOSS_oct[fname] = dict_reproj_PLOSS_oct.get(fname, 0.0) + PLOSS(reproj_oct_pts_torch_PLOSS, oct_from_grdth_PLOSS).item()


### PREPARE FOR WRITE TO CSV FILE

MSE_quad = []
MSE_oct = []
S_Loss_Quad = []  
S_Loss_Oct = []  
reproj_MSE_quad = []
reproj_MSE_oct = []
reproj_S_Loss_Quad = []  
reproj_S_Loss_Oct = [] 
correct_5_reproj = []
correct_10_reproj = []
correct_15_reproj = []
reproj_PLOSS_quad = []
reproj_PLOSS_oct = []
avg_dist = []


for fname in model_fnames:
    MSE_quad.append(dict_MSE_quad[fname] / data_lengths["test"])
    MSE_oct.append(dict_MSE_oct[fname] / data_lengths["test"])
    S_Loss_Quad.append(dict_S_Loss_Quad[fname] / data_lengths["test"])  
    S_Loss_Oct.append(dict_S_Loss_Oct[fname] / data_lengths["test"])  
    reproj_MSE_quad.append(dict_reproj_MSE_quad[fname] / data_lengths["test"])  
    reproj_MSE_oct.append(dict_reproj_MSE_oct[fname] / data_lengths["test"])  
    reproj_S_Loss_Quad.append(dict_reproj_S_Loss_Quad[fname] / data_lengths["test"])  
    reproj_S_Loss_Oct.append(dict_reproj_S_Loss_Oct[fname] / data_lengths["test"])
    correct_5_reproj.append(100 * (dict_correct_5_reproj[fname] / data_lengths["test"]))
    correct_10_reproj.append(100 * (dict_correct_10_reproj[fname] / data_lengths["test"]))
    correct_15_reproj.append(100 * (dict_correct_15_reproj[fname] / data_lengths["test"]))
    reproj_PLOSS_quad.append(dict_reproj_PLOSS_quad[fname] / data_lengths["test"])
    reproj_PLOSS_oct.append(dict_reproj_PLOSS_oct[fname] / data_lengths["test"])
    avg_dist.append(dict_avg_dist[fname] / data_lengths["test"])

results = {
    "fnames": model_fnames, 
    "MSE_quad": MSE_quad, 
    "MSE_oct": MSE_oct, 
    "S_Loss_Quad": S_Loss_Quad,
    "S_Loss_Oct": S_Loss_Oct,
    "reproj_MSE_quad": reproj_MSE_quad,
    "reproj_MSE_oct": reproj_MSE_oct,
    "reproj_S_Loss_Quad": reproj_S_Loss_Quad,
    "reproj_S_Loss_Oct": reproj_S_Loss_Oct,
    "correct_5_reproj": correct_5_reproj,
    "correct_10_reproj": correct_10_reproj,
    "correct_15_reproj": correct_15_reproj,
    "reproj_PLOSS_quad":reproj_PLOSS_quad,
    "reproj_PLOSS_oct":reproj_PLOSS_oct,
    "avg_dist": avg_dist
}


### WRITE TO CSV FILE 

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv("test_results.csv",index=False)