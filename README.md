# Model Training And Evaluation

Train models on the synthetic dataset and evaluate them.

## Getting Started

### Prerequisites

The librairies needed can be installed by creating a new environment with:
```
conda create --name model_train -c anaconda pytorch-gpu torchvision opencv
```
Then 
```
conda activate model_train
```
And install pandas, scikit-image and scikit-learn
```
conda install pandas
conda install -c anaconda scikit-learn
conda install scikit-image
```
## Running the code

The file to train a model is `pose_estimator.py`. All functions are documented and defined in `my_functions.py`. All classes are documented and defined in `my_classes.py`. A few constants are defined in `my_constants.py`. There is a test suite `test_suite.py` which tests the correctness of some of the functions. The script `evaluate_model.py` can be used to evaluate the models and store the results in a CSV file. The models included here are the best performing ones from the report (6), (10) and (13). Each of their respective descriptions and the original scripts used to create them can also be found in their respective subfolders.

## Authors

* **Gil Tinde** 
