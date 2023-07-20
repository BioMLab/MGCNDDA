# MGCNDDA
## Paper "MGCNDDA"
### 'data' directory
Contain PREDICT-Dataset, LAGCN-Dataset and LRSSL-Dataset

### 'Experimental Results' directory
1. Contain the prediction results of each baseline models is training on three dataset
2. Please refer the code of MGCNDDA [here](https://github.com/studyjob/MGCNDDA);

### directory and run
1. To predict drug-disease associations by MGCNDDA, run
  - python main.py in folder of data_in_PREDICT to get the experimental results on dataset PREDICT.
  - python main_LAGCN.py in folder of data_in_LAGCN to get the experimental results on dataset LAGCN.
  - python main_LRSSL.py in folder of data_in_LRSSL to get the experimental results on dataset LRSSL.
2. Five times and ten-fold cross-validation was performed on all three data sets.

### Requirements
Python == 3.7.13  
PyTorch == 1.10.0  
numpy == 1.19.0  
torch_geometric.nn == 2.0.4  
cuda == 11.4

### Contacts
Any questions or comments, please send an email to Li (lyl_0727@126.com).
