# MVSGDR

Multi-view stacked graph convolutional network for Drug Repositioning

# Introduction



# Requirements

- python 3.9.13
- cudatoolkit 11.3.1
- pytorch 1.10.0
- dgl 0.9.0
- numpy 1.23.1
- scikit-learn 0.24.2
- metis 0.2a5
- torch_geometric 2.0.4

#  Code

- data_preprocess.py: Methods of data processing
- metric.py: Metrics calculation
- BIST.py: Code of Bi-level subgraph transformer module
- MVS.py: Code of Multi-view stacked module
- model.py: Model of MVSGDR
- main.py: Train the model

# Datasets

| Statistics     | G-dataset | C-dataset | Lrssl  | L-dataset |
| -------------- | --------- | --------- | ------ | --------- |
| # Diseases     | 313       | 409       | 681    | 598       |
| # Drugs        | 593       | 663       | 763    | 269       |
| # Associations | 1933      | 2532      | 3051   | 18416     |
| # Density      | 0.0104    | 0.0093    | 0.0058 | 0.1144    |

# Usage

Execute ```python main.py``` 
