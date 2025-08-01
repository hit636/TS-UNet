This is an official implementation of TS-UNet: A Multiscale U-Net Architecture for Long-Term Multivariate Time Series Forecasting

Prerequisites
Ensure you are using Python 3.9 and install the necessary dependencies by running:

pip install -r requirements.txt
1. Data Preparation
Download data from AutoFormer. Put all data into a seperate folder ./dataset and make sure it has the following structure:

dataset
├── electricity.csv
├── ETTh1.csv
│── ETTh2.csv
│── ETTm1.csv
│── ETTm2.csv
├── traffic.csv
└── weather.csv

2. Training
The training scripts for all datasets are located in the ./scripts directory.

To train a model using the ETTh1 dataset:

Navigate to the repository's root directory.
Execute the following command:
sh ./scripts/ETTh1.sh
Upon completion of the training:

The trained model will be saved in the ./checkpoints directory.
Visualization outputs can be found in ./test_results.
Numerical results in .npy format are located in ./results.
A summary of the quantitative metrics is available in ./results.txt.
Citation

Contact
If you have any questions, please contact us or submit an issue.

Acknowledgement
We appreciate the following repo for their code and dataset:

https://github.com/thuml/Time-Series-Library
https://github.com/thuml/Autoformer
