

This is an official implementation of [TS-UNet: A Multiscale U-Net Architecture for Long-Term Multivariate Time Series Forecasting ].

### Prerequisites

Ensure you are using Python 3.9 and install the necessary dependencies by running:
```
pip install -r requirements.txt
```

### 1. Data Preparation
Download data from [AutoFormer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Put all data into a seperate folder `./dataset` and make sure it has the following structure:
```
dataset
├── electricity.csv
├── ETTh1.csv
│── ETTh2.csv
│── ETTm1.csv
│── ETTm2.csv
├── traffic.csv
└── weather.csv

```

<!-- 2. Training. All training scripts are in the folder `./scripts`. If you want to train model for ETTh1 dataset, simply run the following command:
    ```
    sh ./scripts/ETTh1.sh
    ```
    When the training is done, you can find model in `./checkpoints`, visualization result in `./test_results`, npy result in `./results`, quantative result in `./results.txt`. -->


### 2. Training

The training scripts for all datasets are located in the `./scripts` directory. 

**To train a model using the ETTh1 dataset:**
1. Navigate to the repository's root directory.
2. Execute the following command:
    ```bash
    sh ./scripts/ETTh1.sh
    ```

**Upon completion of the training:**
- The trained model will be saved in the `./checkpoints` directory.
- Visualization outputs can be found in `./test_results`.
- Numerical results in `.npy` format are located in `./results`.
- A summary of the quantitative metrics is available in `./results.txt`.


### Citation
If you find this repo useful, please cite our paper as follows:

```


```

### Contact
If you have any questions, please contact us or submit an issue.

### Acknowledgement
We appreciate the following repo for their code and dataset:
- https://github.com/thuml/Time-Series-Library
- https://github.com/thuml/Autoformer
