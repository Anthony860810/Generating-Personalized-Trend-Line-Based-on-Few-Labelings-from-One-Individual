# Generating-Personalized-Trend-Line-Based-on-Few-Labelings-from-One-Individual

This repo contains the code to reproduce the result in the paper:

**Petrel: Personalized Trend Line Estimation with Limited Labels from One Individual** published in PAKDD 2023.

Petrel generates customized trend lines that consider user preferences and input time series shapes. Petrel obtains users' preferred trends by asking users to draw trend lines on limited sample trends. 

## Setup
Tested under Python 3.8.13 in Ubuntu.

Install the required packages by
```
conda env create -f environment.yaml
conda activate env_tykuo
```

## File Description
Data Folder Link: https://drive.google.com/file/d/1bc2A4OKITtgTyA1ZDQAnTYl2NLcE7nDC/view?usp=sharing
* A4Benchmark: YahooS5 A4 Dataset csv file
* A4Benchmark_SimulateUser: Generate simulate User
* Done_s2_img_user: The pictures that model predict with differnt user
* Done_user: The user data  json file and pictures that user draw.
* mixer_multiple_full: Generate simulate user by GenerateSimulateUser.py
* trend: original time series dataset which include value, L1 Trend Filtering, HP Filtering, STL trend.


## Execution

### Folder Setting
```
project
│   README.md
│   evvironment.yaml    
│   pretraincnn_model
│   pretrainfc_model
│   pretraintransformer_model
│   pretrainlstm_model
│   OurMethod_model
│
│
└───A4Benchmark_SimulateUser
│   │   A4Benchmark-TS1.csv
│   │   A4Benchmark-TS2.csv
│
└───A4Benchmark
│   │   A4Benchmark-TS1.csv
│   │   A4Benchmark-TS2.csv
│   
└-───Trend
│   │   file1.json
│   │   file2.json
│   │   file3.json 
│
└───mixer_multiple_full     
│   
└───user29(User who you want to check)
│   │   file1.pdf
│   │   file2.pdf
│   │   user29.json(User draw data) 
│
│
└───s2_img_user29(User who you want to check)
    │   file1.pdf
    │   file2.pdf
    │   user29.json(User draw data) 


```

### Initialize folder
To store model output pictures
```
mkdir mixer_multiple_full
mkdir s2_img_user29
cd s2_img_user29
mkdir 1_v2 8_v2 12_v2 20_v2 21_v2 66 74 81 88 91
```
### GenerateSimulateUser
You need to run this code to generate simulate users' data before you run Pretrain model code and OurMethod code.
```
python GenerateSimulateUser.py
```

### CNN
```
python CNN.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### LSTM
```
python LSTM.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### Transformer

```
python Transformer.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### PretrainCNN
#### Train
```
python PretrainCNN_train.py --epoch 1 --lr 0.01 --batch 1000 
```
#### Finetune and Test
```
python PretrainCNN_finetune_test.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### PretrainLSTM
#### Train
```
python PretrainLSTM_train.py --epoch 1 --lr 0.01 --batch 1000
```
#### Finetune and Test
```
python PretrainLSTM_finetune_test.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### Pretrain Fully Connected
#### Train
```
python PretrainFullyConnected_train.py --epoch 1 --lr 0.01 --batch 1000
```
#### Finetune and Test
```
python PretrainFullyConnected_finetune_test.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### Pretrain Transformer
#### Train
```
python PretrainTransformer_train.py --epoch 1 --lr 0.01 --batch 1000
```
#### Finetune and Test
```
python PretrainTransformer_finetune_test.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### OurMethod
#### Train
```
python OurMethod_train.py --epoch 1 --lr 0.01 --batch 1000
```
#### Test
```
python OurMethod_test.py
```

### L1 Trend Filtering & HP Filtering & STL
```
python l1hpstl.py
```
## Citation

Please cite our work if you find Petrel useful in your research.

```
@inproceedings{
    kuo2023petrel,
    title={Petrel: Personalized Trend Line Estimation with Limited Labels from One Individual},
    author={Tong-Yi Kuo and Hung-Hsuan Chen},
    booktitle={The 27th Pacific-Asia Conference on Knowledge Discovery and Data Mining},
    year={2023}
}
```
