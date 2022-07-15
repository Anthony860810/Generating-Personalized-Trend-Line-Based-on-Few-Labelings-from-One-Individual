# Generating-Personalized-Trend-Line-Based-on-Few-Labelings-from-One-Individual

## Setup
Tested under Python 3.8.13 in Ubuntu.

Install the required packages by
```
conda env create -f environment.yaml 
```


## File Description
Data Folder Link: https://drive.google.com/drive/folders/1w4kPEy4SR8acnEaMqhj7-NhbGl4msCv7?usp=sharing
* A4Benchmark: YahooS5 A4 Dataset csv file
* Done_s2_img_user: The pictures that model predict with differnt user
* Done_user: The user data  json file and pictures that user draw.
* mixer_multiple_full: Generate simulate user by GenerateSimulateUser.py
* trend: original time series dataset which include value, L1 Trend Filtering, HP Filtering, STL trend.


## Start

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

```
#### Finetune and Test
```
python PretrainLSTM_finetune_test.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### Pretrain Fully Connected
#### Train
```

```
#### Finetune and Test
```
python PretrainFullyConnected_finetune_test.py --epoch 3 --lr 0.01 --batch 10 --user 29
```

### Pretrain Transformer
#### Train
```


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