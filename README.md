# 時系列感性パラメータを用いたハーモナイゼーションシステム


## データダウンロード

HLSD 
- download raw data at: https://github.com/wayne391/lead-sheet-dataset
- save raw data as: ./HLSD/dataset/event/a/a_day_...
- run command: python3 process_data.py --dataset HLSD 

outputs:
1) saves npy files for the parsed features (saved directory ex: ./HLSD/output/~) 
2) saves train/val/test batches (saved directory ex: ./HLSD/exp/train/batch/~)
3) saves h5py dataset for the train/val/test batches (saved filename ex: ./HLSD_train.h5)


## TRAIN MODEL
  
1)python3 trainer.py HLSD rVTHarm 

outputs:
model parameters/losses checkpoints (saved filename ex: ./trained/rVTHarm_HLSD)

2)python3 process_data_LSTM_4bar --dataset HLSD

outputs:
1) saves npy files for the parsed features (saved directory ex: ./LSTM/output/~) 
2) saves train/val/test batches (saved directory ex: ./LSTM/exp/train/batch/~)
3) saves h5py dataset for the train/val/test batches (saved filename ex: ./LSTM_train.h5)

3)python3 trainer_SurpriseNet_4bar.py

outputs:
model parameters/losses checkpoints (saved filename ex: ./trained/SurpriseNet_raw_meian_HLSD)


## TEST MODEL 
python3 test.py [dataset] [song_ind] [start_point] [model_name] [device_num] [alpha]

* [dataset] -> CMD or HLSD 
* [song_ind] -> index of test files 
* [start_point] -> half-bar index ex) if you want to start at the 1st measure: start_point=0 / at the second half of the 1st measure: start_point=1
* [model_name] -> STHarm / VTHarm / rVTHarm 
* [device_num] -> number of CUDA_DEVICE
* [alpha] -> alpha value for rVTHarm / if not fed (different model), ignored

outputs:
1) prints out quantitative metric results -> CHE, CC, CTR, PCS, CTD, MTD / LD, TPSD, DICD 
2) saves generated(sampled) MIDI / GT MIDI (saved filename ex: ./Sampled__*.mid, ./GT__*.mid)
