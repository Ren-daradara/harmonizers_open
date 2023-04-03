import numpy as np
import math
import os
import gc
import sys 
sys.path.append("./utils")
sys.path.append("./models")
import time
from glob import glob
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd 
import torch.optim as optim
import random
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Manager

from models import STHarm, VTHarm ,LSTM ,SurpriseNet, SurpriseNet_S, SurpriseNet_raw_meian
from models.STHarm import Mask, Compress
from process_data import FeatureIndex
from utils.parse_utils import *   
from data_loader import *
from tqdm import tqdm


## TRAINER ##
def main(dataset=None,
         exp_name=None,
         checkpoint_num=0,
         device_num=0,
         batch_size=64, # 128
         batch_size_val=64, # 128 
         total_epoch=10,
         hidden=256,
         n_layers=4):

    def loss_fn(loss_function, logp, target, length, mean, log_var, anneal_function, step, k, x0):
        # Negative Log Likelihood
        logpp=logp.permute(0,2,1)
        targety = torch.where(target == 72, random.randint(0, 71), target)
        NLL_loss = loss_function(logpp, targety)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight
    
    def kl_anneal_function(anneal_function, step, k, x0):
            if anneal_function == 'logistic':
                return float(1 / (1 + np.exp(-k * (step - x0))))
            elif anneal_function == 'linear':
                return min(1, step/x0) 

    start_time = time.time()
    
    # LOAD DATA
    datapath = 'C:\\Users\\barut\\harmonizers_transformer_self'
    # result_path = os.path.join(datapath, 'results')
    model_path = os.path.join(datapath, 'trained')
    train_data = os.path.join(datapath, '{}_train.h5'.format("LSTM"))
    val_data = os.path.join(datapath, '{}_val.h5'.format("LSTM"))
    FI = FeatureIndex("HLSD")
    #process_data内のFI

    # LOAD DATA LOADER 
    CustomDataset = LSTMDataset
    PadCollate = LSTMPadCollate

    with h5py.File(train_data, "r") as f:#h5py形式を読み込み
        #train.h5を読み込み
        train_x = np.asarray(f["x"])#note_rollをカットしたもの
        train_k = np.asarray(f["c"])#key情報0 or 1
        train_n = np.asarray(f["n"])#何番目のメロディがどこからどこまで鳴っているか
        train_m = np.asarray(f["m"])#どのメロディがどのコードの間隔で鳴っているのか
        train_y = np.asarray(f["y"])#カットされたコード情報
        train_z = np.asarray(f["z"])
        train_s = np.asarray(f["s"])
        
    with h5py.File(val_data, "r") as f:#上記同様
        val_x = np.asarray(f["x"])
        val_k = np.asarray(f["c"])
        val_n = np.asarray(f["n"])
        val_m = np.asarray(f["m"])
        val_y = np.asarray(f["y"])
        val_z = np.asarray(f["z"])
        val_s = np.asarray(f["s"])

    train_len = len(train_x)
    val_len = len(val_x)
    step_size = int(np.ceil(train_len / batch_size))
    #小数点切り上げ

    _time0 = time.time()#時間更新
    load_data_time = np.round(_time0 - start_time, 3)
    print("LOADED DATA")
    print("__time spent for loading data: {} sec".format(load_data_time))

    # LOAD DEVICE
    #cuda_condition = torch.cuda.is_available()
    cuda_condition = False
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")

    weight=[0.028783941754630837, 0.5734214459841475, 1.6909118690538023, 0.9639695595335382, 1.171537419091108, 0.2987013700540707, 1.2332383898299066, 1.7451486648536414, 1.4218736239391698, 1.341448574869369, 1.6501851781845316, 1.4351106165592398, 0.17688445063538535, 0.5631225524337473, 0.9361627453162246, 0.09389186807150847, 0.17666484430759813, 1.5125573055967783, 0.640976294090388, 1.6399446673270033, 1.4121050265227937, 1.6755956383558512, 1.7994723587012256, 1.2713797833298006, 0.1825217153176971, 0.4149523518943158, 1.2592631618412933, 0.12537157470314197, 0.2601037098910096, 1.7002367506846139, 0.0321362261304112, 1.0305613285486686, 1.7534195116065021, 0.6103126310606598, 1.1896190255593955, 0.16271066802224118, 1.476342844967965, 1.6605543848697124, 0.6923119703386452, 1.005357383013511, 1.4328873623120526, 1.7304561129512255, 0.03696240703228685, 0.2808772524665745, 1.7534195116065021, 0.6245299070711884, 0.7541204992844923, 1.3385366025650214, 0.6045286224656404, 1.683218912415705, 0.9609649790882389, 1.5730081502932483, 1.761769128328438, 1.1321037850335738, 0.31231767427736956, 0.7659865775341035, 1.1745127522189587, 0.0285049553862312, 0.13729089986231705, 1.4395778869609805, 0.22387239316771873, 1.2766442958901725, 1.7600928494242243, 1.6605543848697124, 1.6740792622125429, 0.8328940048378478, 0.8783749215312726, 1.0699002803613995, 0.649985096537196, 0.7367015470907446, 0.9203271565894824, 1.782136401488304]
    weight_chord=torch.tensor(weight)
    # LOAD MODEL
    MODEL = SurpriseNet_raw_meian.CVAE

    model = MODEL(model_type="SurpriseNet",device=device)
    model.to(device)
    trainer = optim.Adam(model.parameters(), lr=5e-4)
    #model.parameters()はモデルのパラメータのリスト
    #lrは学習率
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=trainer, lr_lambda=lambda epoch: 0.955 ** epoch)
    
    loss_function = torch.nn.NLLLoss(weight=weight_chord)
    
    #学習率を自動で変更する

    # model_path_ = "./trained/{}".format(exp_num)
    # checkpoint = torch.load(model_path_)
    # model.load_state_dict(checkpoint['state_dict'])
    # trainer.load_state_dict(checkpoint['optimizer'])

    _time1 = time.time()
    load_graph_time = np.round(_time1 - _time0, 3)
    print("LOADED GRAPH")
    print("__time spent for loading graph: {} sec".format(load_graph_time))
    print() 
    print("Start training...")
    print("** step size: {}".format(step_size))
    print()
    bar = 'until next 20th steps: '
    rest = '                    |' 
    shuf = 0

    # TRAIN
    start_train_time = time.time()
    prev_epoch_time = start_train_time
    model.train()#トレーニングしていることを伝える

    loss_list = list()
    val_loss_list = list()
    # loss_list = checkpoint['loss'].tolist()
    # val_loss_list = checkpoint['val_loss'].tolist()
    # load data loader
    train_dataset = CustomDataset(
        train_x, train_k, train_n, train_m, train_y, train_z,train_s,device=device)
        #別のプログラムで上記を扱えるようにする
    val_dataset = CustomDataset(
        val_x, val_k, val_n, val_m, val_y, val_z,val_s,device=device)

    generator = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, 
        collate_fn=PadCollate(), shuffle=True, drop_last=True, pin_memory=False)
    generator_val = DataLoader(
        val_dataset, batch_size=batch_size_val, num_workers=0, 
        collate_fn=PadCollate(), shuffle=False, pin_memory=False)

    loss_function = torch.nn.NLLLoss(weight=None)

    for epoch in range(checkpoint_num, total_epoch):#0から100まで

        epoch += 1
        model.train()
        
        for step, sample in enumerate(generator):#Dataloaderからひとつずつ取り出す
            
            # バッチのロード
            x, k, n, m, y, clab, modes,z,s= sample
            # x, k, n, m, y, clab = next(iter(generator))
            #x = x.long().to(device)
            melody_25=melody2histogram(x,batch_size)
            melody_25=melody_25.long().to(device)
            #k = k.float().to(device)
            #n = n.float().to(device)
            #m = m.float().to(device)
            y = y.long().to(device)
            #z = z.float().to(device)
            #clab = clab.float().to(device)
            #modes = modes.float().to(device)
            k_dim1=k.size()[0]
            zero_modes=torch.zeros(k_dim1)
            #zero_modes=zero_modes.long().to(device)
            #melody_histograms_batch=melody2histogram(x,batch_size)
            #melody_histograms_batch=melody_histograms_batch.float().to(device)
            ZZs=z2list(z,batch_size)
            ZZs.float().to(device)
            one_hot_melody_batch=melody2onehot(melody_25,batch_size)
            one_hot_melody_batch.float().to(device)
            one_hot_chords=chord2onehot(y,batch_size)
            one_hot_chords.float().to(device)
            length=np.array([8])
            length = torch.from_numpy(length.astype(np.float32)).clone()
            length.float().to(device)
            s = s.float().to(device)
            s = torch.permute(s,(0,2,1))
            #y_numpy=y.to('cpu').detach().numpy().copy()
            """
            y_dim3=y.view(2,256,1)
            surprisingness_seqs, TM = markov_chain(y_dim3, 73).create_surprisingness_seqs()
            print(surprisingness_seqs)
            """
            
            
            step += 1         

            ## GENERATOR ## 
            trainer.zero_grad()
            #勾配を初期化

            # forward
            #順伝搬
            pred, logp ,mu, log_var, _ = model(one_hot_chords, length, one_hot_melody_batch,s,ZZs)

            # Arrange 
            pred_flatten = []
            groundtruth_flatten = []
            logp_flatten = []
            y_flatten=[]
            length = length.squeeze()

            for i in range(batch_size):

                # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                logp_flatten.append(logp[i][:])

                # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,12 * 24 * 2)
                pred_flatten.append(pred[i][:])

                # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
                groundtruth_flatten.append(one_hot_chords[i][:])

                y_flatten.append(y[i][:])

            # Rearrange for loss calculation
            logp_flatten = torch.cat(logp_flatten, dim=0)
            pred_flatten = torch.cat(pred_flatten, dim=0)
            y_flatten = torch.cat(y_flatten, dim=0)

            # Loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(loss_function=loss_function, logp=logp, target=y, length=length, mean=mu, log_var=log_var, anneal_function='logistic', step=step, k=0.0025, x0=2500)
            loss = (NLL_loss + KL_weight * KL_loss)
            # compute loss
            # 損失計算 
            #mask = Mask()
            #loss = ST_loss_fn(chord,y)

            loss.backward()
            #勾配を計算

            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            #パラメーターのノルムを単一ベクトルにする
            #勾配が急上昇するのを防ぐため
            trainer.step()
            #更新したパラメータを呼び出す
            loss_list.append([loss.detach().item()])
            #計算から切り離された(backward不可能な)値を取得してlosslistに加える

            if step % 1 == 0:
                bar += '='
                rest = rest[1:]
                print(bar+'>'+rest, end='\r')

            if step % 20 == 0:
                # print losses 
                print()
                print("[{} --> epoch: {} / step: {}]\n".format("LSTM_4bar", epoch, step) + \
                "   --GENERATOR LOSS--\n" + \
                "           recon_chord: {:06.4f}\n".format(loss))
                print()
                bar = 'until next 20th steps: '
                rest = '                    |'

            gc.collect()
            
        _time2 = time.time()
        epoch_time = np.round(_time2 - prev_epoch_time, 3)
        print()
        print()
        print("------------------EPOCH {} finished------------------".format(epoch))
        print()
        print("==> time spent for this epoch: {} sec".format(epoch_time))
        print("==> loss: {:06.4f}".format(loss))  
        
        model.eval()#評価モードに変更

        Xv, Kv, Nv, Mv, Yv, CLABv, MODEv ,Zv, Sv= next(iter(generator_val))
        #評価用のデータセットから取得
        # x, k, n, m, y, clab = next(iter(generator))
        #x = x.long().to(device)
        melody_25v=melody2histogram(Xv,batch_size)
        melody_25v=melody_25v.long().to(device)
        #k = k.float().to(device)
        #n = n.float().to(device)
        #m = m.float().to(device)
        Yv = Yv.long().to(device)
        #z = z.float().to(device)
        #clab = clab.float().to(device)
        #modes = modes.float().to(device)
        k_dim1v=Kv.size()[0]
        zero_modesv=torch.zeros(k_dim1v)
        #zero_modes=zero_modes.long().to(device)
        #melody_histograms_batch=melody2histogram(x,batch_size)
        #melody_histograms_batch=melody_histograms_batch.float().to(device)
        ZZsv=z2list(Zv,batch_size)
        ZZsv.float().to(device)
        one_hot_melody_batchv=melody2onehot(melody_25v,batch_size)
        one_hot_melody_batchv.float().to(device)
        one_hot_chordsv=chord2onehot(Yv,batch_size)
        one_hot_chordsv.float().to(device)
        lengthv=np.array([8])
        lengthv = torch.from_numpy(lengthv.astype(np.float32)).clone()
        lengthv.float().to(device)
        Sv = Sv.float().to(device)
        Sv = torch.permute(Sv,(0,2,1))
        


        # forward
        #順伝搬を計算
        predv, logpv ,muv, log_varv, _v = model(one_hot_chordsv, lengthv, one_hot_melody_batchv,Sv,ZZsv)

        # generate
        #生成
        #chord_ = model.test(Xv, Nv, Mv)
        # Arrange 
        pred_flattenv = []
        groundtruth_flattenv = []
        logp_flattenv = []
        y_flattenv=[]
        lengthv = lengthv.squeeze()

        for i in range(batch_size):

            # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
            logp_flattenv.append(logpv[i][:])

            # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,12 * 24 * 2)
            pred_flattenv.append(predv[i][:])

            # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
            groundtruth_flattenv.append(one_hot_chordsv[i][:])

            y_flattenv.append(Yv[i][:])

        # Rearrange for loss calculation
        logp_flattenv = torch.cat(logp_flattenv, dim=0)
        pred_flattenv = torch.cat(pred_flattenv, dim=0)
        y_flattenv = torch.cat(y_flattenv, dim=0)

        # Loss calculation
        NLL_lossv, KL_lossv, KL_weightv = loss_fn(loss_function=loss_function, logp=logpv, target=Yv, length=lengthv, mean=muv, log_var=log_varv, anneal_function='logistic', step=step, k=0.0025, x0=2500)
        val_loss= (NLL_lossv + KL_weightv * KL_lossv)

        # compute loss 
        #損失計算
        #mask = Mask()
        #val_loss = ST_loss_fn(chord, Yv) 

        #損失を(backward不可能にして)リストに入れる
        val_loss_list.append([val_loss.detach().item()])

        # print losses
        print()
        print()
        print("==> [{} --> epoch {}] Validation:\n".format("LSTM_4bar", epoch) + \
        "   --GENERATOR LOSS--\n" + \
        "           recon_chord: {:06.4f}\n".format(val_loss))


        scheduler.step()

        # save checkpoint & loss
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': trainer.state_dict(),
                'loss_train': np.asarray(loss_list),
                'loss_val': np.asarray(val_loss_list)},
                os.path.join(model_path, "{}_{}".format("SurpriseNet_raw_meian", "HLSD")))

            _time3 = time.time()
            end_train_time = np.round(_time3 - start_train_time, 3)  
            print("__time spent for entire training: {} sec".format(end_train_time))

        prev_epoch_time = time.time()     
        shuf += 1


# LOSS FUNCTIONS
def ST_loss_fn(chord, y):
    #chordが推測値、yがコードのgroundTruth
    #mはどのメロディがどのコード間で鳴っているのか

    # VAE losses
    n, t = y.size(0), y.size(1)

    recon_chord = torch.mean(F.cross_entropy(
        chord.view(-1, 73), y.view(-1),
        reduction='none').view(n, t, 1))
    #meanは平均値を返す、cross_entropyは誤差
    #viewはリサイズ

    return recon_chord

def melody2histogram(x,batch_size):
    melody_histograms=[]
    for i in range(batch_size):
        melody_histogram_onesong=[]
        for j in range(8):
            melody_histogram=[0]*8
            for k in range(8):
                melody_12=x[i,j*8+k]
                if melody_12 == 88:
                    melody_histogram[k]=24
                else:
                    melody_index=melody_12%24
                    melody_histogram[k]=melody_index
            melody_histogram = np.array(melody_histogram)
            melody_histogram = melody_histogram.astype(np.float32)
            meldoy_histogram_tensor = torch.from_numpy(melody_histogram).clone()
            melody_histogram_onesong.append(meldoy_histogram_tensor)
        melody_histograms.append(torch.stack(melody_histogram_onesong))
    melody_histograms_batch=torch.stack(melody_histograms)
    return melody_histograms_batch

def z2list(z,batch_size):
    ZZs=[]
    for i in range(batch_size):
        Zs=[]
        for j in range(8):
            ZZ=np.array([z[i]])
            ZZ=torch.from_numpy(ZZ.astype(np.float32)).clone()
            Zs.append(ZZ)
        Zs=torch.stack(Zs, dim=0)
        ZZs.append(Zs)
    ZZs=torch.stack(ZZs, dim = 0)
    return ZZs

def melody2onehot(melody_25,batch_size):
    one_hot_melody_batch=[]
    for j in range(batch_size):
        one_hot_melodys=[]
        for i in range(8):
            one_hot_melody=F.one_hot(melody_25[0,i],num_classes=25)
            one_hot_melodys.append(one_hot_melody)
        one_hot_melodys=torch.stack(one_hot_melodys, dim = 0)
        #one_hot_melodys.float().to(device)
        one_hot_melody_batch.append(one_hot_melodys)
    one_hot_melody_batch=torch.stack(one_hot_melody_batch, dim = 0)
    return one_hot_melody_batch

def chord2onehot(y, batch_size):
    one_hot_chords=[]
    for i in range(batch_size):
        one_hot_chord=F.one_hot(y[i],num_classes=73)
        one_hot_chords.append(one_hot_chord)
    one_hot_chords=torch.stack(one_hot_chords, dim = 0)
    return one_hot_chords

class markov_chain(): 
    """
    Create surprisingness sequence from integer numpy array with shape of (batch, time, 1(index number)).
    Example: 

    Input: 
        num_chords = 5
        seq = np.random.randint(num_chords, size=(10,40,1))

    Output:
        surprisingness_seqs, TM = surprisingness.markov_chain(seq, num_chords).create_surprisingness_seqs()

    """
    def __init__(self, chord_seqs, chord_nums, all_chords=False):
        
        # Shape of chord_seqs(numpy array) : (batch, time, 1(index number))
        self.chord_seqs = chord_seqs
        self.states = [x for x in range(chord_nums)]
        self.num_state = chord_nums #number of states 
        self.M = [[0] * self.num_state for _ in range(self.num_state)]#73*73の0行列
  
    # Calculate transition_probability
    def transition_probability(self, seq):
        # Convert seq to index seq
        index_seq = np.squeeze(seq, axis=-1).tolist()

        for i,j in zip(index_seq, index_seq[1:]):
            self.M[i][j] += 1

        # Convert to probabilities:
        for row in self.M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
    
    # Create transition matrix from one chord sequence
    def create_transition_matrix(self, seq):
        self.transition_probability(seq)
        return np.array(self.M)
    
    # Calculate surprisingness
    def calculate_surprisingness(self, seq, t, TM):
        
        current = seq[t]
        i_ = self.states.index(current)

        previous = seq[t - 1]
        j_ = self.states.index(previous)

        if TM[i_][j_] == 0:
            surprisingness = -np.log(TM[i_][j_] + 1e-4)
        else:
            surprisingness = -np.log(TM[i_][j_])
            
        return surprisingness
    
    # Create surprisingness sequences
    def create_surprisingness_seqs(self):
    
        surprisingness_seqs = []
        batch = len(self.chord_seqs)
        
        # Calculate surprisingness for chord sequences 
        for i in tqdm(range(batch)):
            seq = self.chord_seqs[i]
            timesteps = range(1, len(seq))
            surprisingness_seq = [0]

            for step in timesteps:
                TM = self.create_transition_matrix(seq[:step]).transpose()
                surprisingness = self.calculate_surprisingness(seq, step, TM)
                surprisingness_seq.append(surprisingness)

                # Re-initiate a new transition matrix for next sequence
                self.M = [[0] * self.num_state for _ in range(self.num_state)]
                   
            surprisingness_seqs.append(np.asarray(surprisingness_seq))
        
        # Pad 0 to the positions if the length of the chord sequence is smaller than max length               
        for i in tqdm(range(len(surprisingness_seqs))):
            surprisingness_seqs[i] = np.pad(surprisingness_seqs[i], (0, 256 - surprisingness_seqs[i].shape[0]),'constant', constant_values = 0)
       
        # Convert all lists to np arrays
        surprisingness_seqs = np.asarray(surprisingness_seqs)
        surprisingness_seqs = np.expand_dims(surprisingness_seqs, axis=-1)

        print(surprisingness_seqs)

        surprisingness_seqs = np.reshape(surprisingness_seqs,(64,8,1))

        return surprisingness_seqs, TM  # surprisingness_seqs (batch, max_seq_length, 1), TM (num_state, num_state)
  


if __name__ == "__main__":
    '''
    python3 trainer.py
    '''
    main()