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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Manager

from models import STHarm, VTHarm
from models.STHarm import Mask, Compress
from process_data import FeatureIndex
from utils.parse_utils import *   
from data_loader import *


## TRAINER ##
def main(dataset=None,
         exp_name=None,
         checkpoint_num=0,
         device_num=0,
         batch_size=64, # 128
         batch_size_val=64, # 128 
         total_epoch=100,
         hidden=256,
         n_layers=4):

    start_time = time.time()
    
    # LOAD DATA
    datapath = 'C:\\Users\\barut\\harmonizers_transformer_self'
    # result_path = os.path.join(datapath, 'results')
    model_path = os.path.join(datapath, 'trained')
    train_data = os.path.join(datapath, '{}_train.h5'.format(dataset))
    val_data = os.path.join(datapath, '{}_val.h5'.format(dataset))
    FI = FeatureIndex(dataset=dataset)
    #process_data内のFI

    # LOAD DATA LOADER 
    if dataset == "CMD":
        CustomDataset = CMDDataset
        PadCollate = CMDPadCollate

    elif dataset == "HLSD":
        CustomDataset = HLSDDataset
        PadCollate = HLSDPadCollate

    with h5py.File(train_data, "r") as f:#h5py形式を読み込み
        #train.h5を読み込み
        train_x = np.asarray(f["x"])#note_rollをカットしたもの
        train_k = np.asarray(f["c"])#key情報0 or 1
        train_n = np.asarray(f["n"])#何番目のメロディがどこからどこまで鳴っているか
        train_m = np.asarray(f["m"])#どのメロディがどのコードの間隔で鳴っているのか
        train_y = np.asarray(f["y"])#カットされたコード情報
        
    with h5py.File(val_data, "r") as f:#上記同様
        val_x = np.asarray(f["x"])
        val_k = np.asarray(f["c"])
        val_n = np.asarray(f["n"])
        val_m = np.asarray(f["m"])
        val_y = np.asarray(f["y"])

    train_len = len(train_x)
    val_len = len(val_x)
    step_size = int(np.ceil(train_len / batch_size))
    #小数点切り上げ

    _time0 = time.time()#時間更新
    load_data_time = np.round(_time0 - start_time, 3)
    print("LOADED DATA")
    print("__time spent for loading data: {} sec".format(load_data_time))

    # LOAD DEVICE
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")

    # LOAD MODEL
    if exp_name == "STHarm":
        MODEL = STHarm.Harmonizer
    elif exp_name == "VTHarm" or exp_name == "rVTHarm":
        MODEL = VTHarm.Harmonizer

    model = MODEL(hidden=hidden, n_layers=n_layers, device=device)
    model.to(device)
    trainer = optim.Adam(model.parameters(), lr=1e-4)
    #model.parameters()はモデルのパラメータのリスト
    #lrは学習率
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=trainer, lr_lambda=lambda epoch: 0.95 ** epoch)
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
        train_x, train_k, train_n, train_m, train_y, device=device)
        #別のプログラムで上記を扱えるようにする
    val_dataset = CustomDataset(
        val_x, val_k, val_n, val_m, val_y, device=device)

    generator = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, 
        collate_fn=PadCollate(), shuffle=True, drop_last=True, pin_memory=False)
    generator_val = DataLoader(
        val_dataset, batch_size=batch_size_val, num_workers=0, 
        collate_fn=PadCollate(), shuffle=False, pin_memory=False)

    for epoch in range(checkpoint_num, total_epoch):#0から100まで

        epoch += 1
        model.train()
        
        for step, sample in enumerate(generator):#Dataloaderからひとつずつ取り出す

            # バッチのロード
            x, k, n, m, y, clab, modes= sample
            # x, k, n, m, y, clab = next(iter(generator))
            x = x.long().to(device)
            k = k.float().to(device)
            n = n.float().to(device)
            m = m.float().to(device)
            y = y.long().to(device)
            clab = clab.float().to(device)
            modes = modes.float().to(device)
            k_dim1=k.size()[0]
            zero_modes=torch.zeros(k_dim1)
            zero_modes=zero_modes.long().to(device)
            step += 1         

            ## GENERATOR ## 
            trainer.zero_grad()
            #勾配を初期化

            if exp_name == "STHarm":
                # forward
                #順伝搬
                chord, kq_attn = model(x, n, m, y) 

                # compute loss
                # 損失計算 
                mask = Mask()
                loss = ST_loss_fn(chord, y, m, mask) 

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
                    print("[{} --> epoch: {} / step: {}]\n".format(exp_name, epoch, step) + \
                    "   --GENERATOR LOSS--\n" + \
                    "           recon_chord: {:06.4f}\n".format(loss))
                    print()
                    bar = 'until next 20th steps: '
                    rest = '                    |'

            elif exp_name == "VTHarm":
                # forward
                c_moments, c, chord, kq_attn = model(x, k, n, m, y) 
                # compute loss 
                mask = Mask()
                loss, recon_chord, kld_c = VT_loss_fn(
                    c_moments, c, chord, y, m, clab, mask) 

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                trainer.step()
                loss_list.append(
                    [loss.detach().item(), \
                    recon_chord.detach().item(), kld_c.detach().item()])

                if step % 1 == 0:
                    bar += '='
                    rest = rest[1:]
                    print(bar+'>'+rest, end='\r')

                if step % 20 == 0:
                    # print losses 
                    print()
                    print("[{} --> epoch: {} / step: {}]\n".format(exp_name, epoch, step) + \
                    "   --GENERATOR LOSS--\n" + \
                    "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c))
                    print()
                    bar = 'until next 20th steps: '
                    rest = '                    |'

            elif exp_name == "rVTHarm":
                # forward
                c_moments, c, chord, kq_attn = model(x, k, n, m, y, zero_modes) 

                # compute loss 
                mask = Mask()
                loss, recon_chord, kld_c, reg_loss = rVT_loss_fn(
                    c_moments, c, chord, y, m, clab, mask,modes) 

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                trainer.step()
                loss_list.append(
                    [loss.detach().item(), \
                    recon_chord.detach().item(), kld_c.detach().item(), reg_loss.detach().item()])

                if step % 1 == 0:
                    bar += '='
                    rest = rest[1:]
                    print(bar+'>'+rest, end='\r')

                if step % 20 == 0:
                    # print losses 
                    print()
                    print("[{} --> epoch: {} / step: {}]\n".format(exp_name, epoch, step) + \
                    "   --GENERATOR LOSS--\n" + \
                    "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c) + \
                    "           reg_loss: {:06.4f}\n".format(-reg_loss))
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

        Xv, Kv, Nv, Mv, Yv, CLABv, MODEv= next(iter(generator_val))
        #評価用のデータセットから取得
        Xv = Xv.long().to(device)
        Kv = Kv.float().to(device)
        Nv = Nv.float().to(device)
        Mv = Mv.float().to(device)
        Yv = Yv.long().to(device)
        CLABv = CLABv.float().to(device)
        MODEv = MODEv.float().to(device)
        Kv_dim1=Kv.size()[0]
        Zero_MODEv=torch.zeros(Kv_dim1)
        Zero_MODEv=Zero_MODEv.long().to(device)

        if exp_name == "STHarm":
            # forward
            #順伝搬を計算
            chord, kq_attn = model(Xv, Nv, Mv, Yv) 

            # generate
            #生成
            chord_ = model.test(Xv, Nv, Mv)

            # compute loss 
            #損失計算
            mask = Mask()
            val_loss = ST_loss_fn(chord, Yv, Mv, mask) 

            #損失を(backward不可能にして)リストに入れる
            val_loss_list.append([val_loss.detach().item()])

            # print losses
            print()
            print()
            print("==> [{} --> epoch {}] Validation:\n".format(exp_name, epoch) + \
            "   --GENERATOR LOSS--\n" + \
            "           recon_chord: {:06.4f}\n".format(val_loss))

        elif exp_name == "VTHarm":
            # forward
            c_moments, c, chord, kq_attn = model(Xv, Kv, Nv, Mv, Yv)
            print(chord) 

            # generate
            chord_, kq_attn_ = model.test(Xv, Kv, Nv, Mv)
            print(chord_)

            # compute loss 
            mask = Mask()
            val_loss, recon_chord, kld_c = VT_loss_fn(
                c_moments, c, chord, Yv, Mv, CLABv, mask) 

            val_loss_list.append(
                [val_loss.detach().item(),
                recon_chord.detach().item(), kld_c.detach().item()])

            # print losses
            print()
            print()
            print("==> [{} --> epoch {}] Validation:\n".format(exp_name, epoch) + \
            "   --GENERATOR LOSS--\n" + \
            "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c))

        elif exp_name == "rVTHarm":
            # forward
            c_moments, c, chord, kq_attn = model(Xv, Kv, Nv, Mv, Yv, Zero_MODEv) 

            # generate
            chord_, kq_attn_ = model.test(Xv, Kv, Nv, Mv ,Zero_MODEv)

            # compute loss 
            mask = Mask()
            val_loss, recon_chord, kld_c, reg_loss = rVT_loss_fn(
                c_moments, c, chord, Yv, Mv, CLABv, mask, MODEv) 

            val_loss_list.append(
                [val_loss.detach().item(),
                recon_chord.detach().item(), kld_c.detach().item(), reg_loss.detach().item()])

            # print losses
            print()
            print()
            print("==> [{} --> epoch {}] Validation:\n".format(exp_name, epoch) + \
            "   --GENERATOR LOSS--\n" + \
            "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c) + \
            "           reg_loss: {:06.4f}\n".format(-reg_loss))
        
        print()
        bar = 'until next 20th steps: '
        rest = '                    |' 
        print()
        print("------------------EPOCH {} finished------------------".format(epoch))
        print()

        scheduler.step()

        # save checkpoint & loss
        if epoch % 4 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': trainer.state_dict(),
                'loss_train': np.asarray(loss_list),
                'loss_val': np.asarray(val_loss_list)},
                os.path.join(model_path, "{}_{}".format(exp_name, dataset)))

            _time3 = time.time()
            end_train_time = np.round(_time3 - start_train_time, 3)  
            print("__time spent for entire training: {} sec".format(end_train_time))

        prev_epoch_time = time.time()     
        shuf += 1


# LOSS FUNCTIONS
def ST_loss_fn(chord, y, m, mask):
    #chordが推測値、yがコードのgroundTruth
    #mはどのメロディがどのコード間で鳴っているのか

    # VAE losses
    n, t = y.size(0), y.size(1)

    recon_chord = torch.mean(mask(F.cross_entropy(
        chord.view(-1, 73), y.view(-1), 
        reduction='none').view(n, t, 1), m.transpose(1, 2)))
    #meanは平均値を返す、cross_entropyは誤差
    #viewはリサイズ

    return recon_chord

def VT_loss_fn(c_moments, c, chord, y, m, clab, mask, mode):

    # VAE losses
    n, t = y.size(0), y.size(1)

    recon_chord = -torch.mean(mask(F.cross_entropy(
        chord.view(-1, 73), y.view(-1), 
        reduction='none').view(n, t, 1), m.transpose(1, 2)))

    kld_c = torch.mean(kld(*c_moments))

    # VAE ELBO
    elbo = recon_chord - 0.1*kld_c

    # total loss
    total_loss = -elbo # negative to minimize

    return total_loss, recon_chord, kld_c

def rVT_loss_fn(c_moments, c, chord, y, m, clab, mask, mode):

    # VAE losses
    n, t = y.size(0), y.size(1)

    recon_chord = -torch.mean(mask(F.cross_entropy(
        chord.view(-1, 73), y.view(-1), 
        reduction='none').view(n, t, 1), m.transpose(1, 2)))

    kld_c = torch.mean(kld(*c_moments))

    mode=mode*12

    # regression lossここが固有コード数の制御関係
    M = c.size(0)#cの行数(64)
    ss = c[:,0]#0列目を切り取る
    ss_l =mode
    ss1 = ss.unsqueeze(0).expand(M, M)#0番目の次元のサイズが1,M×M行列
    ss2 = ss.unsqueeze(1).expand(M, M)#1番目の次元のサイズが1,M×M行列
    ss_l1 = ss_l.unsqueeze(0).expand(M, M)#0番目の次元のサイズが1,M×M行列
    ss_l2 = ss_l.unsqueeze(1).expand(M, M)#1番目の次元のサイズが1,M×M行列
    ss_D = ss1 - ss2
    ss_l_D = ss_l1 - ss_l2 
    reg_dist = (torch.tanh(ss_D) - torch.sign(ss_l_D))**2
    #tanhは-1,+1の範囲にする活性化関数、signはsgn関数x=0の時以外-1,1
    reg_loss = -torch.mean(reg_dist)
    #reg_distの平均値を返す

    # VAE ELBO
    elbo = recon_chord - 0.1*kld_c + reg_loss

    # total loss
    total_loss = -elbo # negative to minimize

    return total_loss, recon_chord, kld_c, reg_loss

def kld(mu, logvar, q_mu=None, q_logvar=None):
    '''
    KL(q(z2|x)||p(z2|u2)))(expectation along q(u2))
        --> b/c p(z2) depends on p(u2) (p(z2|u2))
        --> -0.5 * (1 + logvar - q_logvar 
            - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar)) 
    '''
    if q_mu is None:
        q_mu = torch.zeros_like(mu)
    if q_logvar is None:
        q_logvar = torch.zeros_like(logvar)

    return -0.5 * (1 + logvar - q_logvar - \
        (torch.pow(mu - q_mu, 2) + torch.exp(logvar)) / torch.exp(q_logvar))





if __name__ == "__main__":
    '''
    python3 trainer.py [dataset] [exp_name]

    - dataset: CMD / HLSD 
    - exp_name: STHarm / VTHarm / rVTHarm
    '''
    dataset = sys.argv[1]
    exp_name = sys.argv[2]
    main(dataset=dataset, exp_name=exp_name)



