import numpy as np
import os
import sys 
sys.path.append("./utils")
sys.path.append("./models")
import time
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F 
import importlib
from scipy.stats import truncnorm

from CMD_parser_features import parse_CMD_features
from HLSD_features import parse_HLSD_features
from process_data import *
from utils.parse_utils import *
from models import STHarm, VTHarm ,LSTM ,SurpriseNet ,SurpriseNet_S
from metrics import *


sep = os.sep 

class TruncatedNorm(nn.Module):
    def __init__(self):
        super(TruncatedNorm, self).__init__()

    def forward(self, size, threshold):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        #ランダムな数を生成
        return values.astype('float32')

class TestData(object):#指定のデータを取ってくる
    def __init__(self, 
                 dataset=None,
                 song_ind=None,
                 start_point=None,
                 maxlen=16):
        super(TestData, self).__init__()

        self.test_parent_path = None 
        self.features = None 
        self.data = None
        self.test_name = None 
        sep = os.sep 

        # output 
        self.test_batches = None
        self.m2 = None 
        self.test_notes = None 
        self.key_sig = None


        self.test_parent_path = sep.join([".","LSTM","exp","test","raw"])
        #NPYファイルの保存先へのパス
        test_song_lists = sorted(glob(os.path.join(self.test_parent_path, "features.*.npy")))
        #NPYファイルをソート
        test_song = test_song_lists[song_ind]
        #NPYファイルからインデックス番目の曲を取り出す
        self.test_name = os.path.basename(test_song).split(".")[1].split("_")[0]
        #ファイル名を取得
        self.features = np.load(test_song, allow_pickle=True).tolist()
        #NPYファイルをロードしてリストに変換
        self.HLSD_data(self.features, start_point, maxlen)   
        #NPYファイルを入力
            

    def __call__(self):
        return self.test_batches, self.m2, self.test_notes

    def HLSD_data(self, features, start_point, maxlen):
        #指定されたデータの中からメロディ、コード等の情報を持ってくる

        # get onehot data
        inp, oup, key, onset, onset_xml, inds = get_roll_HLSD(features, chord_type="simple")
        #ここでメジャーもマイナーもCメジャーに変換されている

        # get indices where new chord
        new_chord_ind = list()
        for i, c in enumerate(onset[:,1]):
            if c == 1:
                new_chord_ind.append(i)
        new_chord_ind.append(len(inp))
                
        # get range
        chord_ind = new_chord_ind[start_point:start_point+maxlen+1]
        start, end = chord_ind[0], chord_ind[-1] # (maxlen+1)th chord

        note_inds = list()
        for i in inds:
            if i[1] >= start and i[1] < end: 
                note_inds.append(i[0])

        _x = inp[start:end]
        _k = np.asarray(12 * key)
        nnew_ = onset[start:end, :1]
        cnew_ = onset[start:end, -1:]
        nnew2_ = onset_xml[start:end, :1]
        cnew2_ = onset_xml[start:end, -1:]

        _n = make_align_matrix_roll2note(_x, nnew_) # roll2note 
        _m = make_align_matrix_note2chord(nnew_, cnew_) # note2chord
        _m2 = make_align_matrix_note2chord(nnew2_, cnew2_)
        _y = np.asarray([oup[c] for c in chord_ind[:-1]]) 

        x_pitch = np.argmax(_x, axis=-1)
        y_chord = np.argmax(_y, axis=-1) # labels
        self.key_sig = 0

        test_x = x_pitch
        test_k = _k
        test_m = _m
        test_n = _n
        test_y = y_chord

        self.test_batches = [test_x, test_k, test_m, test_n, test_y]
        self.m2 = _m2 #コード名のデータ
        self.test_notes = note_inds #オンセット情報とノートのインデックス


def test_model(dataset=None, 
               song_ind=None, 
               exp_name=None,
               device_num=None,
               lamb=None, 
               start_point=None,
               maxlen=8):

    model_name = "SurpriseNet_raw_meian"

    ## LOAD DATA ##
    test_data = TestData(dataset=dataset,
                         song_ind=song_ind,
                         start_point=start_point,
                         maxlen=maxlen)
    
    test_batch, _m2, test_notes = test_data()
    features = test_data.features
    key_sig = test_data.key_sig
    test_name = test_data.test_name

    ## LOAD MODEL ##
    module_name = model_name
    model = importlib.import_module(module_name)
    
    #Mask = model.Mask
    #Compress = model.Compress

    model_path = "./trained/{}_{}".format("SurpriseNet_raw_meian", "HLSD")
    cuda_condition = False
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")
    #device = torch.device("cuda:{}".format(device_num))
    #if torch.cuda.is_available():
        #torch.cuda.set_device(device_num) 
    Generator = model.CVAE("SurpriseNet",device)

    # select model
    model = Generator
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']) 

    ## SAMPLE ##
    start_time = time.time()
    model.eval()
    test_x, test_k, test_m, test_n, test_y = test_batch
    test_x_ = torch.from_numpy(test_x.astype(np.int64)).to(device).unsqueeze(0)
    test_k_ = torch.from_numpy(test_k.astype(np.int64)).to(device).unsqueeze(0)
    test_m_ = torch.from_numpy(test_m.astype(np.float32)).to(device).unsqueeze(0)
    test_n_ = torch.from_numpy(test_n.astype(np.float32)).to(device).unsqueeze(0)
    test_y_ = torch.from_numpy(test_y.astype(np.int64)).to(device).unsqueeze(0) 
    k_dim1=test_k_.size()[0]
    zero_modes=torch.zeros(k_dim1)
    zero_modes=zero_modes.long().to(device)
    melody_histogram=melody2histogram(test_x_,1)
    melody_histogram=melody_histogram.float().to(device)
    length=np.array([8])
    length = torch.from_numpy(length.astype(np.float32)).clone()
    length.float().to(device)
    melody_25=melody2histogram(test_x_,1)
    melody_25=melody_25.long().to(device)
    one_hot_melody_batch=melody2onehot(melody_25,1)
    one_hot_melody_batch.float().to(device)
    surprise=[9.6,9.3,7,1.8,0,5.7,9.3,9.6]
    surprise_condition=z2list(surprise,1)
    surprise_condition.float().to(device)
    lands=[1.5,1.4,1.1,0,-0.1,-1.2,-1.4,-1.5]
    lands_condition=z2list(lands,1)
    lands_condition.float().to(device)
    one_hot_chords=chord2onehot(test_y_,1)
    one_hot_chords.float().to(device)
    trunc = TruncatedNorm()
    z = torch.FloatTensor(trunc([8, 64], threshold=3.)).to(device)

    
    chord_, logp = model.test(length,one_hot_melody_batch,surprise_condition,z,lands_condition)

    #chord_, logp ,mu, log_var, _ = model(one_hot_chords, length, one_hot_melody_batch, surprise_condition)

    ## RESULT ##
    x_ = F.one_hot(test_x_, num_classes=89).cpu().data.numpy()[0] # onehot
    #tensorメロディをone-hot表現に変換
    k_ = test_k#numpykey情報0or1
    y_ = test_y#numpyカットされたコード情報
    m_ = test_m#numpyどのメロディがどのコードの間隔で鳴っているのか
    n_ = test_n#numpy何番目のメロディがどこからどこまで鳴っているか
    #mask_ = Mask().seq_mask(test_n_.transpose(1, 2))[0].cpu().detach().numpy()
    FI = FeatureIndex(dataset=dataset)

    ind2chord = FI.ind2chord_func_simple
    ind2root = FI.ind2root

    y_chord_ind = y_#カットされたコード情報
    y_croot_ind = y_chord_ind // 6#6で割って1の位で切り上げた数
    #コードのルート音
    print(y_croot_ind)
    y_ckind_ind = np.mod(y_chord_ind, 6)#各値について6で割った余り
    #コードの種類
    y_chord_lab = ["{}{}".format(ind2root[cr], ind2chord[ck]) \
        for cr, ck in zip(y_croot_ind, y_ckind_ind)]
        #コードのルート音とコードの種類をソート
    y_ckind_lab = ["{}".format(ind2chord[ck]) \
        for ck in y_ckind_ind]
        #コードの種類だけソート
    print(y_ckind_lab)

    test_chord = F.log_softmax(chord_[0], dim=-1).cpu().detach().numpy()
    test_chord_ind = np.argmax(test_chord, axis=-1)
    print(test_chord_ind)
    test_croot_ind = test_chord_ind // 6
    print(test_croot_ind)
    test_ckind_ind = np.mod(test_chord_ind, 6)
    test_chord_lab = ["{}{}".format(ind2root[cr], ind2chord[ck]) \
        for cr, ck in zip(test_croot_ind, test_ckind_ind)]
    test_ckind_lab = ["{}".format(ind2chord[ck]) \
        for ck in test_ckind_ind]
    #生成されたコードについて上記と同様の処理

    chord = np.argmax(test_chord, axis=-1)

    """
    # results for model
    result1, result2 = compute_metrics(x_, test_chord_ind, n_, m_, key_sig, dataset=dataset)
    # results for GT
    gt1, gt2 = compute_metrics(x_, y_chord_ind, n_, m_, key_sig, dataset=dataset)

    CHE, CC, CTR, PCS, CTD, MTD = result1  
    gCHE, gCC, gCTR, gPCS, gCTD, gMTD = gt1  
    TPS, DIC = result2 
    gTPS, gDIC = gt2 

    model_results = [np.asarray(TPS), np.asarray(DIC), test_chord_ind]
    gt_results = [np.asarray(gTPS), np.asarray(gDIC), y_chord_ind]

    TPSD, DICD, LD = chord_similarity(model_results, gt_results)

    print()
    print("{}:".format(exp_name))
    print("     > CHE: {:.4f} / CC: {:.4f} / CTnCTR: {:.4f}".format(CHE, CC, CTR))
    print("     > PCS: {:.4f} / CTD: {:.4f} / MCTD: {:.4f}".format(PCS, CTD, MTD))
    print("     > LD: {:.4f} / TPSD: {:.4f} / DICD: {:.4f}".format(LD, TPSD, DICD))

    
    """
    
    render_melody_chord_HLSD(y_croot_ind, y_ckind_lab, features, test_notes, _m2, 
        savepath="GT__{}__s{}_p{}-{}.mid".format(
            test_name, song_ind, start_point, start_point+maxlen-1))
    render_melody_chord_HLSD(test_croot_ind, test_ckind_lab, features, test_notes, _m2, 
        savepath="Sampled__{}__s{}_{}_p{}-{}.mid".format(
            test_name, song_ind, exp_name, start_point, start_point+maxlen-1))

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

def z2list(z,batch_size):
    ZZs=[]
    for i in range(batch_size):
        Zs=[]
        for j in range(8):
            ZZ=np.array([z[j]])
            ZZ=torch.from_numpy(ZZ.astype(np.float32)).clone()
            Zs.append(ZZ)
        Zs=torch.stack(Zs, dim=0)
        ZZs.append(Zs)
    ZZs=torch.stack(ZZs, dim = 0)
    return ZZs

def chord2onehot(y, batch_size):
    one_hot_chords=[]
    for i in range(batch_size):
        one_hot_chord=F.one_hot(y[i],num_classes=73)
        one_hot_chords.append(one_hot_chord)
    one_hot_chords=torch.stack(one_hot_chords, dim = 0)
    return one_hot_chords


if __name__ == "__main__":
    '''
    Ex) python3 test.py CMD 0 0 rVTHarm 3 (3)
    '''
    dataset = sys.argv[1]
    song_ind = int(sys.argv[2])
    start_point = int(sys.argv[3])
    exp_name = sys.argv[4]
    device_num = int(sys.argv[5])
    try:
        lamb = float(sys.argv[6])
    except IndexError:
        lamb = None

    test_model(
        dataset=dataset, song_ind=song_ind, start_point=start_point, 
        exp_name=exp_name, device_num=device_num, lamb=lamb, maxlen=8)