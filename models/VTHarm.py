import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from itertools import chain
import numpy as np
from scipy.stats import truncnorm
import math
from torch.autograd import Variable


# FUNCTION BLOCKS
class ConvBlock(nn.Module):#畳み込み層
    def __init__(self,
                 in_dim=None, 
                 out_dim=None, 
                 kernel=None, 
                 stride=1, 
                 padding=0, 
                 dilation=1,
                 batchnorm=True,
                 dropout=None,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvBlock, self).__init__()

        modules = [nn.Conv1d(in_dim, out_dim, kernel, stride, padding, dilation)]
        #1次元畳み込みの定義

        if batchnorm is True:
            modules.append(nn.BatchNorm1d(out_dim))
            #バッチ正規化
        if nonlinearity is not None:
            modules.append(nonlinearity)
            #LeakyReLUという活性化関数
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
            #Dropout学習中にニューロンを非アクティブ化する
        
        self.kernel_size = kernel
        self.layer = nn.Sequential(*modules)
        #以上のmodulesを用いて、モデル生成

    def forward(self, x):
        if self.kernel_size % 2 == 0:
            return self.layer(x)[:,:,:-1]
        else:
            return self.layer(x) 

class Mask(nn.Module):
    def __init__(self, m=None):
        super(Mask, self).__init__()   

    def forward(self, x, y):
        mask_expand = self.seq_mask(y).unsqueeze(-1)
        out = x * mask_expand  
        return out  

    def seq_mask(self, x):
        mask = torch.sign(torch.abs(torch.sum(x, dim=-1)))
        #sumでxの次元毎に合計を取り、絶対値をとり、sgn関数に通す
        return mask

    def get_subsequent_mask(self, seq):
        ''' 
        For masking out the subsequent info. 
        その後の情報をマスキングするため。
        Ref: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
        '''
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        #1×len_s×len_sの1行列の上三角行列を返す
        return subsequent_mask

    def attn_causal_mask(self, x, attn_heads=None):
        seq_mask = self.seq_mask(x)
        mask = seq_mask.unsqueeze(-2).bool() & self.get_subsequent_mask(seq_mask) # [n, t, t]
        mask = mask.unsqueeze(1).repeat(1, attn_heads, 1, 1)
        return mask

    def attn_noncausal_mask(self, x, attn_heads=None):
        mask = self.seq_mask(x).unsqueeze(1).unsqueeze(1).repeat(
            1, attn_heads, x.size(1), 1) # [b, h, q_t, k_t]
        return mask

    def attn_key_mask(self, x, y, attn_heads=None):
        mask = self.seq_mask(x).unsqueeze(1).unsqueeze(1).repeat(
            1, attn_heads, y.size(1), 1) # [b, h, q_t, k_t]
        return mask

class TruncatedNorm(nn.Module):#切断正規分布
    def __init__(self):
        super(TruncatedNorm, self).__init__()

    def forward(self, size, threshold=2.):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return values.astype('float32')

class Compress(nn.Module):
    def __init__(self):
        super(Compress, self).__init__()    

    def forward(self, x, m):
        out = torch.matmul(x.transpose(1, 2), m).transpose(1, 2)
        return out

    def mean(self, x, m):
        #メロディの埋め込みとどのメロディがどこからどこまで鳴っているか
        out = torch.matmul(x.transpose(1, 2), m).transpose(1, 2)
        #二つの行列積をまとめたもの
        m_ = torch.empty_like(m).copy_(m)
        #mと同じサイズの初期化されていないテンソルにmを
        #コピーした
        m_sum = torch.sum(m_, dim=1).unsqueeze(-1)
        #次元1に対しての総和を一列にまとめる
        m_sum = torch.where(m_sum==0, torch.ones_like(m_sum), m_sum)
        #m_sum=0の所は1でそれ以外はそのまま
        out = torch.div(out, m_sum)
        #out/m_sum
        return out
    
    def reverse(self, x, m):
        out = torch.matmul(
            x.transpose(1, 2), m.transpose(1, 2)).transpose(1, 2)
        return out

class ScaledPositionalEmbedding(nn.Module):
    '''
    https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/embedding/position.py#L6
    '''
    def __init__(self, d_model, max_len=1000, device=None):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.alpha * self.pe[:, :x.size(1)]


## MODULES ##
class FFN(nn.Module):#順伝搬型ネットワーク
    """
    Positionwise Feed-Forward Network
    https://github.com/soobinseo/Transformer-TTS/blob/7a1f23baa8cc703f63cc2f11405f6898e3217865/module.py#L114
    """
    
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.w_1 = ConvBlock(d_model, d_model * 4, 1, 1, 0, batchnorm=False)
        self.w_2 = nn.Conv1d(d_model * 4, d_model, 1, 1, 0)
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_):
        # FFN Network
        x = input_.transpose(1, 2) 
        x = self.w_2(self.w_1(x))
        x = x.transpose(1, 2) 

        # residual connection
        x = x + input_ 

        # dropout
        # x = self.dropout(x) 

        # layer normalization
        x = self.norm(x) 

        return x

class Attention(nn.Module):
    """
    https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/attention/single.py#L8

    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, h, d_model):
        super(Attention, self).__init__()
        self.h = h
        self.d_h = d_model // h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

    def forward(self, query, key, value, mask=None, dropout=None):
        batch_size = query.size(0)

        # do all the linear projections in batch from d_model => h x d_h
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_h).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # input size = (batch_size, h, T, d_model/h)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1)) # (batch_size, h, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        self.attn = p_attn

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):#アテンションを並列実行
    """
    Orig. Transformer: https://github.com/Kyubyong/transformer/blob/master/modules.py
    Transformer TTS: https://github.com/soobinseo/Transformer-TTS/blob/7a1f23baa8cc703f63cc2f11405f6898e3217865/module.py#L114
    BERT: https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/attention/multi_head.py#L5

    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, maxlen=256, dropout=0.1, device=None):
        super().__init__()
        assert d_model % h == 0
        
        self.d_model = d_model
        # self.attention = RelativeAttention(device=device)
        self.attention = Attention(h=h, d_model=d_model)
        self.output_linear = nn.Linear(d_model*2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Attention
        x, attn = self.attention(query, key, value, mask=mask, dropout=None) # VA

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        self.VA = x
        
        qx = torch.cat([query, x], dim=-1)
        result = self.output_linear(qx)
        
        # residual connections
        result = result + query
        result = self.norm(result)

        return result

class TransformerBlock(nn.Module):
    """
    https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/transformer.py#L7

    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, dropout=0.1, device=None):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, device=device)
        self.ffn = FFN(d_model=hidden)
        
    def forward(self, x, mask):
        x = self.attention(x, x, x, mask=mask)
        x = self.ffn(x)
        return x

class TransformerBlockED(nn.Module):
    """
    https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/transformer.py#L7

    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, dropout=0.1, device=None):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super(TransformerBlockED, self).__init__()
        self.slf_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, device=device)
        self.enc_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, device=device)
        self.ffn = FFN(d_model=hidden)

    def forward(self, x, y, slf_mask, enc_mask):
        y = self.slf_attention(y, y, y, mask=slf_mask)
        y = self.ffn(self.enc_attention(y, x, x, mask=enc_mask))
        return y


## MODULES ##
class MelodyEncoder(nn.Module):
    
    def __init__(self, m_dim, hidden, attn_heads, n_layers, device):
        super(MelodyEncoder, self).__init__()

        # layers
        self.h = attn_heads
        self.mask = Mask()
        self.comp = Compress()
        self.embedding = nn.Embedding(13, hidden//2)
        self.frame_pos = ScaledPositionalEmbedding(d_model=hidden//2, device=device)
        self.pos = ScaledPositionalEmbedding(d_model=hidden, device=device)
        self.pos_dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden//2, hidden)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, device=device) for _ in range(n_layers)])

    def forward(self, x, k, nm, cm):
        #メロディ、キー埋め込み、何番目のメロディがどこからどこまで鳴っているか
        #どのメロディがどのコードの間隔でなっているのかを入力すると
        # embeddings
        x_norm = torch.where(x==88, torch.ones_like(x)*12, (x + 21) % 12)
        #x=88の時は12でそれ以外は(x+21)/12の余り
        #メロディの埋め込み
        emb = self.embedding(x_norm) + self.frame_pos(x)
        #embeddingと位置エンコードを対数空間で一度計算する。
        # note-wise encodeノートの面からエンコード
        note = self.comp.mean(emb, nm) # roll2note
        #メロディと鳴っている個所を埋め込み
        note = self.linear(note)
        #全結合層
        note = torch.cat([k.unsqueeze(1), note], dim=1)#追加
        #noteにキー情報を追加
        note = note + self.pos(note)
        note = self.pos_dropout(note)
        #ノートの埋め込み
        cm_ = torch.cat([cm[:,:1], cm], dim=1)#追加
        key_mask = self.mask.attn_noncausal_mask(cm_, attn_heads=self.h)
        # self-attention
        for transformer in self.transformer_blocks:
            note = transformer.forward(note, mask=key_mask)
        attn = self.transformer_blocks[-1].attention.attention.attn

        return note, attn

class ContextEncoder(nn.Module):
    def __init__(self, z_dim, hidden, attn_heads, n_layers, device):
        super(ContextEncoder, self).__init__()

        # layers
        self.h = attn_heads
        self.mask = Mask()
        self.pos = ScaledPositionalEmbedding(d_model=hidden, device=device)
        self.pos_dropout = nn.Dropout(p=0.2)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, device=device) for _ in range(n_layers)])
        self.mu = nn.Linear(hidden*2, z_dim)
        self.logvar = nn.Linear(hidden*2, z_dim) 

    def forward(self, x, m, k, cm):
        x = torch.cat([k.unsqueeze(1), x], dim=1)
        out = x + self.pos(x)
        out = self.pos_dropout(out)
        cm_ = torch.cat([cm[:,:1], cm], dim=1)
        query_mask = self.mask.attn_noncausal_mask(cm_, attn_heads=self.h)
        # self-attention
        for transformer in self.transformer_blocks:
            out = transformer.forward(out, mask=query_mask)
        # inference
        mout = torch.mean(m, dim=1)
        cout = torch.mean(out, dim=1)
        out = torch.cat([mout, cout], dim=-1)
        mu = self.mu(out)
        logvar = self.logvar(out)#ここがVAE関係
        epsilon = torch.randn_like(logvar)
        c = mu + torch.exp(0.5 * logvar) * epsilon # [batch, z_dim]
        return [mu, logvar], c

class Generate(nn.Module):
    def __init__(self, z_dim, c_dim, hidden, attn_heads, n_layers, device):
        super(Generate, self).__init__()

        # layers
        self.z_dim = z_dim
        self.attn_heads = attn_heads
        self.device = device

        self.mask = Mask()
        self.pos = ScaledPositionalEmbedding(d_model=hidden, device=device)
        self.pos_dropout = nn.Dropout(p=0.2)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockED(hidden, attn_heads, device=device) for _ in range(n_layers)])
        self.decoder = nn.Linear(hidden, c_dim)

    def forward(self, key, query, key_m, query_m):

        # masks
        key_m = torch.cat([key_m[:,:1], key_m], dim=1)
        query_mask = self.mask.attn_causal_mask(query_m, attn_heads=1)
        key_mask = self.mask.attn_key_mask(key_m, query_m, attn_heads=1)
        
        # transformer blocks
        query = query + self.pos(query)
        query = self.pos_dropout(query)
        for transformer in self.transformer_blocks:
            query = transformer.forward(key, query,
                slf_mask=query_mask, enc_mask=key_mask)
        slf_attn = self.transformer_blocks[-1].slf_attention.attention.attn
        enc_attn = self.transformer_blocks[-1].enc_attention.attention.attn

        chord = self.decoder(query)

        return query, chord, slf_attn, enc_attn


class Harmonizer(nn.Module):#モデルの定義
    def __init__(self,
                 m_dim=89,
                 c_dim=73,
                 n_dim=12,
                 z_dim=16,
                 attn_heads=4,
                 hidden=None,
                 n_layers=None,
                 device=None):
        super(Harmonizer, self).__init__()#initを継承

        # attributes属性
        self.z_dim = z_dim
        self.device = device

        # layers層を定義
        self.melody_encoder = MelodyEncoder(m_dim=m_dim, hidden=hidden, 
            attn_heads=attn_heads, n_layers=n_layers, device=device)
        self.key_embedding = nn.Embedding(24, hidden)#辞書の次元数24,埋め込む次元=hidden
        self.chord_embedding = nn.Embedding(c_dim, hidden)#辞書の次元数コード数,埋め込む次元=hidden
        self.proj_c = nn.Linear(z_dim, hidden, bias=False)#一次変換を行う
        self.context_encoder = ContextEncoder(z_dim=z_dim, hidden=hidden, 
            attn_heads=attn_heads, n_layers=1, device=device)
        self.decoder = Generate(z_dim=z_dim, c_dim=c_dim, hidden=hidden, 
            attn_heads=attn_heads, n_layers=n_layers, device=device)

    def forward(self, x, kk, note_m, chord_m, chord, zero_modes):#フォワードパスを定義(順伝搬)
        # prepare dataデータ準備
        n = x.size(0)
        

        ## Encoder ##
        k_emb = self.key_embedding(zero_modes.long())#Major:0,Minor:12
        query = self.chord_embedding(chord)#コードの埋め込み
        key, key_attn = self.melody_encoder(x, k_emb, note_m, chord_m)
        #メロディ、キー埋め込み、何番目のメロディがどこからどこまで鳴っているか
        #どのメロディがどのコードの間隔でなっているのかを入力すると
        #ノートとアテンションの情報

        c_moments, c = self.context_encoder(query, key, k_emb, chord_m.transpose(1, 2))
        #コード、メロディ、キー、どのメロディがどのコードの間隔で鳴っているのか
        #を埋め込んだもの
        
        ## Decoder ##
        sos = k_emb + self.proj_c(c)#cの全結合とキー情報を足す
        query = torch.cat([sos.unsqueeze(1), query[:,:-1]], dim=1) 
        #sosとコードの埋め込みの結合
        query, est_chord, query_attn, kq_attn = \
            self.decoder(key, query, chord_m, chord_m.transpose(1, 2))
            #ノートの埋め込み、sosとコードの埋め込み、どのメロディがどのコード間隔で鳴っているか
            #とそれを整えたもの

        return c_moments, c, est_chord, kq_attn

    def test(self, x, kk, note_m, chord_m, k, c=None):
        
        ## Decoder ##
        n, t = chord_m.size(0), chord_m.size(2)
        #0次元と2次元のサイズ
        y_est = torch.LongTensor([72]).view(1, 1).repeat(n, 1).to(self.device)
        #torch.LongTensorにviewで大きさを合わせて、それをn×1行列に変換、
        # 指定されたデバイスを持つTensorに変換

        chord_list = list()#ここまでSTHarmと共通 

        if c is None:
            c = torch.randn(x.size(0), self.z_dim).to(self.device)
            #ランダムでデバイスを持つ、テンソルを作成する

        k_emb = self.key_embedding(k)
        #辞書の最大長24次元、hiddenの埋め込みベクトル
        key, key_attn = self.melody_encoder(x, k_emb, note_m, chord_m)
        #ノートとアテンションの埋め込み

        for i in range(t):
            query = self.chord_embedding(y_est)

            sos = k_emb + self.proj_c(c)
            query = torch.cat([sos.unsqueeze(1), query[:,1:]], dim=1) 
            _, est_chord, query_attn, kq_attn = \
                self.decoder(key, query, chord_m, chord_m[:,:,:i+1].transpose(1, 2))
            #ここからSTHarmと共通
            y_new = torch.argmax(torch.softmax(est_chord, dim=-1), dim=-1)
            #STHarmと共通
            y_est = torch.cat([y_est, y_new[:,-1].unsqueeze(1)], dim=1)
            #STHarmと共通
            chord_list.append(est_chord[:,-1])
            #STHarmと共通

        chord_list = torch.stack(chord_list, dim=1)

        return chord_list, kq_attn



