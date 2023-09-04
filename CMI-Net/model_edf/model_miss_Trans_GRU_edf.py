import torch.nn.functional as F
import torch
import torch.nn as nn
import copy
from copy import deepcopy
import math
import numpy as np
import torchsummary
from timm.models.layers import PatchEmbed, Mlp, DropPath
from transformer_encoder.encoder import TransformerEncoder
from torch.autograd import Variable
def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    #print(" d_k", d_k)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)


    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)

        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        #print("a",self.pe[:, : x.size(1)])
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result
class MultiHeadedAttention(nn.Module):
    def __init__(self, h_out, h_in, d_model_out, d_model_in, afr_reduced_cnn_size_out, afr_reduced_cnn_size_in, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model_out % h_out == 0
        assert d_model_in % h_in == 0
        self.d_k_out = d_model_out // h_out#16
        self.h_out = h_out#5
        self.d_k_in = d_model_in // h_in  # 16
        self.h_in = h_in  # 5
        self.convs_out_q = clones(
            CausalConv1d(afr_reduced_cnn_size_out, afr_reduced_cnn_size_out, kernel_size=7, stride=1), 1)
        self.convs_out = clones(CausalConv1d(afr_reduced_cnn_size_out, afr_reduced_cnn_size_out//2, kernel_size=7, stride=1), 4)
        self.convs_in = clones(CausalConv1d(afr_reduced_cnn_size_in, afr_reduced_cnn_size_in // 2, kernel_size=7, stride=1), 4)
        self.convs = clones(
            CausalConv1d(afr_reduced_cnn_size_in*2, afr_reduced_cnn_size_in, kernel_size=7, stride=1), 1)
        #self.linear_out = nn.Linear(d_model_out, d_model_out)
        #self.linear_in = nn.Linear(d_model_in, d_model_in)
        self.dropout = nn.Dropout(p=dropout)
        self.CE_out = nn.GRU(5120, 2560, num_layers=2, dropout=0.5, bidirectional=True)
        self.CE_in = nn.GRU(128, 64, num_layers=2, dropout=0.5, bidirectional=True)
        #self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout_1 = nn.Dropout(p=0.5)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, 0))
    def forward(self, query_in, key_in_o, value_in_o):
        "Implements Multi-head attention"
        nbatches_in = query_in.size(0)
        #B_in, T_in, C_in = query_in.size()

        B_in, T_in, C_in = query_in.size()
        #print("query_in.shape",query_in.shape)


        query_in = torch.cat([self.time_shift(query_in)[:, :T_in, :C_in // 2], query_in[:, :T_in, C_in // 2:]],
                             dim=2)  # 只需增加这句
        key_in_o = torch.cat([self.time_shift(key_in_o)[:, :T_in, :C_in // 2], key_in_o[:, :T_in, C_in // 2:]],
                           dim=2)  # 只需增加这句
        value_in_o = torch.cat([self.time_shift(value_in_o)[:, :T_in, :C_in // 2], value_in_o[:, :T_in, C_in // 2:]],
                             dim=2)  # 只需增加这句




        query_in = query_in.view(nbatches_in, -1, self.h_in, self.d_k_in).transpose(1, 2)

        key_in = self.convs_in[0](key_in_o)
        #print("key_in_o.shape", key_in_o.shape)
        value_in = self.convs_in[1](value_in_o)

        #融合

        x_in_copy_lstm,_ = self.CE_in(key_in_o.transpose(0, 1))
        #x_in_copy_lstm = x_in_copy_lstm.transpose(0, 1)
        x_in_copy_lstm = self.dropout_1(x_in_copy_lstm.transpose(0, 1))
        x_in_copy_lstm = self.gelu(x_in_copy_lstm)


        key_in_copy_lstm = x_in_copy_lstm
        key_in_copy = self.convs_in[2](key_in_copy_lstm)#通道降维
        #key_in_copy = self.gelu(key_in_copy)
        key_in_cat = torch.cat([key_in, key_in_copy], dim=1)
        key_in_cat = key_in_cat.view(nbatches_in, -1, self.h_in, self.d_k_in).transpose(1, 2)
        #print("key_in_cat", key_in_cat.shape)


        #value_in_copy_lstm,_ = self.CE_in(value_in_o.transpose(0, 1))
        value_in_copy_lstm = x_in_copy_lstm
        value_in_copy = self.convs_in[3](value_in_copy_lstm)
        value_in_cat = torch.cat([value_in, value_in_copy], dim=1)
        value_in_cat = value_in_cat.view(nbatches_in, -1, self.h_in, self.d_k_in).transpose(1, 2)

        x_in = attention(query_in, key_in_cat, value_in_cat, dropout=self.dropout)
        x_in = x_in.transpose(1, 2).contiguous() \
            .view(nbatches_in, -1, self.h_in * self.d_k_in)
        #print("1",x_in.shape)




        BT, C, D = x_in_copy_lstm.shape
        #print("1",x_in_copy_lstm.shape)

        query_out = x_in_copy_lstm.contiguous().view(-1, 25, C * D)#lstm到注意力
        query_out = self.convs_out_q[0](query_out)
        nbatches_out = query_out.size(0)
        B_out, T_out, C_out = query_out.size()
        query_out = query_out.view(nbatches_out, -1, self.h_out, self.d_k_out).transpose(1, 2)

        x_out = x_in.contiguous().view(-1, 25, C * D)
        x_out_copy_lstm,_ = self.CE_out(x_out.transpose(0, 1))
        x_out_copy_lstm = self.dropout_1(x_out_copy_lstm.transpose(0, 1))#注意力到LSTM
        #x_out_copy_lstm = x_out_copy_lstm.transpose(0, 1)  # 注意力到LSTM
        x_out_copy_lstm = self.gelu(x_out_copy_lstm)


        key_out = x_in_copy_lstm.contiguous().view(-1, 25, C * D)  # lstm到注意力
        key_out = self.convs_out[0](key_out)

        value_out = x_in_copy_lstm.contiguous().view(-1, 25, C * D)  # lstm到注意力
        value_out = self.convs_out[1](value_out)

        key_out_copy_lstm = x_out_copy_lstm
        key_out_copy = self.convs_out[2](key_out_copy_lstm)  # 通道降维
        key_out_cat = torch.cat([key_out, key_out_copy], dim=1)
        key_out_cat = key_out_cat.view(nbatches_out, -1, self.h_out, self.d_k_out).transpose(1, 2)
        #print(key_out_cat.shape)

        value_out_copy_lstm = x_out_copy_lstm
        value_out_copy = self.convs_out[3](value_out_copy_lstm)
        value_out_cat = torch.cat([value_out, value_out_copy], dim=1)
        value_out_cat = value_out_cat.view(nbatches_out, -1, self.h_out, self.d_k_out).transpose(1, 2)
        #print(value_out_cat.shape)
        x_out = attention(query_out, key_out_cat, value_out_cat, dropout=self.dropout)
        x_out = x_out.transpose(1, 2).contiguous() \
            .view(nbatches_out, -1, self.h_out * self.d_k_out)
        #print("a",x_out.shape)
        #print("b", x_out_copy_lstm.shape)
        x_out = x_out.contiguous().view(-1, C, D)  # lstm到注意力
        x_out_copy_lstm = x_out_copy_lstm.contiguous().view(-1, C, D)
        #print("a",x_out.shape)
        #print("b", x_out_copy_lstm.shape)
        #x_out = x_in.permute(0, 2, 1)
        #x_out_copy_lstm = x_out_copy_lstm.permute(0, 2, 1)
        x_cat = torch.cat([x_out, x_out_copy_lstm], dim=1)
        x_cat = self.convs[0](x_cat)
        #x_cat = self.gelu(x_cat)

        #print("a1", x_out.shape)
        #print("b1", x_out_copy_lstm.shape)
        #x_sum = torch.add(x_out, x_out_copy_lstm)
        #x_sum = torch.div(x_sum, 2)
        #print("a",x_sum.shape)
        #return self.linear_in(x_sum)
        return x_cat
class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size_out, size_in, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size_in)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x_in, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #print("s", x_out.shape)
        #print("s", x_in.shape)
        x_in = sublayer(self.norm(x_in))
        return x_in + self.dropout(x_in)
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size_in)

    def forward(self, x):
        #print(x_in.shape)
        #print(x_out.shape)
        for layer in self.layers:
            x = layer(x)
            #print(1)
        #x_in = x_in.permute(0, 2, 1)
        #B, T, D = x_in.size()
        #print(x_in.permute(0,2,1).shape)
        #print(x_out.contiguous().view(-1, T, D).shape)
        #x = torch.cat([x_out.contiguous().view(-1, T, D), x_in], dim=1)
        #print(x.shape)
        return self.norm(x)
class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''

    def __init__(self, size_out, size_in, self_attn, feed_forward,  afr_reduced_cnn_size_out, afr_reduced_cnn_size_in, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size_out, size_in, dropout), 2)
        self.size_out = size_out
        self.size_in = size_in
        #self.conv_out = CausalConv1d(afr_reduced_cnn_size_out, afr_reduced_cnn_size_out, kernel_size=7, stride=1, dilation=1)
        self.conv_in = CausalConv1d(afr_reduced_cnn_size_in, afr_reduced_cnn_size_in, kernel_size=7, stride=1, dilation=1)
        self.global_pool = nn.AdaptiveAvgPool1d(size_in)
        #self.PE_out = PositionalEncoding(d_model=size_out, dropout=0.1, max_len=20)
        #self.PE_in = PositionalEncoding(d_model=size_in, dropout=0.1, max_len=40)


    def forward(self, x):
        "Transformer Encoder"
        bt, c, d = x.shape
        #print(x.)
        #x_in = x.permute(0, 2, 1)
        x_in = x
        #print(x_out.shape)
        #print(x_in.shape)
        #x_out = self.PE_out(x_out)
        #x_in = self.PE_in(x_in)
        #print("a",x_out.shape)
        #print("b", x_in.shape)
        query_in = self.conv_in(x_in)
        x_in = self.sublayer_output[0](query_in, lambda x_in: self.self_attn(query_in, x_in, x_in))  # Encoder self-attention
        x_in = self.sublayer_output[1](x_in, self.feed_forward)
        #print(x.shape)

        #print("a1", x_out.shape)
        #print("b1", x_in.shape)

        #x_cat = torch.cat([x_out, x_in], dim=2)
        #x_cat = self.global_pool(x_cat)
        return x_in
class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model_out, d_model_in, d_ff_out, d_ff_in, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1_in = nn.Linear(d_model_in, d_ff_in)
        self.w_2_in = nn.Linear(d_ff_in, d_model_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in):
        "Implements FFN equation."
        return self.w_2_in(self.dropout(F.relu(self.w_1_in(x_in))))
class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
class MultiAutoencoder(nn.Module):
    def __init__(self):
        super(MultiAutoencoder, self).__init__()
        # Define Conv, SepConv
        conv = lambda in_f, out_f, kernel, s=None, pad=None: nn.Sequential(nn.Conv1d(in_f, out_f, (kernel,), stride=s, padding=pad), nn.BatchNorm1d(out_f), nn.GELU())
        sepconv_same = lambda in_f, out_f, kernel: nn.Sequential(nn.Conv1d(in_f, out_f, (kernel,), padding=int(kernel/2), groups=in_f),
                                                            nn.Conv1d(out_f, out_f, (1,)), nn.BatchNorm1d(out_f), nn.GELU())
        sepconvtrans_same = lambda in_f, out_f, kernel, s=None, pad=None: nn.Sequential(nn.ConvTranspose1d(in_f, out_f, (kernel,), stride=s, padding=pad, groups=out_f),
                                                                      nn.ConvTranspose1d(out_f, out_f, (1,)), nn.BatchNorm1d(out_f), nn.GELU())
        self.gelu = nn.GELU()
        # EEG Encoder
        self.conv_11 = conv(1, 32, 50, 6, 24)
        self.sepconv_12 = sepconv_same(32, 64, 25)
        self.maxpool_12 = nn.MaxPool1d(kernel_size=25, stride=5, padding=12)
        self.dropout_12 = nn.Dropout(0.5)
        self.sepconv_13 = sepconv_same(64, 128, 8)
        self.maxpool_13 = nn.MaxPool1d(kernel_size=8, stride=5, padding=3)
        self.dropout_13 = nn.Dropout(0.5)
        self.sepconv_14 = sepconv_same(128, 128, 3)
        self.maxpool_14 = nn.MaxPool1d(kernel_size=3, stride=4, padding=1)
        # self.dropout_13 = nn.Dropout(0.5)

        # EEG Decoder
        self.sepconv_trans_15 = sepconvtrans_same(128, 128, 3, 4, 0)
        self.sepconv_trans_16 = sepconvtrans_same(128, 64, 8, 5, 0)
        self.sepconv_trans_17 = sepconvtrans_same(64, 32, 25, 5, 5)
        self.trans_18 = nn.ConvTranspose1d(32, 1, 50, stride=6, padding=22)
        self.sigmoid_18 = nn.Sigmoid()
        ###########################
        # EOG Encoder
        self.conv_21 = conv(1, 32, 50, 6, 24)
        self.sepconv_22 = sepconv_same(32, 64, 25)
        self.maxpool_22 = nn.MaxPool1d(kernel_size=25, stride=5, padding=12)
        self.dropout_22 = nn.Dropout(0.5)
        self.sepconv_23 = sepconv_same(64, 128, 8)
        self.maxpool_23 = nn.MaxPool1d(kernel_size=8, stride=5, padding=3)
        self.dropout_23 = nn.Dropout(0.5)
        self.sepconv_24 = sepconv_same(128, 128, 3)
        self.maxpool_24 = nn.MaxPool1d(kernel_size=3, stride=4, padding=1)
        #self.dropout_13 = nn.Dropout(0.5)
        # EOG Decoder
        self.sepconv_trans_25 = sepconvtrans_same(128, 128, 3, 4, 0)
        self.sepconv_trans_26 = sepconvtrans_same(128, 64, 8, 5, 0)
        self.sepconv_trans_27 = sepconvtrans_same(64, 32, 25, 5, 5)
        self.trans_28 = nn.ConvTranspose1d(32, 1, 50, stride=6, padding=22)
        self.sigmoid_28 = nn.Sigmoid()
        #self.conv_6 = conv(32, 1, 6, 1, 3)
        ########################################


        self.fc1 = nn.Linear(128 * 5, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128 * 5)
        self.fc_bn2 = nn.BatchNorm1d(128 * 5)

    def reparameterize1(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.cuda.FloatTensor(std.size()).normal_()

        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def forward(self, x_EEG, x_EOG, y_miss_EEG, y_miss_EOG):
        #########################################
        x_EEG = self.conv_11(x_EEG)
        x_EEG = self.sepconv_12(x_EEG)
        x_EEG = self.maxpool_12(x_EEG)
        x_EEG = self.dropout_12(x_EEG)
        x_EEG = self.sepconv_13(x_EEG)
        x_EEG = self.maxpool_13(x_EEG)
        x_EEG = self.dropout_13(x_EEG)
        x_EEG = self.sepconv_14(x_EEG)
        x_EEG = self.maxpool_14(x_EEG)

        x_EOG = self.conv_21(x_EOG)
        x_EOG = self.sepconv_22(x_EOG)
        x_EOG = self.maxpool_22(x_EOG)
        x_EOG = self.dropout_22(x_EOG)
        x_EOG = self.sepconv_23(x_EOG)
        x_EOG = self.maxpool_23(x_EOG)
        x_EOG = self.dropout_23(x_EOG)
        x_EOG = self.sepconv_24(x_EOG)
        x_EOG = self.maxpool_24(x_EOG)

        #print(x_xyz.shape)
        x_EEG_logits = x_EEG
        x_EOG_logits = x_EOG
        y_miss_EEG = y_miss_EEG.unsqueeze(-1).unsqueeze(-1)
        y_miss_EOG = y_miss_EOG.unsqueeze(-1).unsqueeze(-1)


        y_miss_EEG_EOG = torch.add(y_miss_EEG, y_miss_EOG)

        x = torch.div(torch.add(torch.mul(x_EEG_logits, y_miss_EEG), torch.mul(x_EOG_logits, y_miss_EOG)), y_miss_EEG_EOG)
        #print("a",x.shape)

        ####计算特征概率分布
        x = x.contiguous().view(-1, 128 * 5)
        x = self.gelu(self.fc_bn1(self.fc1(x)))
        mu = self.fc11(x)
        logvar = self.fc12(x)
        z = self.reparameterize(mu, logvar)
        x = self.gelu(self.fc_bn2(self.fc2(z))).contiguous().view(-1, 128, 5)
        ##################
        x_EEG = self.sepconv_trans_15(x)
        x_EEG = self.sepconv_trans_16(x_EEG)
        x_EEG = self.sepconv_trans_17(x_EEG)
        x_EEG = self.trans_18(x_EEG)
        x_EEG = self.sigmoid_18(x_EEG)

        x_EOG = self.sepconv_trans_25(x)
        x_EOG = self.sepconv_trans_26(x_EOG)
        x_EOG = self.sepconv_trans_27(x_EOG)
        x_EOG = self.trans_28(x_EOG)
        x_EOG = self.sigmoid_28(x_EOG)

        return x_EEG, x_EOG, mu, logvar
class CNN_transformer_hr_xyz(nn.Module):
    def __init__(self):
        super().__init__()
        # 数据补全
        self.MAE = MultiAutoencoder()
        # 特征提取
        drate = 0.5
        #self.features_dim = 128
        #self.output_dim = 128
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.

        N = 1 # number of TCE clones
        d_model_out = 5120  # set to be 100 for SHHS dataset
        d_model_in = 128  # set to be 100 for SHHS dataset
        d_ff_out = 3072  # dimension of feed forward
        d_ff_in = 512
        h_out = 8  # number of attention heads
        h_in = 2  # number of attention heads
        dropout = 0.5#0.5
        afr_reduced_cnn_size_out = 25  # 通道数
        afr_reduced_cnn_size_in = 40  # 通道数
        attn = MultiHeadedAttention(h_out, h_in, d_model_out, d_model_in, afr_reduced_cnn_size_out, afr_reduced_cnn_size_in)
        ff= PositionwiseFeedForward(d_model_out, d_model_in, d_ff_out, d_ff_in, dropout)
        self.tce = TCE(EncoderLayer(d_model_out, d_model_in, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size_out, afr_reduced_cnn_size_in, dropout), N)

        self.features_EEG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.Conv1d(64, 64, kernel_size=25, stride=3, bias=False, padding=12),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),
        )
        self.features_EOG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.Conv1d(64, 64, kernel_size=25, stride=3, bias=False, padding=12),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),
            #nn.Dropout(0.1)
        )
        # 对比学习
        self.dropout_1 = nn.Dropout(0.5)

        self.features_dim = 3000
        self.output_dim = 128
        #self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.Projector_EEG = nn.Sequential(
            nn.Linear(self.features_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.Projector_EOG = nn.Sequential(
            nn.Linear(self.features_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        )

        self.device = torch.device('cuda')

        #self.CE = nn.LSTM(3840, 1920, num_layers=2, dropout=0.5, bidirectional=True)
    def forward(self, x_EEG, x_EOG, y_miss_EEG, y_miss_EOG):#
        b, t, c, d = x_EEG.shape
        x_EEG = x_EEG.contiguous().view(-1, c, d)
        x_EOG = x_EOG.contiguous().view(-1, c, d)


        x_EEG_g, x_EOG_g, mu, logvar = self.MAE(x_EEG, x_EOG, y_miss_EEG, y_miss_EOG)

        # 对比学习
        x_EEG_g_c = torch.flatten(x_EEG_g,1)
        x_EOG_g_c = torch.flatten(x_EOG_g,1)


        xx_EEG = self.Projector_EEG(x_EEG_g_c)
        xx_EEG = F.normalize(xx_EEG, dim=1)
        xx_EOG = self.Projector_EOG(x_EOG_g_c)
        xx_EOG = F.normalize(xx_EOG, dim=1)

        xx_EEG_EOG = torch.cat([xx_EEG, xx_EOG], dim=0)



        y_miss_EEG = y_miss_EEG.unsqueeze(-1).unsqueeze(-1)
        y_miss_EOG = y_miss_EOG.unsqueeze(-1).unsqueeze(-1)
        #print("y_miss_hr",y_miss_hr.shape)
        #print("y_miss_xyz", y_miss_xyz.shape)

        y_miss_EEG_f = torch.ones((b*t)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        y_miss_EEG_f = torch.sub(y_miss_EEG_f, y_miss_EEG)
        #print("y_miss_hr_f",y_miss_hr_f.shape)
        y_miss_EOG_f = torch.ones((b*t)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        y_miss_EOG_f = torch.sub(y_miss_EOG_f, y_miss_EOG)
        x_EEG = torch.add(torch.mul(x_EEG, y_miss_EEG), torch.mul(x_EEG_g, y_miss_EEG_f))
        x_EOG = torch.add(torch.mul(x_EOG, y_miss_EOG), torch.mul(x_EOG_g, y_miss_EOG_f))
        x_EEG = self.features_EEG(x_EEG)
        x_EOG = self.features_EOG(x_EOG)
        #print("a",x_EEG.shape)

        x = torch.cat([x_EEG, x_EOG], dim=2)
        #print(x_EEG.shape)
        #print(x_EOG.shape)
        bt, c, d = x.shape
        #x = x.contiguous().view(b,t,c*d)
        x = x.permute(0, 2, 1)
        print("x.shape",x.shape)
        x = self.tce(x)
        #x,_ = self.CE(x.transpose(0, 1))
        x = x.permute(0, 2, 1)
        x = self.dropout_1(x)
        #x = self.dropout_1(x.transpose(0, 1))
        x = x.contiguous().view(b, t, c, d)
        return x, x_EEG_g, x_EOG_g, mu, logvar, xx_EEG_EOG
class CMI_Net(nn.Module):
    def __init__(self):
        super().__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        #self.RELU = nn.ReLU()
        self.features2 = CNN_transformer_hr_xyz()
        #self.dropout = nn.Dropout(drate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        #self.fc_1 = nn.Linear(256, 256)
        #self.fc_2 = nn.Linear(256, 256)
        self.fc = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 5)
        #self.softmax = nn.Softmax(dim=1)
        self.RELU1 = nn.ReLU(True)
    def forward(self, x_EEG, x_EOG, y_miss_EEG, y_miss_EOG):

        x, x_EEG_g, x_EOG_g, mu, logvar, xx_EEG_EOG = self.features2(x_EEG, x_EOG, y_miss_EEG, y_miss_EOG)#局部 [2, 128, 64]
        b, t, c1, d1 = x.shape
        #print(x.shape)
        x = x.reshape(b*t,c1,d1)
        #x = x.contiguous().view(x.shape[0], -1)
        x = self.global_pool(x)
        x = torch.squeeze(x, -1)
        #print(feat_cat_sum.shape)
        x = self.fc(x)
        x = self.RELU1(x)
        #self.feat = x
        x = self.fc1(x)
        x = x.contiguous().view(b,t,5)
        x_EEG_g = x_EEG_g.contiguous().view(b, t, 1, 3000)
        #print(x_xyz_g.shape)
        x_EOG_g = x_EOG_g.contiguous().view(b, t, 1, 3000)
        mu = mu.contiguous().view(b, t, 256)
        logvar = logvar.contiguous().view(b, t, 256)
        xx_EEG_EOG = xx_EEG_EOG.contiguous().view(2*b, t, 128).unsqueeze(2)


        """
        x = self.global_pool(feat)
        x = x.squeeze(2)
        x = self.fc(x)
        """
        return x, x_EEG_g, x_EOG_g, mu, logvar, xx_EEG_EOG
if __name__ == '__main__':
    device = torch.device('cuda')
    model = CMI_Net().to(device)

    # model1 = MultiAutoencoder()
    x1 = torch.randn((2, 25, 1, 3000)).to(device)
    x2 = torch.randn((2, 25, 1, 3000)).to(device)
    y_miss_hr = np.array([0, 1, 1, 1, 0])
    y_miss_xyz = np.array([1, 0, 0, 1, 1])
    y_miss_hr = torch.from_numpy(y_miss_hr).to(device)
    y_miss_xyz = torch.from_numpy(y_miss_xyz).to(device)
    y_miss_hr = y_miss_hr.repeat(10).to(device)
    y_miss_xyz = y_miss_xyz.repeat(10).to(device)
    # print(y_miss_hr.shape)
    # out = model(x1,x2)
    out1, out2, out3, out4, out5, out6 = model(x1, x2, y_miss_hr, y_miss_xyz)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)
    print(out6.shape)
    """
    device = torch.device('cuda')
    model = MultiAutoencoder().to(device)
    x1 = torch.randn((50, 1, 3000)).to(device)
    x2 = torch.randn((50, 1, 3000)).to(device)
    x3 = torch.randn((50, 1, 3000)).to(device)

    y_miss_EEG = np.array([0, 1, 1, 1, 0])
    y_miss_EOG = np.array([1, 0, 0, 1, 1])
    y_miss_EMG = np.array([1, 0, 0, 1, 1])
    y_miss_EEG = torch.from_numpy(y_miss_EEG).to(device)
    y_miss_EOG = torch.from_numpy(y_miss_EOG).to(device)
    y_miss_EMG = torch.from_numpy(y_miss_EMG).to(device)
    y_miss_EEG = y_miss_EEG.repeat(10).to(device)
    y_miss_EOG = y_miss_EOG.repeat(10).to(device)
    y_miss_EMG = y_miss_EMG.repeat(10).to(device)

    out1, out2, out3, out4, out5 = model(x1, x2, x3, y_miss_EEG, y_miss_EOG, y_miss_EMG)

    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)
    
    

    state_dict = model.state_dict()
    """

    """
        for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
    device = torch.device('cpu')
    model.to(device)
    torchsummary.summary(model.cuda(), (1, 30))
    """