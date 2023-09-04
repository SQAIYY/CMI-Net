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
        self.CE_out = nn.GRU(3072, 1536, num_layers=2, dropout=0.5, bidirectional=True)
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


        query_in = torch.cat([self.time_shift(query_in)[:, :T_in, :C_in // 2], query_in[:, :T_in, C_in // 2:]],
                             dim=2)  # 只需增加这句
        key_in_o = torch.cat([self.time_shift(key_in_o)[:, :T_in, :C_in // 2], key_in_o[:, :T_in, C_in // 2:]],
                           dim=2)  # 只需增加这句
        value_in_o = torch.cat([self.time_shift(value_in_o)[:, :T_in, :C_in // 2], value_in_o[:, :T_in, C_in // 2:]],
                             dim=2)  # 只需增加这句




        query_in = query_in.view(nbatches_in, -1, self.h_in, self.d_k_in).transpose(1, 2)

        key_in = self.convs_in[0](key_in_o)
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

        query_out = x_in_copy_lstm.contiguous().view(-1, 20, C * D)#lstm到注意力
        query_out = self.convs_out_q[0](query_out)
        nbatches_out = query_out.size(0)
        B_out, T_out, C_out = query_out.size()
        query_out = query_out.view(nbatches_out, -1, self.h_out, self.d_k_out).transpose(1, 2)

        x_out = x_in.contiguous().view(-1, 20, C * D)
        x_out_copy_lstm,_ = self.CE_out(x_out.transpose(0, 1))
        x_out_copy_lstm = self.dropout_1(x_out_copy_lstm.transpose(0, 1))#注意力到LSTM
        #x_out_copy_lstm = x_out_copy_lstm.transpose(0, 1)  # 注意力到LSTM
        x_out_copy_lstm = self.gelu(x_out_copy_lstm)


        key_out = x_in_copy_lstm.contiguous().view(-1, 20, C * D)  # lstm到注意力
        key_out = self.convs_out[0](key_out)

        value_out = x_in_copy_lstm.contiguous().view(-1, 20, C * D)  # lstm到注意力
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
        #print(x_out.shape)
        x_out = x_out.contiguous().view(-1, C, D)  # lstm到注意力
        x_out_copy_lstm = x_out_copy_lstm.contiguous().view(-1, C, D)
        #print(x_out.shape)
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
        #######################
        # acc Encoder
        self.conv_11 = conv(3, 32, 9, 3, 4)
        self.sepconv_12 = sepconv_same(32, 64, 7)
        self.maxpool_12 = nn.MaxPool1d(kernel_size=7, stride=2, padding=1)
        self.dropout_12 = nn.Dropout(0.5)
        self.sepconv_13 = sepconv_same(64, 128, 3)
        self.maxpool_13 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout_13 = nn.Dropout(0.5)
        self.sepconv_14 = sepconv_same(128, 128, 3)
        self.maxpool_14 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.dropout_13 = nn.Dropout(0.5)

        # gyro Encoder
        self.conv_21 = conv(3, 32, 9, 3, 4)
        self.sepconv_22 = sepconv_same(32, 64, 7)
        self.maxpool_22 = nn.MaxPool1d(kernel_size=7, stride=2, padding=1)
        self.dropout_22 = nn.Dropout(0.5)
        self.sepconv_23 = sepconv_same(64, 128, 3)
        self.maxpool_23 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout_23 = nn.Dropout(0.5)
        self.sepconv_24 = sepconv_same(128, 128, 3)
        self.maxpool_24 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # tot_acc Encoder
        self.conv_31 = conv(3, 32, 9, 3, 4)
        self.sepconv_32 = sepconv_same(32, 64, 7)
        self.maxpool_32 = nn.MaxPool1d(kernel_size=7, stride=2, padding=1)
        self.dropout_32 = nn.Dropout(0.5)
        self.sepconv_33 = sepconv_same(64, 128, 3)
        self.maxpool_33 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout_33 = nn.Dropout(0.5)
        self.sepconv_34 = sepconv_same(128, 128, 3)
        self.maxpool_34 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # acc Decoder
        self.sepconv_trans_15 = sepconvtrans_same(128, 128, 3, 2, 0)
        self.sepconv_trans_16 = sepconvtrans_same(128, 64, 3, 2, 1)
        self.sepconv_trans_17 = sepconvtrans_same(64, 32, 7, 2, 2)
        self.trans_18 = nn.ConvTranspose1d(32, 3, 9, stride=3, padding=4,output_padding=1)
        self.sigmoid_18 = nn.Sigmoid()

        # gyro Decoder
        self.sepconv_trans_25 = sepconvtrans_same(128, 128, 3, 2, 0)
        self.sepconv_trans_26 = sepconvtrans_same(128, 64, 3, 2, 1)
        self.sepconv_trans_27 = sepconvtrans_same(64, 32, 7, 2, 2)
        self.trans_28 = nn.ConvTranspose1d(32, 3, 9, stride=3, padding=4, output_padding=1)
        self.sigmoid_28 = nn.Sigmoid()

        # tot_acc Decoder
        self.sepconv_trans_35 = sepconvtrans_same(128, 128, 3, 2, 0)
        self.sepconv_trans_36 = sepconvtrans_same(128, 64, 3, 2, 1)
        self.sepconv_trans_37 = sepconvtrans_same(64, 32, 7, 2, 2)
        self.trans_38 = nn.ConvTranspose1d(32, 3, 9, stride=3, padding=4, output_padding=1)
        self.sigmoid_38 = nn.Sigmoid()

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
    def forward(self, x_acc, x_gyro, x_tot_acc, y_miss_acc, y_miss_gyro, y_miss_tot_acc):
        #########################################
        #b, t, c, d = x_acc.shape
        #x_acc = x_acc.contiguous().view(-1, c, d)
        x_acc = self.conv_11(x_acc)
        x_acc = self.sepconv_12(x_acc)
        x_acc = self.maxpool_12(x_acc)
        x_acc = self.dropout_12(x_acc)
        x_acc = self.sepconv_13(x_acc)
        x_acc = self.maxpool_13(x_acc)
        x_acc = self.dropout_13(x_acc)
        x_acc = self.sepconv_14(x_acc)
        x_acc = self.maxpool_14(x_acc)

        #x_gyro = x_gyro.contiguous().view(-1, c, d)
        x_gyro = self.conv_21(x_gyro)
        x_gyro = self.sepconv_22(x_gyro)
        x_gyro = self.maxpool_22(x_gyro)
        x_gyro = self.dropout_22(x_gyro)
        x_gyro = self.sepconv_23(x_gyro)
        x_gyro = self.maxpool_23(x_gyro)
        x_gyro = self.dropout_23(x_gyro)
        x_gyro = self.sepconv_24(x_gyro)
        x_gyro = self.maxpool_24(x_gyro)

        #x_tot_acc = x_tot_acc.contiguous().view(-1, c, d)
        x_tot_acc = self.conv_31(x_tot_acc)
        x_tot_acc = self.sepconv_32(x_tot_acc)
        x_tot_acc = self.maxpool_32(x_tot_acc)
        x_tot_acc = self.dropout_32(x_tot_acc)
        x_tot_acc = self.sepconv_33(x_tot_acc)
        x_tot_acc = self.maxpool_33(x_tot_acc)
        x_tot_acc = self.dropout_33(x_tot_acc)
        x_tot_acc = self.sepconv_34(x_tot_acc)
        x_tot_acc = self.maxpool_34(x_tot_acc)

        x_acc_logits = x_acc
        x_gyro_logits = x_gyro
        x_tot_acc_logits = x_tot_acc

        y_miss_acc = y_miss_acc.unsqueeze(-1).unsqueeze(-1)
        y_miss_gyro = y_miss_gyro.unsqueeze(-1).unsqueeze(-1)
        y_miss_tot_acc = y_miss_tot_acc.unsqueeze(-1).unsqueeze(-1)

        y_miss_agt = torch.add(y_miss_acc, y_miss_gyro)
        y_miss_agt = torch.add(y_miss_agt, y_miss_tot_acc)

        x = torch.add(torch.mul(x_acc_logits, y_miss_acc), torch.mul(x_gyro_logits, y_miss_gyro))
        x = torch.div(torch.add(x, torch.mul(x_tot_acc_logits, y_miss_tot_acc)), y_miss_agt)
        ####计算特征概率分布
        x = x.contiguous().view(-1, 128 * 5)
        x = self.gelu(self.fc_bn1(self.fc1(x)))
        mu = self.fc11(x)
        logvar = self.fc12(x)
        z = self.reparameterize(mu, logvar)
        x = self.gelu(self.fc_bn2(self.fc2(z))).contiguous().view(-1, 128, 5)

        x_acc = self.sepconv_trans_15(x)
        x_acc = self.sepconv_trans_16(x_acc)
        x_acc = self.sepconv_trans_17(x_acc)
        x_acc = self.trans_18(x_acc)
        x_acc = self.sigmoid_18(x_acc)

        x_gyro = self.sepconv_trans_25(x)
        x_gyro = self.sepconv_trans_26(x_gyro)
        x_gyro = self.sepconv_trans_27(x_gyro)
        x_gyro = self.trans_28(x_gyro)
        x_gyro = self.sigmoid_28(x_gyro)

        x_tot_acc = self.sepconv_trans_35(x)
        x_tot_acc = self.sepconv_trans_36(x_tot_acc)
        x_tot_acc = self.sepconv_trans_37(x_tot_acc)
        x_tot_acc = self.trans_38(x_tot_acc)
        x_tot_acc = self.sigmoid_38(x_tot_acc)
        return x_acc, x_gyro, x_tot_acc, mu, logvar
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
        d_model_out = 3072  # set to be 100 for SHHS dataset
        d_model_in = 128  # set to be 100 for SHHS dataset
        d_ff_out = 3072  # dimension of feed forward
        d_ff_in = 512
        h_out = 8  # number of attention heads
        h_in = 2  # number of attention heads
        dropout = 0.5#0.5
        afr_reduced_cnn_size_out = 20  # 通道数
        afr_reduced_cnn_size_in = 24  # 通道数
        attn = MultiHeadedAttention(h_out, h_in, d_model_out, d_model_in, afr_reduced_cnn_size_out, afr_reduced_cnn_size_in)
        ff= PositionwiseFeedForward(d_model_out, d_model_in, d_ff_out, d_ff_in, dropout)
        self.tce = TCE(EncoderLayer(d_model_out, d_model_in, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size_out, afr_reduced_cnn_size_in, dropout), N)

        self.features_acc = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=9, stride=3, bias=False, padding=4),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.Conv1d(64, 64, kernel_size=7, stride=3, bias=False, padding=4),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(drate),
        )

        self.features_gyro = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=9, stride=3, bias=False, padding=4),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.Conv1d(64, 64, kernel_size=7, stride=3, bias=False, padding=4),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(drate),
        )
        self.features_tot_acc = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=9, stride=3, bias=False, padding=4),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.Conv1d(64, 64, kernel_size=7, stride=3, bias=False, padding=4),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(drate),
        )
        # 对比学习
        self.dropout_1 = nn.Dropout(0.5)

        self.features_dim = 384
        self.output_dim = 128
        #self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.Projector_acc = nn.Sequential(
            nn.Linear(self.features_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.Projector_gyro = nn.Sequential(
            nn.Linear(self.features_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.Projector_tot_acc = nn.Sequential(
            nn.Linear(self.features_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        )

        self.device = torch.device('cuda')

        #self.CE = nn.LSTM(3840, 1920, num_layers=2, dropout=0.5, bidirectional=True)
    def forward(self, x_acc, x_gyro, x_tot_acc, y_miss_acc, y_miss_gyro, y_miss_tot_acc):#
        b, t, c, d = x_acc.shape
        x_acc = x_acc.contiguous().view(-1, c, d)
        x_gyro = x_gyro.contiguous().view(-1, c, d)
        x_tot_acc = x_tot_acc.contiguous().view(-1, c, d)


        x_acc_g, x_gyro_g, x_tot_acc_g, mu, logvar = self.MAE(x_acc, x_gyro, x_tot_acc, y_miss_acc, y_miss_gyro, y_miss_tot_acc)

        # 对比学习
        x_acc_g_c = torch.flatten(x_acc_g,1)
        x_gyro_g_c = torch.flatten(x_gyro_g,1)
        x_tot_acc_g_c = torch.flatten(x_tot_acc_g, 1)


        xx_acc = self.Projector_acc(x_acc_g_c)
        xx_acc = F.normalize(xx_acc, dim=1)
        xx_gyro = self.Projector_gyro(x_gyro_g_c)
        xx_gyro = F.normalize(xx_gyro, dim=1)
        xx_tot_acc = self.Projector_tot_acc(x_tot_acc_g_c)
        xx_tot_acc = F.normalize(xx_tot_acc, dim=1)

        xx_agt = torch.cat([xx_acc, xx_gyro, xx_tot_acc], dim=0)
        #xx_agt = torch.cat([xx_agt, xx_tot_acc], dim=0)

        y_miss_acc = y_miss_acc.unsqueeze(-1).unsqueeze(-1)
        y_miss_gyro = y_miss_gyro.unsqueeze(-1).unsqueeze(-1)
        y_miss_tot_acc = y_miss_tot_acc.unsqueeze(-1).unsqueeze(-1)
        #print("y_miss_hr",y_miss_hr.shape)
        #print("y_miss_xyz", y_miss_xyz.shape)

        y_miss_acc_f = torch.ones((b*t)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        y_miss_acc_f = torch.sub(y_miss_acc_f, y_miss_acc)
        #print("y_miss_hr_f",y_miss_hr_f.shape)
        y_miss_gyro_f = torch.ones((b*t)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        y_miss_gyro_f = torch.sub(y_miss_gyro_f, y_miss_gyro)
        y_miss_tot_acc_f = torch.ones((b * t)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        y_miss_tot_acc_f = torch.sub(y_miss_tot_acc_f, y_miss_tot_acc)

        x_acc = torch.add(torch.mul(x_acc, y_miss_acc), torch.mul(x_acc_g, y_miss_acc_f))
        x_gyro = torch.add(torch.mul(x_gyro, y_miss_gyro), torch.mul(x_gyro_g, y_miss_gyro_f))
        x_tot_acc = torch.add(torch.mul(x_tot_acc, y_miss_tot_acc), torch.mul(x_tot_acc_g, y_miss_tot_acc_f))
        x_acc = self.features_acc(x_acc)

        x_gyro = self.features_gyro(x_gyro)
        x_tot_acc = self.features_tot_acc(x_tot_acc)
        #print("a",x_EEG.shape)

        x = torch.cat([x_acc, x_gyro, x_tot_acc], dim=2)
        #print(x.shape)
        #print(x.shape)
        #x = torch.cat([x, x_tot_acc], dim=2)
        bt, c, d = x.shape
        #x = x.contiguous().view(b,t,c*d)
        x = x.permute(0, 2, 1)
        x = self.tce(x)
        #x,_ = self.CE(x.transpose(0, 1))
        x = x.permute(0, 2, 1)
        x = self.dropout_1(x)
        #x = self.dropout_1(x.transpose(0, 1))
        x = x.contiguous().view(b, t, c, d)
        return x, x_acc_g, x_gyro_g, x_tot_acc_g, mu, logvar, xx_agt
class CMI_Net(nn.Module):
    def __init__(self):
        super().__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        #self.RELU = nn.ReLU()
        self.features = CNN_transformer_hr_xyz()
        #self.dropout = nn.Dropout(drate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        #self.fc_1 = nn.Linear(256, 256)
        #self.fc_2 = nn.Linear(256, 256)
        self.fc = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 6)
        #self.softmax = nn.Softmax(dim=1)
        self.RELU1 = nn.ReLU(True)
    def forward(self, x_acc, x_gyro, x_tot_acc, y_miss_acc, y_miss_gyro, y_miss_tot_acc):

        x, x_acc_g, x_gyro_g, x_tot_acc_g, mu, logvar, xx_agt = self.features(x_acc, x_gyro, x_tot_acc, y_miss_acc, y_miss_gyro, y_miss_tot_acc)#局部 [2, 128, 64]
        b, t, c1, d1 = x.shape
        #print(x.shape)
        x = x.reshape(b*t, c1, d1)
        #x = x.contiguous().view(x.shape[0], -1)
        x = self.global_pool(x)
        x = torch.squeeze(x, -1)
        #print(feat_cat_sum.shape)
        x = self.fc(x)
        x = self.RELU1(x)
        #self.feat = x
        x = self.fc1(x)
        x = x.contiguous().view(b,t,6)
        x_acc_g = x_acc_g.contiguous().view(b, t, 3, 128)
        #print(x_xyz_g.shape)
        x_gyro_g = x_gyro_g.contiguous().view(b, t, 3, 128)
        x_tot_acc_g = x_tot_acc_g.contiguous().view(b, t, 3, 128)
        mu = mu.contiguous().view(b, t, 256)
        logvar = logvar.contiguous().view(b, t, 256)
        xx_agt = xx_agt.contiguous().view(3*b, t, 128).unsqueeze(2)


        """
        x = self.global_pool(feat)
        x = x.squeeze(2)
        x = self.fc(x)
        """
        return x, x_acc_g, x_gyro_g, x_tot_acc_g, mu, logvar, xx_agt
if __name__ == '__main__':
    device = torch.device('cuda')
    model = CMI_Net().to(device)

    # model1 = MultiAutoencoder()
    x1 = torch.randn((2, 20, 3, 128)).to(device)
    x2 = torch.randn((2, 20, 3, 128)).to(device)
    x3 = torch.randn((2, 20, 3, 128)).to(device)
    y1 = np.array([0, 1, 1, 1])
    y2 = np.array([1, 0, 0, 1])
    y3 = np.array([1, 0, 0, 1])
    y1 = torch.from_numpy(y1).to(device)
    y2 = torch.from_numpy(y2).to(device)
    y3 = torch.from_numpy(y3).to(device)
    y1 = y1.repeat(10).to(device)
    y2 = y2.repeat(10).to(device)
    y3 = y3.repeat(10).to(device)
    x, x_acc_g, x_gyro_g, x_tot_acc_g, mu, logvar, xx_agt = model(x1, x2, x3, y1, y2, y3)
    print(x.shape, x_acc_g.shape, x_gyro_g.shape, x_tot_acc_g.shape, mu.shape, logvar.shape, xx_agt.shape)


