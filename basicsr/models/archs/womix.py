## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.ops.deform_conv import DeformConv2d
from pdb import set_trace as stx
import numbers
import math
from ptflops import get_model_complexity_info
from einops import rearrange
import numpy as np
import torchvision


freqs_dict = dict()

##########################################################################
## Layer Norm
##rotary_pos_embed


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape,path):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (path, normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        #assert len(normalized_shape) == 1
        self.path=path
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(sigma + 1e-5)
        x = rearrange(x, '(p b) n c -> b n p c',p=self.path)

        return rearrange(x * self.weight, 'b n p c -> (p b) n c')


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, path):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (path, normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        #assert len(normalized_shape) == 1
        self.path = path
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5)

        x=rearrange(x,'(p b) n p c -> b n p c',p=self.path)
        return rearrange(x * self.weight + self.bias, 'b n p c -> (p b) n c')


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type,path):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim,path)
        else:
            self.body = WithBias_LayerNorm(dim,path)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, num_path):
        super(FeedForward, self).__init__()

        hidden_features = int(dim//num_path * ffn_expansion_factor)*num_path

        self.project_in_1 = nn.Conv2d(dim, hidden_features , kernel_size=1,groups=num_path, bias=bias)
        self.project_in_2 = nn.Conv2d(dim, hidden_features, kernel_size=1, groups=num_path, bias=bias)

        self.dwconv_1 = nn.Conv2d(hidden_features , hidden_features , kernel_size=3, stride=1, padding=1,
                                groups=hidden_features , bias=bias)
        self.dwconv_2 = nn.Conv2d(hidden_features , hidden_features , kernel_size=3, stride=1, padding=1,
                                groups=hidden_features , bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1,groups=num_path, bias=bias)
        self.num_path=num_path
    def forward(self, x):
        x = rearrange(x, '(p B) c h w -> B (p c) h w',  p=self.num_path)
        x1 = self.dwconv_1(self.project_in_1(x))
        x2 = self.dwconv_2(self.project_in_2(x))
        #x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, 'B (p c) h w -> (p B) c h w',  p=self.num_path)
        return x

class refine_att(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window,path):

        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:

            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2#��ֹ�߶Ȳ���
            cur_conv=nn.Conv2d(
                cur_head_split * Ch*path,
                cur_head_split*path,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split*path,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.num_path=path
        self.channel_splits = [ x * Ch for x in self.head_splits]

    def forward(self, v, size):
        """foward function"""
        B, h, N, Ch = v.shape
        H, W = size

        v_img = v

        v_img = rearrange(v_img, "B h (H W) Ch -> B h Ch H W", H=H, W=W)
        v_img = rearrange(v_img , "b h Ch H W -> b (h Ch) H W", H=H, W=W)#���ԭά��
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        v_img_list_reshape=[]
        for i in range(len(v_img_list)):#ͬ�߶ȵ���һ��
            v_img_list_reshape.append(rearrange(v_img_list[i], "(p B) c H W -> B (p c) H W", H=H, W=W, p=self.num_path))
        v_att_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list_reshape)
        ]#ÿ��ͨ������һ������ͼ
        v_img_list_reshape=[]
        for i in range(len(v_att_list)):
            v_img_list_reshape.append(rearrange(v_att_list[i], "B (p c) H W -> (p B) c H W", H=H, W=W, p=self.num_path))

        v_att = torch.cat(v_img_list_reshape, dim=1)
        v_att = rearrange(v_att, "B (h Ch) H W -> B h (H W) Ch", h=h)


        return v_att


class ECABlock(nn.Module):
    """Efficient Channel Attention Block"""

    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.channels = channels

        # ����Ӧѡ������˴�С
        k_size = int(abs((math.log2(channels) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        # Feature descriptor
        y = self.avg_pool(x)

        # 1D convolution
        y = y.view(batch_size, 1, -1)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.sigmoid(y)
        y = y.view(batch_size, -1, 1, 1)  # [B, C, 1, 1]

        return x * y
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,shared_refine_att=None,qk_norm=1,focusing_factor=8,N=256*256,init_value=0.5):
        super(Attention, self).__init__()
        self.norm=qk_norm
        self.num_heads = num_heads
        self.scale1 = nn.Parameter(torch.ones(1,dim, 1, 1))
        #self.Leakyrelu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1,groups=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # print(self.qkv_dwconv)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1,groups=1,bias=bias)
        self.dconv=nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim,bias=bias)
        # if num_heads == 8:
        #     crpe_window = {
        #         3: 2,
        #         5: 3,
        #         7: 3
        #     }
        # elif num_heads == 1:
        #     crpe_window = {
        #         3: 1,
        #     }
        # elif num_heads == 2:
        #     crpe_window = {
        #         3: 2,
        #     }
        # elif num_heads == 4:
        #     crpe_window = {
        #         3: 2,
        #         5: 2,
        #     }
        # self.refine_att = refine_att(Ch=dim // num_heads,
        #                              h=num_heads,
        #                              window=crpe_window,
        #                              path=1)
        # self.focusing_factor=focusing_factor
        self.scale = nn.Parameter(torch.ones(1,dim, 1, 1))
        self.scale_v = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.rule = nn.Parameter(torch.ones(1, dim, 1, 1))
        pool_size = int(dim ** 0.5)
        self.raw_value = nn.Parameter(torch.tensor(init_value))
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=(pool_size, pool_size))
        self.attn_drop = nn.Dropout(0.1)
        # #self.N=N
        # self.one_M=nn.Parameter(torch.ones(1), requires_grad=False)
        self.eca=ECABlock(dim)
    def forward(self, x):
        b, c, h, w = x.shape
        
        relu = nn.ReLU(inplace=False)
        rule = self.rule.expand(b, c, h, w)
        #x= rearrange(x, '(p B) c h w -> B (p c) h w', B=b, p=1)#һ��ͼƬ�Ķ�߶�������ͨ��ά�Ƚ����ں�
        qkv = self.qkv_dwconv(self.qkv(x))#ͬһ�߶ȸ�����������˳���ŷ�
        derect=relu(self.dconv(x))
        factor=rule*derect
        #qkv = rearrange(qkv, 'B (p c) h w -> (p B) c h w', B=b, p=1)
        q, k, v = qkv.chunk(3, dim=1)#8 x 24 x h x w
        scale_v= self.sigmoid(self.scale_v)
        v=v*scale_v
        #agent_tokens = self.pool(x)
        
        #agent_tokens = rearrange(agent_tokens, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        n=self.sigmoid(self.raw_value)
        project_Q=n*self.pool(q)+(1-n)*self.max_pool(q)
        project_K=n*self.pool(k)+(1-n)*self.max_pool(k)
        project_Q = rearrange(project_Q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        project_K = rearrange(project_K, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        
        project_Q_norm= torch.norm(project_Q, p=2, dim=-1, keepdim=True)+1e-8
        project_Q_1= torch.div(project_Q, project_Q_norm)
        project_K_norm= torch.norm(project_K, p=2, dim=-2, keepdim=True)+1e-8
        project_K_1= torch.div(project_K, project_K_norm)
        q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)+1e-8
        q_1 = torch.div(q, q_norm)#���й�һ��
        k_norm = torch.norm(k, p=2, dim=-2, keepdim=True)+1e-8
        k_1 = torch.div(k, k_norm)
        

        attn_k=F.softmax(project_Q_1@k_1,dim=-1)
        attn_k=self.attn_drop(attn_k)
        attn_q=F.softmax(q_1@project_K_1,dim=-1)
        attn_q=self.attn_drop(attn_q)
        agent_v=attn_k@v
        out=attn_q@agent_v
        #attn_weights = q_1 @ k_1  # [b, heads, seq_len, seq_len]
        #attn_weights = F.softmax(attn_weights /(c**0.5),dim=-1)  # Ӧ��softmax
        #out = attn_weights @ v  # [b, heads, seq_len, dim]������δ���
        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)
        # q_2 = relu(q) ** self.focusing_factor#��ǿǿ������������������
        # k_2 = relu(k) ** self.focusing_factor
        #
        # q_2=(q_2/(q_2.norm(dim=-1, keepdim=True)+1e-8))#*q_norm
        # k_2 = (k_2 / (k_2.norm(dim=-2, keepdim=True)+1e-8)) #* k_norm

        # refine_weight = self.refine_att(v, size=(h, w))#���������Ŀռ�λ�ú�ͨ������ԣ�����һ��Ȩ�ؾ���������ǿ��Ҫ����������޹ر�����ע������Ӧ��b head (h w) 1
        # refine_weight = self.sigmoid(refine_weight)#���δ��һ����Ȩ�أ�����ͨ�� sigmoid ӳ�䵽 (0,1) ��Χ��b head (h w) 1
        # attn_2 = k_2@v
        # attn_1=k_1@v
        scale= self.sigmoid(self.scale)
        scale1 = self.sigmoid(self.scale1)
        out1=self.eca(x)
        out=factor*(out+scale1*out1+scale*x)
        #out_numerator = torch.sum(v, dim=-2).unsqueeze(2)+(q_1@attn_1)+scale*(q_2@attn_2)#self.one_M \  ����value���

       #  N=h*w
       #  one_M=self.one_M*N
       #  target_shape = (N, c // self.num_heads)#��hxw,ch��
       #  expanded_tensor = one_M.expand(target_shape)
       #  out_denominator = expanded_tensor + q_1 @ torch.sum(k_1, dim=-1).unsqueeze(
       #      3).repeat(1, 1, 1, c // self.num_heads) + \
       #                    q_2 @ torch.sum(scale * k_2, dim=-1).unsqueeze(3).repeat(1, 1, 1, c // self.num_heads) + 1e-8
       #
       #  out = torch.div(out_numerator, out_denominator)# ʵ������ softmax �Ĺ�һ����ʵ��̩��չ��
       # #b head (h w) c
       #  out = out* (self.temperature.repeat_interleave(b,0))+refine_weight


        #out = rearrange(out, '(p b) c h w-> b (p c) h w',  h=h, w=w, p=1)
        out = self.project_out(out)
       
        #out = rearrange(out, 'b (p c) h w-> (p b) c h w', h=h, w=w, p=1)
        return out

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if isinstance(self.normalized_shape, int):
            # ���� (C,) -> (C, 1, 1) �����
            weight = self.weight.view(-1, 1, 1)
            bias = self.bias.view(-1, 1, 1)
        else:
            # ���� [C, H, W] �����
            weight = self.weight
            bias = self.bias
        return x * weight + bias
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,shared_refine_att=None,qk_norm=1,N=256*256, path_emb_dim=48,num_path=2):
        super(TransformerBlock, self).__init__()
        #self.num_path=num_path#�߶�
        #self.norm1 = LayerNorm(dim, LayerNorm_type,num_path)#bxcxhxw
        self.Dyt1 = DynamicTanh(dim)
        self.attn = Attention(dim, num_heads, bias,shared_refine_att=shared_refine_att,qk_norm=qk_norm)#bxcxhxw
        #self.norm2 = LayerNorm(dim, LayerNorm_type,num_path)
        self.Dyt2 = DynamicTanh(dim)
        self.ffn = FeedForward(dim*num_path, ffn_expansion_factor, bias, num_path)

    def forward(self, x):

        x = x + self.attn(self.Dyt1(x))

        x = x + self.ffn(self.Dyt2(x))
  

        return x


class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""

    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1,
            N=256*256,
            num_path=4,
            path_emb_dim=48
    ):
        super().__init__()

        self.num_layers = num_layers
        self.MHCA_layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm,
                N=N,
                path_emb_dim=path_emb_dim,
                num_path=num_path
            ) for idx in range(self.num_layers)
        ])



    def forward(self, x, size):
        # b,_,_,_=x[0].shape
        b, _, _, _ = x.shape
        """foward function"""



        # x=torch.cat(x,dim=0)
        x = x.flatten(2).transpose(1, 2).contiguous()#(B, C, H*W)to (B, H*W, C),contiguous()��ȷ���ڴ������������������������
        H, W = size
        B = x.shape[0]
        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()#ͳһ��ͬ����߶ȵĿռ�ߴ絽 (H, W)

        for i in range(len(self.MHCA_layers)):
            x = self.MHCA_layers[i](x)

        return x





class MHCA_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""

    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            num_path=4,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1,
            N=256*256,
            path_emb_dim=48
    ):
        super().__init__()
        self.mhca_blk=MHCAEncoder(
                embed_dim,
                num_layers,
                num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm,
                N=N,
                num_path=num_path,
                path_emb_dim=path_emb_dim,
            )
        #self.aggregate = SKFF(embed_dim,height=num_path)
        #self.num_path=num_path

    def forward(self, inputs):
        """foward function"""
        #att_outputs = [self.InvRes(inputs[0])]
        # b,_,h,w=inputs[0].shape
        b, _, h, w = inputs.shape
        x=inputs
        #for idx in range(len(inputs)):
        #    torch.cat((inputs[idx],pos[idx]),1)


        out=self.mhca_blk(x, size=(h,w))#bxcxhxw
        #att_outputs=out.chunk(self.num_path, dim=0)#��ά�ȷֿ�����4x24x126x126��,(4x24x126x126)
        #out = self.aggregate(att_outputs)


        return out



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class Conv2d_BN(nn.Module):


    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):

        x = self.conv(x)
        x = self.act_layer(x)

        return x


class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        #print(inp_feats)
        batch_size = inp_feats.shape[0]
        n_feats = inp_feats.shape[1]//self.height#ͨ���� 24
        
        #inp_feats = torch.cat(inp_feats, dim=1)#���߶�ƴ�� 4x48x126x126
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])#���5ά��4x2x24x126x126
        #print(inp_feats)
        feats_U = torch.sum(inp_feats, dim=1)#4x24x126x126,�߶Ƚ������ں�
        feats_S = self.avg_pool(feats_U)#4X24x1x1
        feats_Z = self.conv_du(feats_S)#4x4x1x1
        #print(feats_Z)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]#[(4X24x1x1),(4X24x1x1)]
        attention_vectors = torch.cat(attention_vectors, dim=1)#4x48x1x1
        #print(attention_vectors)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)#4x2x24x1x1
        # stx()
        #print("DEBUG-softmax-in:", attention_vectors.cpu(), attention_vectors.min(), attention_vectors.max(), torch.isnan(attention_vectors).any(), flush=True)
        attention_vectors = self.softmax(attention_vectors)
        #print("DEBUG-softmax-out:", attention_vectors.shape, attention_vectors.sum().item(), torch.isnan(attention_vectors).any().item(), flush=True)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V



class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1,1),

    ):
        super().__init__()
        self.offset_clamp=offset_clamp
        self.offset_generator=nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,
                                                      stride= 1,padding= 1,bias= False,groups=in_ch),
                                            nn.Conv2d(in_channels=in_ch, out_channels=2*(2*kernel_size+3)*(2*kernel_size+3),
                                                      kernel_size=1,
                                                      stride=1, padding=0, bias=False)

                                            )
        self.dcn=DeformConv2d(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    kernel_size=2*kernel_size+3,
                    stride= 1,
                    padding= kernel_size+1,
                    bias= False,
                    groups=in_ch
                    )#.cuda(7)
        #self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)


        #self.bn = norm_layer(out_ch)
        #self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()




    def forward(self, x):

        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset=torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])#.cuda(7)1
        x = self.dcn(x,offset)
        #x=self.pwconv(x)

        #x = self.act(x)
        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 idx=0,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1,1)):
        super().__init__()


        self.patch_conv = DWConv2d_BN(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                act_layer=act_layer,
                offset_clamp=offset_clamp

            )


    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""
    """�ɶ�� DWCPatchEmbed ����ɵ���ȿɷ��������Ƕ��׶Ρ�"""

    def __init__(self, in_chans, embed_dim, num_path=4, isPool=False,offset_clamp=(-1,1)):
        super(Patch_Embed_stage, self).__init__()

        # self.patch_embeds = nn.ModuleList([
        #     DWCPatchEmbed(
        #         in_chans=in_chans//3 if idx == 0 else embed_dim//3,
        #         embed_dim=embed_dim,
        #         patch_size=idx,
        #         stride=1,
        #         idx=idx,
        #         offset_clamp=offset_clamp
        #     ) for idx in range(3)
        # ])
        self.conv_d1 = nn.Conv2d(in_channels=in_chans//3 , out_channels=in_chans//3, kernel_size=3, stride=1, padding=1, dilation=1)  # ��׼����������Ұ=3
        self.conv_d2 = nn.Conv2d(in_channels=in_chans//3 , out_channels=in_chans//3, kernel_size=3, stride=1, padding=2,dilation=2)  # �ն�����������Ұ=5��ע��paddingҲҪ��Ӧ����Ϊ dilation*(kernel_size-1)//2
        self.conv_d3 = nn.Conv2d(in_channels=in_chans//3 , out_channels=in_chans//3, kernel_size=3, stride=1, padding=3, dilation=3)  # �ն�����������Ұ=7
        self.conv_1x1=nn.Conv2d(in_chans, in_chans, 1, 1, 0)

    def forward(self, x):
        """foward function"""
        # att_inputs = []
        _,c,_,_=x.shape
        split_1,split_2,split_3 = torch.split(x, c // 3, dim=1)
        # split_1=self.patch_embeds[0](split_1)
        # split_2=self.patch_embeds[1](split_2)
        # split_3=self.patch_embeds[2](split_3)
        split_1=self.conv_d1(split_1)
        split_2=self.conv_d2(split_2)
        split_3=self.conv_d3(split_3)
        fused = torch.cat([split_1, split_2, split_3], dim=1)  # [B, C, H, W]
        output=self.conv_1x1(fused)
        # for idx in range(len(self.patch_embeds)):
        #     x = self.patch_embeds[idx](x)
        #     #print("patch�׶Σ�x����״",x.shape)
        #     att_inputs.append(x)



        return output


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
class MultiScaleProcess(nn.Module):
    def __init__(self,
                 dim=[24,48,72,96],
                 num_blocks=[2,3,3,4],
                 heads=[1, 1, 1, 1],
                 num_path=[2,2,2,2],  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 qk_norm=1,
                 offset_clamp=(-1,1),
                 N=128**2,
                 idx=0,
                 path_emb_dim=12):
        super(MultiScaleProcess, self).__init__()

        #

        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0], num_path=num_path[1], isPool=False,
                                                            offset_clamp=offset_clamp)
        # self.encoder_level1 = MHCA_stage(dim[0], dim[0], num_layers=num_blocks[0], num_heads=heads[0],
        #                                  ffn_expansion_factor=2.66,
        #                                  num_path=num_path[0], bias=False, LayerNorm_type='BiasFree', qk_norm=qk_norm,
        #                                  N=N // 4, path_emb_dim=path_emb_dim * 4)
        self.encoder_level1= EBlock(dim[0],4)

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[1], isPool=False,
                                                            offset_clamp=offset_clamp)
        # self.encoder_level2 = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[1], num_heads=heads[1],
        #                                  ffn_expansion_factor=2.66,
        #                                  num_path=num_path[1], bias=False, LayerNorm_type='BiasFree', qk_norm=qk_norm,
        #                                  N=N // 4, path_emb_dim=path_emb_dim * 4)


        self.patch_embed_encoder_level3 = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[2],
                                                            isPool=False, offset_clamp=offset_clamp)
        # self.encoder_level3 = MHCA_stage(dim[2], dim[2], num_layers=num_blocks[2], num_heads=heads[2],
        #                                  ffn_expansion_factor=2.66,
        #                                  num_path=num_path[2], bias=False, LayerNorm_type='BiasFree', qk_norm=qk_norm,
        #                                  N=N // 16, path_emb_dim=path_emb_dim * 4)


        self.patch_embed_encoder_level4= Patch_Embed_stage(dim[3], dim[3], num_path=num_path[3],
                                                    isPool=False, offset_clamp=offset_clamp)
        # self.encoder_level4 = MHCA_stage(dim[3], dim[3], num_layers=num_blocks[3], num_heads=heads[3],
        #                          ffn_expansion_factor=2.66, num_path=num_path[3], bias=False,
        #                          LayerNorm_type='BiasFree', qk_norm=qk_norm, N=N // 64, path_emb_dim=path_emb_dim * 4)

        # �ɱ��ξ�����ǿ (��ѡ)


    def forward(self,x_1, x_2,x_4,x_8):


        # ��ÿ���߶�Ӧ�ÿɱ��ξ�����ǿ
        x_1x_list = self.patch_embed_encoder_level1(x_1)
        x_1x = self.encoder_level1(x_1x_list)

        x_2x_list = self.patch_embed_encoder_level2(x_2)
        #x_2x = self.encoder_level2(x_2x_list)

        x_4xlist = self.patch_embed_encoder_level3(x_4)
        #x_4x = self.encoder_level3(x_4xlist)


        x_8x_list = self.patch_embed_encoder_level4(x_8)
       # x_8x = self.encoder_level4(x_8x_list)




        return x_1x+x_1,x_2x_list+x_2, x_4xlist +x_4, x_8x_list+x_8


class MultiScaleProcessBlock(nn.Module):
    def __init__(self,
                 dim=[24, 48, 72, 96],
                 num_blocks=[2, 3, 3, 4],
                 heads=[1, 2, 4, 8],
                 num_path=[2, 2, 2, 2],  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 qk_norm=1,
                 offset_clamp=(-1, 1),
                 N=128 ** 2,
                 path_emb_dim=12):
        super(MultiScaleProcessBlock, self).__init__()
        self.multiscale = nn.ModuleList([
            MultiScaleProcess(
                dim=dim,
                num_blocks=num_blocks,
                heads=heads,
                num_path=num_path,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                qk_norm=qk_norm,
                offset_clamp=offset_clamp,
                N=N,
                idx=idx,
                path_emb_dim=path_emb_dim
            ) for idx in range(2)
        ])
        self.down1_2 = Downsample(dim[0], dim[1])

        self.down2_4 = Downsample(dim[1], dim[2])
        self.up2_1 = Upsample(dim[1], dim[0])

        self.down4_8 = Downsample(dim[2], dim[3])
        self.up4_2 = Upsample(dim[2], dim[1])

        self.up8_4 = Upsample(dim[3], dim[2])

        #self.conv1=nn.Sequential(SKFF(dim[0],2),nn.ReLU(),
        #                    DWConv2d_BN(dim[0],dim[0],kernel_size=7,stride=1,act_layer=[],offset_clamp=offset_clamp),nn.ReLU()
         #                 )
        self.conv1_fused=SKFF(dim[0],2)                   
        self.conv1=nn.Sequential(nn.ReLU(),
                            Conv2d_BN(dim[0], dim[0]),nn.ReLU(),Conv2d_BN(dim[0], dim[0])
                          )
  
        self.conv2_fused =SKFF(dim[1],3)                    
        self.conv2 = nn.Sequential(nn.ReLU(),
                              DWConv2d_BN(dim[1], dim[1], kernel_size=5, stride=1, act_layer=[],
                                          offset_clamp=offset_clamp), nn.ReLU(),DWConv2d_BN(dim[1], dim[1], kernel_size=5, stride=1, act_layer=[],
                                          offset_clamp=offset_clamp)
                              )
                              
        self.conv4_fused = SKFF(dim[2],3)
        self.conv4 = nn.Sequential( nn.ReLU(),
                              DWConv2d_BN(dim[2], dim[2], kernel_size=5, stride=1, act_layer=[],
                                          offset_clamp=offset_clamp), nn.ReLU(),DWConv2d_BN(dim[2], dim[2], kernel_size=5, stride=1, act_layer=[],
                                          offset_clamp=offset_clamp)
                              )
                              
        self.conv8_fused =SKFF(dim[3],2)                      
        self.conv8 = nn.Sequential(nn.ReLU(),
                              DWConv2d_BN(dim[3], dim[3], kernel_size=3, stride=1, act_layer=[],
                                          offset_clamp=offset_clamp), nn.ReLU(),DWConv2d_BN(dim[3], dim[3], kernel_size=3, stride=1, act_layer=[],
                                          offset_clamp=offset_clamp)
                              )
    def forward(self,x, x_2, x_4, x_8):

        x_1x,x_2x,x_4x,x_8x = self.multiscale[0](x,x_2,x_4,x_8)
  
        x_1=torch.cat([x_1x,self.up2_1(x_2x)],1)
        x_1_fused=self.conv1_fused(x_1)
        x_1 = self.conv1(x_1_fused)+x_1_fused
      
        x_2=torch.cat([x_2x,self.down1_2(x_1),self.up4_2(x_4x)],1)
        x_2_fused=self.conv2_fused(x_2)
        x_2 = self.conv2(x_2_fused)+x_2_fused
       
        x_4=torch.cat([x_4x,self.down2_4(x_2),self.up8_4(x_8x)],1)
        x_4_fused=self.conv4_fused(x_4)
        x_4 = self.conv4(x_4_fused)+x_4_fused
       
        x_8=torch.cat([x_8x,self.down4_8(x_4)],1)
        x_8_fused=self.conv8_fused(x_8)
        x_8 = self.conv8(x_8_fused)+x_8_fused
    
        x_1, x_2, x_4, x_8 = self.multiscale[1](x_1, x_2, x_4, x_8)
      

        return x_1,x_2, x_4, x_8




##########################################################################
class MSSM(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=[24,48,72,96],
                 num_blocks=[2,3,3,4],
                 heads=[1, 1, 1, 1],
                 num_path=[1,1,1,1],  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 qk_norm=1,
                 offset_clamp=(-1,1),
                 N=128**2,
                 path_emb_dim=12

                 ):

        super(MSSM, self).__init__()
        #�¼�����
        self.multiscale=MultiScaleProcessBlock( dim,num_blocks,heads,num_path,qk_norm,offset_clamp, N,path_emb_dim)
        self.up8to4=Upsample(dim[3],dim[2])
        self.up4to2 = Upsample(dim[2], dim[1])
        self.up2to1 = Upsample(dim[1], dim[0])
        #��ʼ����
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim[0])

        self.down1_2 = Downsample(dim[0],dim[1])  ## From Level 1 to Level 2

        self.down2_3 = nn.Sequential(Downsample(dim[0],dim[1]),
                                     Downsample(dim[1],dim[2]) ) ## From Level 2 to Level 3
        self.down3_4 = nn.Sequential(Downsample(dim[0],dim[1]),
                                     Downsample(dim[1],dim[2]),
                                     Downsample(dim[2],dim[3])) ## From Level 3 to Level 4

        #### For Dual-Pixel Defocus Deblurring Task ####

        self.output = nn.Sequential(
            nn.Conv2d(dim[0], 3, kernel_size=3, stride=1, padding=1, bias=False, ),
        )
        self.output_2 = nn.Sequential(
            nn.Conv2d(dim[1], 3, kernel_size=3, stride=1, padding=1, bias=False, ),
        )
        self.output_4= nn.Sequential(
            nn.Conv2d(dim[2], 3, kernel_size=3, stride=1, padding=1, bias=False, ),
        )
        self.output_8= nn.Sequential(
            nn.Conv2d(dim[3], 3, kernel_size=3, stride=1, padding=1, bias=False, ),
        )
        self.conv1_fused = SKFF(dim[0],2)
        self.conv2_fused = SKFF(dim[1],2)
        self.conv3_fused = SKFF(dim[2],2)

    def forward(self, inp_img):
        #print(f"yuantu:{inp_img}",flush=True)
        #��һ���߶�c
        inp_enc_level1 = self.patch_embed(inp_img)  # 3X3������������ͨ����Ϊc��24�� x h x w

        # �ڶ����߶�2c x h/2 x w/2
        inp_enc_level2 = self.down1_2(inp_enc_level1)

        # �������߶�4c x h/4 x w/4
        inp_enc_level3 = self.down2_3(inp_enc_level1)

        # ���ĸ��߶�8c x h/8 x w/8
        inp_enc_level4 = self.down3_4(inp_enc_level1)

        out_finl1,out_finl2,out_finl3,out_finl4=self.multiscale(inp_enc_level1,inp_enc_level2,inp_enc_level3,inp_enc_level4)


        out_temp1=self.conv3_fused(torch.cat([out_finl3,self.up8to4(out_finl4)],1)) 
        out_temp2 = self.conv2_fused(torch.cat([out_finl2 ,self.up4to2(out_temp1)],1))             
        out_temp3=self.conv1_fused(torch.cat([out_finl1,self.up2to1(out_temp2)],1))        
        out_temp4=out_finl4
        # inp_latent_list = self.patch_embed_decoder_final1(out_temp3)
        # out_dec_level1 = self.decoder_final1(inp_latent_list)

        out1 = self.output(out_temp3+ inp_enc_level1) + inp_img
        out2=self.output_2(out_temp2+  inp_enc_level2 )
        out3=self.output_4(out_temp1+ inp_enc_level3)
        out4=self.output_8(out_temp4+inp_enc_level4)

        return [out4,out3,out2,out1]


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


from thop import profile
from thop import clever_format
import time
if __name__ == "__main__":
    # Create a random input tensor
    # x = torch.rand(1, 3, 128, 128).cuda()
    input_size = torch.rand(1, 3, 256, 256).cuda()
    # Initialize the model and move it to GPU
    # model = MB_TaylorFormer().cuda()
    model = MSSM().cuda()
    print(f"Number of parameters: {count_param(model)}")

    # Set the model to evaluation mode
    model.eval()
    
    
    
    
    # 计算MACs和参数量
    macs, params = get_model_complexity_info(
        model,
        (3, 256,256),
        as_strings=False,
        print_per_layer_stat=False,  # 打印每层统计
        verbose=True
    )
    
    # MACs（乘加运算次数）通常等于FLOPs/2
    # 对于PyTorch，get_model_complexity_info返回的是MACs
    flops = macs * 2  # 转换为FLOPs
    
    print("\n" + "="*60)
    print(f"参数量: {params/1e6:.2f} M")
    print(f"MACs: {macs/1e9:.2f} G")
    print(f"FLOPs: {flops/1e9:.2f} G")
    print("="*60)
    # Measure inference time
    num_runs = 10  # Number of runs to average the inference time
    total_time = 0.0

    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_size)
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_inference_time = total_time / num_runs
    print(f"Average inference time per image: {avg_inference_time:.6f} seconds")

# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
# https://arxiv.org/abs/2111.09881






