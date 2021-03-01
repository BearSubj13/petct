#https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def attention(q, k, v, d_k, mask=None):    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
      
    output = torch.matmul(scores, v)
    return output


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len].to(device)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128):
        super().__init__() 
        # We set d_ff as a default to 128
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):               
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        
    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.attn(x2, x2, x2, mask)
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        #self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = src
        #x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        #self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model + 1, heads), N)
        self.norm = Norm(d_model + 1)
        self.fc1 = nn.Linear(d_model + 1, d_model)

    def forward(self, x, fi):
        fi = fi.unsqueeze(1)
        x = torch.cat([x, fi], dim=1)
        x = x.unsqueeze(1)
        for i in range(self.N):
            x = self.layers[i](x)
        x = self.norm(x)
        y = self.fc1(x)
        return y


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, heads=heads, N=2)
        self.decoder = Decoder(d_model=d_model, heads=heads, N=2)
        #self.out = nn.Linear(d_model, trg_vocab)
        self.special_token = nn.Parameter(torch.ones([1, 1, d_model]))

    def forward(self, src, trg=None):
        special_token = self.special_token.repeat(src.shape[0], 1, 1)
        src = torch.cat([special_token, src], dim=1)
        e_outputs = self.encoder(src, mask=None)
        special_token_output = e_outputs[:,0,:]
        fi = torch.zeros([src.shape[0]])
        d_output = self.decoder(special_token_output, fi)
        return d_output


def generate_curve(number_of_points=100, number_of_curves=1):
    w = torch.randint(low=0, high=number_of_points, size=(number_of_curves, 1))
    w = w.repeat(1, number_of_points).unsqueeze(2)   
    x = torch.linspace(0, 1, steps=number_of_points)
    x = x.unsqueeze(0).repeat(number_of_curves, 1).unsqueeze(2)
    cos_array = torch.cos(math.pi*w*x)
    sin_array = torch.sin(math.pi*w*x)
    y = torch.cat([cos_array, sin_array], dim=2)
    return x, y


if __name__ == "__main__":
    d_model = 2
    N = 2
    heads = 1
    device = "cpu"
    encoder = Encoder(d_model=d_model, heads=heads, N=2)
    transformer = Transformer(d_model=d_model, heads=heads, N=2)
    x = torch.zeros([1, 3, d_model])
    
    x, y = generate_curve(number_of_curves=3)
    output = transformer(y)

    # decoder = Decoder(d_model=d_model, heads=heads, N=2)
    # x = torch.zeros([3, d_model])
    # fi = torch.FloatTensor([0, 0, 1])
    # y = decoder(x, fi)
    print(output.shape)
