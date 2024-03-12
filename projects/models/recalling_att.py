import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm, dr_rate=0.0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x, sub_layer):
        out = x
        out, temp= sub_layer(out)
        out = self.norm(out)
        out = self.dropout(out)
        out = out + x
        return out, temp

class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, roi_feats, tgt_mask=None, src_tgt_mask=None):        
        out = roi_feats.permute(1, 0, 2)  # roi_feats.size() [B*NP, 49, 256]
        m_atps = []
        for layer in self.layers:
            out, m_atp = layer(out, tgt_mask, src_tgt_mask)
            m_atps.append(m_atp)
            
        tmp = torch.zeros_like(m_atp)
        
        for mat in m_atps:
            tmp += mat
        
        out = out.permute(1, 0, 2)  # roi_feats.size() [49, B*NP, 256]
        
        return out, tmp/len(m_atps)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0.0, d_model=256, d_emb=256):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

        self.codebook = nn.Embedding(200, 768).weight.data
        self.codebook = self.codebook.float().cuda() # codebook.size() = (num_cluster, 768)
        self.codebook.requires_grad = False
        
    def forward(self, roi_feats, tgt_mask, src_tgt_mask):
        out = roi_feats

        # out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        
        out, m_atp = self.residual2(out, lambda out: self.cross_attention(query=out, key=self.codebook.detach(), value=self.codebook.detach(), mask=src_tgt_mask))
        
        # out = self.residual3(out, self.position_ff)
        
        return out, m_atp

  
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0, cross=False):
        super(MultiHeadAttentionLayer, self).__init__()        
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)
        self.cross = cross
        if self.cross == False:
            self.k_fc = copy.deepcopy(qkv_fc)
            self.v_fc = copy.deepcopy(qkv_fc)
            
        self.out_fc = out_fc
        self.dropout = nn.Dropout(p=dr_rate)

    def calculate_attention(self, query, key, value, mask=None):
        
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, seq_len, seq_len)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = 100 * attention_score / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)

        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob) # [600, 8, 49, 200]
        
        out = torch.matmul(attention_prob, value)
        mean_atp =  torch.mean(attention_prob, dim = (1,2))
        return out, mean_atp         # mean_atp = [600, 200]

    def forward(self, query, key, value, mask=None):        
        n_batch = query.size(0)

        def transform(x, fc):                                               
            out = fc(x)                                                     
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)       
            out = out.transpose(1, 2)                                       
            return out
        
        def transform_codebook(x, n_size):                                        
            out = x.expand(n_size, x.size(-2), x.size(-1))                  
            out = out.view(n_size, -1, self.h, x.size(-1)//self.h)          
            out = out.transpose(1, 2)                                       
            return out
        
        query = transform(query, self.q_fc)         
        
        if self.cross == False:            
            key = transform(key, self.k_fc)         
            value = transform(value, self.v_fc)     
        else:
            key = transform_codebook(key, n_size=n_batch)
            value = transform_codebook(value, n_size=n_batch)

        out, m_atp = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)
        return out, m_atp

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2, dr_rate=0.0):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc2 = fc2

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def build_recalling_att(cfg):
    assert isinstance(cfg, dict)
    assert cfg.R_ATT == True

    d_emb = cfg.R_DEMB
    n_layer = cfg.R_NLAYER
    d_model = cfg.R_DMODEL
    h = cfg.R_H
    d_ff = cfg.R_DFF
    dr_rate = cfg.R_DROP
    norm_eps = cfg.R_NORM_EPS 

    self_attention = MultiHeadAttentionLayer(d_model=d_model,
                                            h=h,
                                            qkv_fc=nn.Linear(d_emb, d_model),
                                            out_fc=nn.Linear(d_model, d_emb),
                                            dr_rate=dr_rate,
                                            cross=False)

    cross_attention = MultiHeadAttentionLayer(d_model=d_model,
                                              h=h,
                                              qkv_fc=nn.Linear(d_emb, d_model),
                                              out_fc=nn.Linear(d_model, d_emb),
                                              dr_rate=dr_rate,
                                              cross=True)

    position_ff = PositionWiseFeedForwardLayer(fc1=nn.Linear(d_emb, d_ff),
                                               fc2=nn.Linear(d_ff, d_emb),
                                               dr_rate=dr_rate)

    norm = nn.LayerNorm(d_emb, eps=float(norm_eps))

    decoder_block = DecoderBlock(self_attention=self_attention,
                                cross_attention=cross_attention,
                                position_ff=copy.deepcopy(position_ff),
                                norm=copy.deepcopy(norm),
                                dr_rate=dr_rate,
                                d_model=d_model,
                                d_emb=d_emb)

    decoder = Decoder(decoder_block=decoder_block,
                      n_layer=n_layer,
                      norm=copy.deepcopy(norm))

    return decoder