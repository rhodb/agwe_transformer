import logging as log

import torch
import torch.nn as nn
import numpy as np

#from layer.transformer.Transformer import TransformerEncoder, TransformerEncoderLayer
from layer.linear import Linear
from layer.utils import pack_padded_sequence, pad_packed_sequence
from saver.saver import Saver


class TransformerEncoder_default(nn.Module, Saver):

  def __init__(self,
               num_layers,
               input_size,
               d_model,
               nhead,
               dim_feedforward,
               dropout,
               activation,
               num_embeddings=None,
               norm=None,
               proj=None):
    nn.Module.__init__(self)
    Saver.__init__(self)

    log.info(f" >> num_layers= {num_layers}")
    log.info(f" >> input_size= {input_size}")
    log.info(f" >> d_model= {d_model}")
    log.info(f" >> nhead= {nhead}")
    log.info(f" >> dim_feedforward= {dim_feedforward}")
    log.info(f" >> dropout= {dropout}")
    log.info(f" >> activation= {activation}")
    
    if num_embeddings is not None:
      log.info(f" >> num input embeddings= {num_embeddings}")
      self.emb = nn.Embedding(num_embeddings=num_embeddings + 1,
                              embedding_dim=d_model,
                              padding_idx=0)
    else:
      self.emb = nn.Linear(in_features=input_size, out_features=d_model)
    
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                    nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout, 
                                                    activation=activation)
    
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, 
                                                     num_layers=num_layers, 
                                                     norm=norm)
    
    self.pos_encoder = PositionalEncoding(d_model=d_model)
    
    #self.linear = nn.Linear(in_features = input_size, out_features = d_model)
    
    #self.input_size = input_size
    

    if proj is not None:
      log.info(f" >> proj after transformer= {proj}")
      self.proj = Linear(d_model, proj)
      self.project = True
    else:
      self.project = False

  @property
  def input_size(self):
    return input_size
    #return self.input_size
    
  @property
  def nhead(self):
    return nhead
    #return self.input_size

  @property
  def d_model(self):
    return self.encoder_layer.d_model

  @property
  def output_size(self):
    if hasattr(self, "proj"):
      return self.proj.output_size
    else:
      return self.d_model

  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, x, lens):

    x = x.to(self.device)
    
    x_out = []
    
    #acoustics view
    if len(x.shape) == 3:
        for i in range(x.shape[0]):
            x_i = x[i,0:lens[i],:]
            
            x_i = self.emb(x_i)
            
            x_i = x_i.unsqueeze(1)
            
            x_i = self.pos_encoder(x_i)
            
            x_i = self.transformer_encoder(x_i)
            
            x_i = x_i.sum(0)
            
            x_out.append(x_i)
        
    #phoneme view
    if len(x.shape) == 2:
        for i in range(x.shape[0]):
            x_i = x[i,0:lens[i]]
            
            x_i = self.emb(x_i)
            
            x_i = x_i.unsqueeze(1)
            
            x_i = self.pos_encoder(x_i)
            
            x_i = self.transformer_encoder(x_i)
            
            x_i = x_i.sum(0)
            
            x_out.append(x_i)

                  
    x_out = torch.squeeze(torch.stack(x_out))
    
    if self.project:
        x_out = self.proj(x_out)
        
    x_out = x_out/lens.cuda().unsqueeze(1)
    
    return x_out, lens



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)