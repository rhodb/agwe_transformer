import logging as log

import torch
import torch.nn as nn

from layer.linear import Linear
from layer.utils import pack_padded_sequence, pad_packed_sequence
from saver.saver import Saver


class TransformerEncoder_default(nn.Module, Saver):

  def __init__(self,
               num_layers,
               input_size,
               d_model,
               nheads,
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
    log.info(f" >> nheads= {nheads}")
    log.info(f" >> dim_feedforward= {dim_feedforward}")
    log.info(f" >> dropout= {dropout}")
    log.info(f" >> activation= {activation}")

    if num_embeddings is not None:
      log.info(f" >> num input embeddings= {num_embeddings}")
      self.emb = nn.Embedding(num_embeddings=num_embeddings + 1,
                              embedding_dim=input_size,
                              padding_idx=0)
    
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                    nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout, 
                                                    activation=activation)
    
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, 
                                                     num_layers=num_layers, 
                                                     norm=norm)
    
    self.input_size = input_size
    

    if proj is not None:
      log.info(f" >> proj after transformer= {proj}")
      self.proj = Linear(self.output_size, proj)

  @property
  def input_size(self):
    return self.input_size

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

    if hasattr(self, "emb"):
      x = self.emb(x)

    x = pack_padded_sequence(x, lens)
    
    x = self.transformer_encoder(x)

    x, lens = pad_packed_sequence(x)

    if hasattr(self, "proj"):
      x = self.proj(x)

    return x, lens