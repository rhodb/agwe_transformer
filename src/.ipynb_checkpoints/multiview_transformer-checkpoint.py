import logging as log
import torch
import torch.nn as nn

from layer.transformer.transformer import TransformerEncoder_default
from layer.linear import Linear
from saver.saver import Saver


class MultiViewTransformerEncoder(nn.Module, Saver):

  def __init__(self,
               view1_num_layers,
               view1_input_size,
               view1_d_model,
               view1_nheads,
               view1_dim_feedforward,
               view1_dropout,
               view1_activation,
               view1_proj,
               view1_norm,
               view2_num_layers,
               view2_input_size,
               view2_d_model,
               view2_nheads,
               view2_dim_feedforward,
               view2_dropout,
               view2_activation,
               view2_num_embeddings,
               view2_proj,
               view2_norm,
               proj=None,
               norm=None,
               loss_fn=None):
    nn.Module.__init__(self)
    Saver.__init__(self)

    self.net = nn.ModuleDict()

    log.info(f"view1:")
    self.net["view1"] = TransformerEncoder_default(num_layers=view1_num_layers,
                                                   input_size=view1_input_size,
                                                   d_model=view1_d_model,
                                                   nheads=view1_nheads,
                                                   dim_feedforward=view1_dim_feedforward,
                                                   dropout=view1_dropout,
                                                   activation=view1_activation,
                                                   norm=view1_norm,
                                                   proj=view1_proj):

    log.info(f"view2:")
    self.net["view2"] = TransformerEncoder_default(num_layers=view2_num_layers,
                                                   input_size=view2_input_size,
                                                   d_model=view2_d_model,
                                                   nheads=view2_nheads,
                                                   dim_feedforward=view2_dim_feedforward,
                                                   dropout=view2_dropout,
                                                   activation=view2_activation,
                                                   num_embeddings=view2_num_embeddings,
                                                   norm=view2_norm,
                                                   proj=view2_proj):

    if proj is not None:
      log.info(f"proj:")
      self.net["proj"] = Linear(self.net["view1"].output_size, proj)

    if loss_fn is not None:
      self.loss_fn = loss_fn

  @property
  def output_size(self):
    if "proj" in self.net:
      return self.net["proj"].output_size
    else:
      return self.net["view1"].output_size

  def forward_view(self, batch, view):

    if view == "view1":
      if "starts" not in batch:
        _, _, emb = self.net["view1"](batch["view1"], batch["view1_lens"])
      else:
        emb = []
        out, lens, _ = self.net["view1"](batch["view1"], batch["view1_lens"])
        for i in range(len(out)):
          for j in range(len(batch["starts"][i])):
            start = batch["starts"][i][j]
            end = batch["ends"][i][j]
            emb.append(out[i, start:end + 1].mean(0))
        emb = torch.stack(emb)
    if view == "view2":
      _, _, emb = self.net["view2"](batch["view2"], batch["view2_lens"])

    if "proj" in self.net:
      emb = self.net["proj"](emb)

    return emb

  def forward(self, batch, view=None):
    h1 = self.forward_view(batch, "view1")
    h2 = self.forward_view(batch, "view2")
    return h1, h2

  def backward(self, loss, max_grad_norm=10.0):
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(
        self.parameters(), max_grad_norm)
    return loss.data.item(), grad_norm

  def train_step(self, batch, batch_iter):

    inv = batch.pop("inv")
    ids = batch.pop("ids")
    view1, view2 = self.forward(batch)

    loss = self.loss_fn(view1, view2, inv)
    loss_val, grad_norm = self.backward(loss)
    if not self.loss_fn.average:
      loss_val /= float(len(inv))

    log.info(f"{batch_iter}) "
             f"loss={loss_val:.3f} "
             f"grad_norm={grad_norm:.2f} "
             f"num_segments={len(inv)} "
             f"num_words={len(ids)}")
