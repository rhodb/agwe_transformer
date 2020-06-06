import logging as log

import torch
import torch.nn.functional as F


class MultiViewTripletLoss:

  def __init__(self, margin, k, min_k=None, extra=0, average=True):

    log.info(f" >> margin= {margin}")
    log.info(f" >> k= {k}")
    log.info(f" >> min_k= {min_k}")
    log.info(f" >> extra= {extra}")
    log.info(f" >> average= {average}")

    self.margin = margin
    self.k = k
    self.min_k = min_k or k
    self.extra = extra
    self.average = average

  def get_sims(self, x, y, inv, y_extra=None):

    n, d = x.shape
    m = y.shape[0]

    same = F.cosine_similarity(x, y[inv]).unsqueeze(-1)

    perms = torch.cat([(inv + i) % m for i in range(1, m)])
    diff = F.cosine_similarity(
        x.view(n, 1, d), y[perms].view(n, m - 1, d), dim=2)

    if y_extra is not None:
      diff_extra = F.cosine_similarity(
          x.view(n, 1, d), y_extra.view(1, -1, d), dim=2)
      diff = torch.cat([diff, diff_extra], dim=1)

    diff[diff > same] = -2.

    return same, diff, perms

  def get_word_sims(self, y, y_extra=None):

    m, d = y.shape
    m_range = torch.arange(m)

    rolls = torch.cat([torch.roll(m_range, i) for i in range(1, m)])
    word_diff = F.cosine_similarity(
        y.view(m, 1, d), y[rolls].view(m, m - 1, d), dim=2)

    if y_extra is not None:
      word_diff_extra = F.cosine_similarity(
          y.view(m, 1, d), y_extra.view(1, -1, d), dim=2)
      word_diff = torch.cat([word_diff, word_diff_extra], dim=1)

    word_diff[word_diff == 1.] = -2.

    return word_diff

  def get_topk(self, x, k, dim=0):

    return x.topk(min(k, x.shape[dim]), dim=dim)[0]

  def __call__(self, x, y, inv, y_extra=None):

    n, d = x.shape
    m = y.shape[0]

    if self.k > self.min_k:
      self.k -= 1

    k = min(self.k, m - 1)

    same, diff, perms = self.get_sims(x, y, inv, y_extra=y_extra)

    word_diff = self.get_word_sims(y, y_extra=y_extra)

    # Most offending words per utt
    diff_k = self.get_topk(diff, k=k, dim=1)
    obj0 = F.relu(self.margin + diff_k - same)

    # Most offending words per word
    word_diff_k = self.get_topk(word_diff, k=k, dim=1)
    obj1 = F.relu(self.margin + word_diff_k[inv] - same)

    # Most offending utts per word
    utt_diff_k = torch.zeros(m, k, device=diff.device)
    for i in range(m):
      utt_diff_k[i] = self.get_topk(diff.view(-1)[perms == i], k=k)
    obj2 = F.relu(self.margin + utt_diff_k[inv] - same)

    loss = (obj0 + obj1 + obj2).mean(1)
    
    #loss = (obj0 + obj2).mean(1)

    return loss.mean() if self.average else loss.sum()
