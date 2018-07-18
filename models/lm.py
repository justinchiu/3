import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_ as clip_

class LM(nn.Module):
    def _loop(self, iter, learn, args):
        context = torch.enable_grad if learn else torch.no_grad
        loss = 0
        ntokens = 0
        with context():
            for batch in iter:
                x = batch.text
                y = batch.target
                N, T = y.shape
                logprobs = self(x)
                logp = logprobs.view(N*T, -1).gather(-1, y.view(N*T, 1))
                kl = 0
                nll = logp.sum()
                nelbo = nll + kl
                if learn:
                    nelbo.backward()
                    if self.clip > 0:
                        for param in self.rnn_parameters:
                            clip_(param, self.clip)
                    args.optimizer.step()
                loss += (nelbo.item() / bsz)
                ntokens += nwords.item()
        return loss, ntokens

    def train(self, iter, args):
        return self._loop(iter=iter, learn=True, args=args)

    def validate(self, iter, args):
        return self._loop(iter=iter, learn=False, args=args)

    def forward(self, _):
        raise NotImplementedError

