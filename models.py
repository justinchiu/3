import torch
import torch.nn as nn
import torch.nn.functional as F

class LM(nn.Module):
    def _loop(self, iter, learn):
        context = torch.enable_grad if learn else torch.no_grad
        loss = 0
        ntokens = 0
        with context:
            for batch in iter:
                x = batch.text
                y = batch.target
                logits = self(x)
                logits.gather()
                kl = 0
                nll = 0
                nelbo = nll + kl
                loss += (nelbo.item() / bsz)
                if learn:
                    loss.backward()
                    optimizer.step()
                ntokens += nwords.item()
        return loss, ntokens

    def train(self, iter):
        return self._loop(iter)

    def validate(self, iter):
        return self._loop(iter)
