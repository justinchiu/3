from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_ as clip_

class LM(nn.Module):
    def _loop(self, iter, learn, args):
        context = torch.enable_grad if learn else torch.no_grad

        loss = 0
        ntokens = 0
        rloss = 0
        rntokens = 0
        hidden_states = None
        with context():
            t = tqdm(iter)
            for i, batch in enumerate(t):
                if learn:
                    args.optimizer.zero_grad()
                x = batch.text
                y = batch.target
                nwords = y.ne(1).sum()
                T, N = y.shape
                if hidden_states is None:
                    hidden_states = self.init_hidden(N)
                logits, hidden_states = self(x, hidden_states)
                if learn:
                    hidden_states = (
                        [tuple(x.detach() for x in tup) for tup in hidden_states[0]],
                        tuple(x.detach() for x in hidden_states[1]),
                        hidden_states[2].detach(),
                    )
                logprobs = F.log_softmax(logits, dim=-1)
                # Ignore padding tokens
                logprobs.data[:,:,1].fill_(0)
                logp = logprobs.view(T*N, -1).gather(-1, y.view(T*N, 1))
                kl = 0
                nll = -logp.sum()
                nelbo = nll + kl
                if learn:
                    nelbo.div(nwords.item()).backward()
                    if args.clip > 0:
                        gnorm = clip_(self.parameters(), args.clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, args.clip)
                    args.optimizer.step()
                loss += nelbo.item()
                ntokens += nwords.item()
                rloss += nelbo.item()
                rntokens += nwords.item()
                if args is not None and i % args.report_interval == -1 % args.report_interval:
                    t.set_postfix(loss = rloss / rntokens, gnorm = gnorm)
                    rloss = 0
                    rntokens = 0
        return loss, ntokens

    def train_epoch(self, iter, args=None):
        return self._loop(iter=iter, learn=True, args=args)

    def validate(self, iter, args=None):
        return self._loop(iter=iter, learn=False, args=args)

    def forward(self):
        raise NotImplementedError

    def rnn_parameters(self):
        raise NotImplementedError

    def init_hidden(self):
        raise NotImplementedError
