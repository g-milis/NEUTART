"""Default Recurrent Neural Network Languge Model in `lm_train.py`."""

from typing import Any
from typing import List
from typing import Tuple

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.lm_interface import LMInterface
from espnet.nets.pytorch_backend.e2e_asr import to_device
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.utils.cli_utils import strtobool


class ClassifierWithState(nn.Module):
    """A wrapper for pytorch RNNLM."""

    def __init__(
        self, predictor, lossfun=nn.CrossEntropyLoss(reduction="none"), label_key=-1
    ):
        """Initialize class.

        :param torch.nn.Module predictor : The RNNLM
        :param function lossfun : The loss function to use
        :param int/str label_key :

        """
        if not (isinstance(label_key, (int, str))):
            raise TypeError("label_key must be int or str, but is %s" % type(label_key))
        super(ClassifierWithState, self).__init__()
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.label_key = label_key
        self.predictor = predictor

    def forward(self, state, *args, **kwargs):
        """Compute the loss value for an input and label pair.

        Notes:
            It also computes accuracy and stores it to the attribute.
            When ``label_key`` is ``int``, the corresponding element in ``args``
            is treated as ground truth labels. And when it is ``str``, the
            element in ``kwargs`` is used.
            The all elements of ``args`` and ``kwargs`` except the groundtruth
            labels are features.
            It feeds features to the predictor and compare the result
            with ground truth labels.

        :param torch.Tensor state : the LM state
        :param list[torch.Tensor] args : Input minibatch
        :param dict[torch.Tensor] kwargs : Input minibatch
        :return loss value
        :rtype torch.Tensor

        """
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = "Label key %d is out of bounds" % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[: self.label_key] + args[self.label_key + 1 :]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        state, self.y = self.predictor(state, *args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        return state, self.loss

    def final(self, state, index=None):
        """Predict final log probabilities for given state using the predictor.

        :param state: The state
        :return The final log probabilities
        :rtype torch.Tensor
        """
        if hasattr(self.predictor, "final"):
            if index is not None:
                return self.predictor.final(state[index])
            else:
                return self.predictor.final(state)
        else:
            return 0.0


# Definition of a recurrent net for language modeling
class RNNLM(nn.Module):
    """A pytorch RNNLM."""

    def __init__(
        self,
        n_vocab,
        n_layers,
        n_units,
        n_embed=None,
        typ="lstm",
        dropout_rate=0.5,
        emb_dropout_rate=0.0,
        tie_weights=False,
    ):
        """Initialize class.

        :param int n_vocab: The size of the vocabulary
        :param int n_layers: The number of layers to create
        :param int n_units: The number of units per layer
        :param str typ: The RNN type
        """
        super(RNNLM, self).__init__()
        if n_embed is None:
            n_embed = n_units

        self.embed = nn.Embedding(n_vocab, n_embed)

        if emb_dropout_rate == 0.0:
            self.embed_drop = None
        else:
            self.embed_drop = nn.Dropout(emb_dropout_rate)

        if typ == "lstm":
            self.rnn = nn.ModuleList(
                [nn.LSTMCell(n_embed, n_units)]
                + [nn.LSTMCell(n_units, n_units) for _ in range(n_layers - 1)]
            )
        else:
            self.rnn = nn.ModuleList(
                [nn.GRUCell(n_embed, n_units)]
                + [nn.GRUCell(n_units, n_units) for _ in range(n_layers - 1)]
            )

        self.dropout = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(n_layers + 1)]
        )
        self.lo = nn.Linear(n_units, n_vocab)
        self.n_layers = n_layers
        self.n_units = n_units
        self.typ = typ

        logging.info("Tie weights set to {}".format(tie_weights))
        logging.info("Dropout set to {}".format(dropout_rate))
        logging.info("Emb Dropout set to {}".format(emb_dropout_rate))

        if tie_weights:
            assert (
                n_embed == n_units
            ), "Tie Weights: True need embedding and final dimensions to match"
            self.lo.weight = self.embed.weight

        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def zero_state(self, batchsize):
        """Initialize state."""
        p = next(self.parameters())
        return torch.zeros(batchsize, self.n_units).to(device=p.device, dtype=p.dtype)

    def forward(self, state, x):
        """Forward neural networks."""
        if state is None:
            h = [to_device(x, self.zero_state(x.size(0))) for n in range(self.n_layers)]
            state = {"h": h}
            if self.typ == "lstm":
                c = [
                    to_device(x, self.zero_state(x.size(0)))
                    for n in range(self.n_layers)
                ]
                state = {"c": c, "h": h}

        h = [None] * self.n_layers
        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(x))
        else:
            emb = self.embed(x)
        if self.typ == "lstm":
            c = [None] * self.n_layers
            h[0], c[0] = self.rnn[0](
                self.dropout[0](emb), (state["h"][0], state["c"][0])
            )
            for n in range(1, self.n_layers):
                h[n], c[n] = self.rnn[n](
                    self.dropout[n](h[n - 1]), (state["h"][n], state["c"][n])
                )
            state = {"c": c, "h": h}
        else:
            h[0] = self.rnn[0](self.dropout[0](emb), state["h"][0])
            for n in range(1, self.n_layers):
                h[n] = self.rnn[n](self.dropout[n](h[n - 1]), state["h"][n])
            state = {"h": h}
        y = self.lo(self.dropout[-1](h[-1]))
        return state, y
