""" AI-ECG for LVH (FCN in different configurations) """

__author__ = "Thomas Kaplan"

import math
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class FCN1DConfig(Enum):
    WANG2016 = "WANG2016"
    ZHOU2024 = "ZHOU2024"


class LogCoshLoss(nn.Module):
    """CREDIT: https://datascience.stackexchange.com/a/102234"""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return torch.mean(
            (y_pred - y_true)
            + torch.nn.functional.softplus(-2.0 * (y_pred - y_true))
            - math.log(2.0)
        )


class GaussInvCDFLoss(nn.Module):

    def __init__(self, mean, std, k=1, eps=1e-6): # k=1
        super().__init__()
        self.normal_dist = dist.Normal(mean, std)
        self.k = k
        self.eps = eps

    def forward(self, pred, target):
        cdf_values = self.normal_dist.cdf(target)
        #weight = 1 / (1 - cdf_values + self.eps)
        weight = 1 / (torch.clamp(1 - cdf_values, min=1e-2) + self.eps)
        return (self.k * weight * torch.abs(pred - target)).mean()


class FCN1D(nn.Module):
    """Fully Convolutional Neural Network (1D)

    Ismail Fawaz, H., Forestier, G., Weber, J., Idoumghar, L., & Muller, P.-A. (2019).
    Deep learning for time series classification: A review.
    Data Mining and Knowledge Discovery, 33(4), 917–963. https://doi.org/10.1007/s10618-019-00619-1
    Keras: github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/classifiers/fcn.py
    """

    def __init__(
        self,
        n_channels,
        n_steps,
        n_meta,
        n_out,
        config: FCN1DConfig = FCN1DConfig.ZHOU2024,
        conv_batch_norm=True,
        conv_max_pool=True,
        conv_dropout=0.2,
        linear_dropout=0.2,
        seq_trans=False,
        multi_reg=False,
    ):
        super().__init__()
        self._n_meta = n_meta
        self._n_out = n_out
        self._logits = n_out > 1 and not multi_reg

        def _create_c1d_block(in_c, out_c, ksize):
            conv_layers = [
                nn.Conv1d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=ksize,
                    stride=1,
                    padding="same",
                ),
            ]
            if conv_batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_c))
            conv_layers.append(nn.LeakyReLU())
            if conv_max_pool:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, padding=1))
            return nn.Sequential(*conv_layers)

        if config == FCN1DConfig.WANG2016:
            # Similar to Wang et al. (2016) - includes BatchNorm1d, no MaxPool1d
            self.in1 = _create_c1d_block(n_channels, 128, 8)
            self.in2 = _create_c1d_block(128, 256, 5)
            self.in3 = _create_c1d_block(256, 128, 3)
            out_layers = [
                nn.Linear(128 + n_meta, 128 + n_meta),
                nn.LeakyReLU(),
                nn.Dropout(linear_dropout),
                nn.Linear(128 + n_meta, n_out),
            ]
        elif config == FCN1DConfig.ZHOU2024:
            # Similar to Zhou et al. (2024)'s encoder - inludes MaxPool1d, no BatchNorm1d nor GAP
            self.in1 = _create_c1d_block(n_channels, 32, 3)
            self.in2 = _create_c1d_block(32, 32, 3)
            self.in3 = _create_c1d_block(32, 32, 3)
            out_layers = [
                nn.Linear(32 + n_meta, 32 + n_meta),
                nn.LeakyReLU(),
                nn.Dropout(linear_dropout),
                nn.Linear(32 + n_meta, n_out),
            ]
        else:
            raise NotImplementedError(f"Unsupported config: {config}")

        self.seq_trans = None
        if seq_trans:
            self.seq_trans = SeqTrans(n_channels, n_steps)

        if self._logits:
            # Multi-class we'll use a loss-function without a nested sigmoid
            out_layers.append(nn.Softmax(dim=1))

        self.conv_drop = nn.Dropout(conv_dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Sequential(*out_layers)

        # GradCAM for last batch through `forward`
        self._cams = None

    @property
    def cams(self):
        if not self._logits or self._cams is None:
            return None
        else:
            return self._cams.detach().numpy()

    def forward(self, xs, xs_meta):
        # Optional sequence transformation
        if self.seq_trans is not None:
            xs = self.seq_trans(xs)
        # Primary stack
        features1 = self.in1(xs)
        features2 = self.in2(features1)
        features3 = self.in3(features2)
        features = self.conv_drop(features3)
        pooled_features = self.gap(features)
        pooled_features_flat = torch.flatten(pooled_features, 1)
        # Concatenate meta for final linear layer (dim 1 to reflect minibatch)
        if self._n_meta > 0:
            all_features_flat = torch.cat((pooled_features_flat, xs_meta), dim=1)
        else:
            all_features_flat = pooled_features_flat
        out = self.out(all_features_flat)

        if self._logits:
            cams = torch.matmul(self.out[-2].weight[:, : -self._n_meta], features)
            cams = F.relu(cams)
            self._cams = cams

        return out


def salience_by_dimension(model, xs, xs_meta):
    model.eval()
    assert xs.requires_grad, "Gradient required for tensor to calculate salience"
    assert xs_meta.requires_grad, "Gradient required for tensor to calculate salience"
    _ = model(xs, xs_meta)
    grads = xs.grad[:]
    grads /= torch.sqrt(torch.mean(torch.square(grads))) + 1e-6
    grads_meta = xs_meta.grad[:]
    grads_meta /= torch.sqrt(torch.mean(torch.square(grads_meta))) + 1e-6
    return grads.detach().numpy(), grads_meta.detach().numpy()
