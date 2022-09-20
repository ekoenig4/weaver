import time
from collections import Counter, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm

from .tools import _flatten_label, _flatten_preds, _get_batch_ptr


class Lightning(pl.LightningModule):

    def set(self, config, loss, opt=None):
        self.loss = loss
        self.config = config
        self.opt = opt

    def get_input(self, X, y, Z):
        """Called at the start of step in training. 
        You can redefine features or labels at this step. 
        batch_ptr is an extra variable calculated to keep track of batch number for each entry. 
        This function can be overriden for model specific operations

        Args:
            X (dict(Tensor)): dictionary of tensors defining as the input features to the network
            y (dict(Tensor)): dictionary of tensors defining the labels for the output of the network
            Z (dict(Tensor)): dictionary of tensors for observering variables (NOTE nothing should be done with this variable)

        Returns:
            tuple( list(Tensor), tuple(Tensor, Tensor), Tensor, dict(Tensor)): list of input tensors, label tensor, label mask, batch_ptr, and observing variables
        """
        inputs = [X[k] for k in self.config.input_names]
        label = y[self.config.label_names[0]]
        try:
            label_mask = y[self.config.label_names[0] + '_mask'].bool()
        except KeyError:
            label_mask = None
        batch_ptr = _get_batch_ptr(label_mask)

        return inputs, (label, label_mask), batch_ptr, Z

    def get_metrics(self, output, label, batch_ptr):
        """Defines metrics to calculate for the model in addition to loss.
        This is called at the end of each step and adds the new metrics to the training log.

        Args:
            output (Tensor): sparse output tensor from the model after padded values are masked out
            label (Tensor): sparse label tensor from the data after padded values are masked out
            batch_ptr (Tensor): sparse batch tensor which can be used as indicies in torch_scatter operations

        Returns:
            dict: dictionary of metrics calculated
        """
        return {}

    def get_loss(self, output, label, batch_ptr):
        """Defines loss to calculate for the model.
        This can be overriden for any special handling of input information

        Args:
            output (Tensor): sparse output tensor from the model after padded values are masked out
            label (Tensor): sparse label tensor from the data after padded values are masked out
            batch_ptr (Tensor): sparse batch tensor which can be used as indicies in torch_scatter operations

        Returns:
            Tensor: Loss calculated from loss function defined in model file
        """
        return self.loss(output, label, batch_ptr)

    def flatten_output(self, output, label, batch_ptr, label_mask):
        output = _flatten_preds(output, label_mask).squeeze()
        label = _flatten_label(label, label_mask)
        batch_ptr = _flatten_label(batch_ptr, label_mask)
        return output, label, batch_ptr

    def shared_step(self, batch, batch_idx, tag=None):
        inputs, (label, label_mask), batch_ptr, Z = self.get_input(*batch)
        output = self(*inputs)
        output, label, batch_ptr = self.flatten_output(output, label, batch_ptr, label_mask)

        metrics = {
            'loss': self.get_loss(output, label, batch_ptr),
            **self.get_metrics(output, label, batch_ptr)
        }
        arrays = {
            'score':output,
            self.config.label_names[0]:label,
            'batch':batch_ptr
        }

        self.log_scalar(metrics, tag=tag, prog_bar=True, on_epoch=True)
        self.log_histos({key:value for key,value in arrays.items() if key in ('score')}, tag=tag)

        return metrics, arrays, Z

    def training_step(self, batch, batch_idx):
        metrics, arrays, observers = self.shared_step(batch, batch_idx, tag='train')
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics, arrays, observers = self.shared_step(batch, batch_idx, tag='val')
        return { f'val_{key}':value for key,value in metrics.items() }

    def on_test_start(self):
        self.arrays = defaultdict(list)
        self.observers = defaultdict(list)

    def test_step(self, batch, batch_idx):
        metrics, arrays, observers = self.shared_step(batch, batch_idx, tag='test')

        for key, array in arrays.items():
            self.arrays[key].append( array.detach().cpu().numpy() )
        for key, array in observers.items():
            self.observers[key].append( array.detach().cpu().numpy() )

        return {f'test_{key}': value for key, value in metrics.items()}

    def on_test_end(self):
        self.arrays = { key:np.concatenate(array) for key, array in self.arrays.items() }
        self.observers = { field:np.concatenate(array) for field, array in self.observers.items() }

    def configure_optimizers(self):
        if self.opt is None:
            self.opt = torch.optim.Adam(self.parameters(), lr=0.02)
        return self.opt

    def log_scalar(self, metrics, tag=None, **kwargs):
        if tag is None: return
        for key, scalar in metrics.items():
            self.log(f'{tag}/{key}', scalar, **kwargs)

    def log_histos(self, histos, tag=None):
        if tag is None: return
        for key, histo in histos.items():
            if histo.ndim > 1: histo = histo[:,1]
            self.logger.experiment.add_histogram(
                f'{key}/{tag}',
                histo,
                self.current_epoch,
            )
