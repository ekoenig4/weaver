
import numpy as np
import tqdm
import time
import torch
import os

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import awkward, _concat
from ..logger import _logger

from .lightning import Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from pytorch_lightning.utilities.warnings import PossibleUserWarning

import warnings
warnings.simplefilter("ignore", category=PossibleUserWarning)

trainer = None

try:
    from pytorch_lightning.callbacks import RichProgressBar as LitProgressBar
    LitProgressBar()
except ModuleNotFoundError:
    from pytorch_lightning.callbacks import TQDMProgressBar as LitProgressBar

# Called in train.py to set trainer to be used to train lightning model
def set_trainer(args):
    path = os.path.dirname(args.model_prefix)

    return Trainer(
                accelerator="gpu" if args.gpus else "cpu",
                precision=16,
                max_epochs=1,
                max_steps=args.steps_per_epoch if args.steps_per_epoch else -1,
                enable_checkpointing=False, # Weaver already handles checkpoints
                default_root_dir=path,
                logger=TensorBoardLogger(
                    f'{path}/',
                    version='tb',
                ),
                callbacks=LitProgressBar()
            )

def lr_finder(model, loss_func, opt, data_config, train_loader, start_lr, end_lr, num_iter):
    min_lr, max_lr = float(start_lr), float(end_lr)
    if start_lr > end_lr: min_lr, max_lr = end_lr, start_lr
    model.set(data_config, loss_func, opt)
    lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, min_lr=min_lr, max_lr=max_lr, num_training=int(num_iter))
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_finder.png')
    
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    _logger.info(f"suggested starting lr={new_lr:0.1e}", color="green")

def train_lightning(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    assert isinstance(model, Lightning), "Model must be a subclass of Lightning"

    model.train()
    data_config = train_loader.dataset.config
    model.set(data_config, loss_func, opt=opt)

    start_time = time.time()
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.fit_loop.max_epochs += 1
    time_diff = time.time() - start_time


    metrics = [ f'{metric}={value:0.3f}' for metric, value in trainer.logged_metrics.items() ]
    _logger.info(f'training elapsed={time_diff:0.3f}s, ' + ', '.join(metrics), color='green')
    # _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    # _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
    #               (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    # if tb_helper:
    #     tb_helper.write_scalars([
    #         ("Loss/train (epoch)", total_loss / num_batches, epoch),
    #         ("MSE/train (epoch)", sum_sqr_err / count, epoch),
    #         ("MAE/train (epoch)", sum_abs_err / count, epoch),
    #         ])
    #     if tb_helper.custom_fn:
    #         with torch.no_grad():
    #             tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
    #     # update the batch state
    #     tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_lightning(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None, eval_metrics=None, tb_helper=None):
    assert isinstance(model, Lightning), "Model must be a subclass of Lightning"

    model.eval()
    if loss_func is None:
        loss_func = lambda *args: torch.Tensor([0])

    data_config = test_loader.dataset.config
    model.set(data_config, loss_func)

    mode = 'validation' if for_training else 'testing'
    evaluate = trainer.validate if for_training else trainer.test

    start_time = time.time()
    stats = evaluate(model, test_loader, verbose=False)
    time_diff = time.time() - start_time

    loss = next( value for metric, value in stats[0].items() if 'loss' in metric )
    metrics = [ f'{metric}={value:0.3f}' for metric, value in stats[0].items() ]
    _logger.info(f'{mode} elapsed={time_diff:0.3f}s, ' + ', '.join(metrics), color='green')

    # _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    # if tb_helper:
    #     tb_mode = 'eval' if for_training else 'test'
    #     tb_helper.write_scalars([
    #         ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
    #         ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
    #         ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
    #         ])
    #     if tb_helper.custom_fn:
    #         with torch.no_grad():
    #             tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    # scores = np.concatenate(scores)
    # labels = {k: _concat(v) for k, v in labels.items()}
    # metric_results = evaluate_metrics(
    #     labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    # _logger.info('Evaluation metrics: \n%s', '\n'.join(
    #     ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return loss
        # return total_loss / count
    else:
        # convert 2D labels/scores
        scores = model.arrays['score']
        labels = { key:array for key, array in model.arrays.items() if key != 'score' }
        observers = model.observers
        return loss, scores, labels, observers
