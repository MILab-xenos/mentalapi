# logger/visualization.py

from importlib import import_module
from datetime import datetime
import torch

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            # --- 核心修改点：同时捕获 ImportError 和 AttributeError ---
            try:
                self.writer = torch.utils.tensorboard.SummaryWriter(log_dir)
                self.selected_module = "torch.utils.tensorboard"
            except (ImportError, AttributeError): # <--- 修改在这里
                try:
                    self.writer = import_module('tensorboardX').SummaryWriter(log_dir)
                    self.selected_module = "tensorboardX"
                except ImportError:
                    logger.warning('Warning: TensorboardX not found. Result won\'t be logged to Tensorboard.')

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_figure'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        if self.writer is None:
            return
        step = global_step if global_step is not None else self.step
        self.writer.add_figure(tag, figure, step, close, walltime)

    def __getattr__(self, name):
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is None:
                    return
                if name not in self.tag_mode_exceptions:
                    tag = '{}/{}'.format(tag, self.mode)
                add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            try:
                attr = object.__getattribute__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr