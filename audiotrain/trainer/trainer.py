import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    # 修正点1: 参数名改为 metric_ftns
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        # 修正点2: super调用也传 metric_ftns
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based iterator
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based iterator
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # 修正点3: 使用 self.metric_ftns
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.model.train()
        self.train_metrics.reset()
        
        all_preds = []
        all_targets = []

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            
            # 收集数据
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # 修正点4: 循环中使用 self.metric_ftns
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(all_targets, all_preds, epoch, phase='Train')

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # 修正点5: 循环中使用 self.metric_ftns
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # 绘制混淆矩阵
        self._plot_confusion_matrix(all_targets, all_preds, epoch, phase='Valid')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _plot_confusion_matrix(self, targets, preds, epoch, phase):
        cm = confusion_matrix(targets, preds, labels=[0, 1])
    
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Healthy(0)', 'Unhealthy(1)'],
            yticklabels=['Healthy(0)', 'Unhealthy(1)']
        )
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'{phase} Confusion Matrix - Epoch {epoch}')
    
        # 👇 核心：figure → image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = torch.from_numpy(img).permute(2, 0, 1)
    
        if self.writer is not None:
            self.writer.add_image(
                f'Confusion_Matrix/{phase}',
                img,
                epoch
            )
    
        plt.close(fig)