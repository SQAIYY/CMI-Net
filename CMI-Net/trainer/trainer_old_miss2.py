import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn

selected_d = {"outs": [], "trg": [],"probs": []}
class Trainer1(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, criterion_Miss_SC, criterion_Miss_MSE_x, criterion_Miss_MSE_h, metrics_ftns, optimizer, Miss_SC_optimizer, Miss_MSEx_optimizer, Miss_MSEh_optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metrics_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.criterion_Miss_SC = criterion_Miss_SC
        self.criterion_Miss_MSE_x = criterion_Miss_MSE_x
        self.criterion_Miss_MSE_h = criterion_Miss_MSE_h
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.lr_Miss_SC_scheduler = Miss_SC_optimizer
        self.lr_Miss_MSEx_scheduler = Miss_MSEx_optimizer
        self.lr_Miss_MSEh_scheduler = Miss_MSEh_optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.device = torch.device('cuda:0')
        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        overall_probs = []
        for batch_idx, (data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal, target_hr_modal) in enumerate(self.data_loader):
            data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal, target_hr_modal = \
                data_xyz.float().to(self.device), data_hr.float().to(self.device), data_xyz_miss.float().to(self.device), data_hr_miss.float().to(self.device), \
                target.to(self.device), target_xyz_miss.to(self.device), target_hr_miss.to(self.device), target_xyz_modal.to(self.device), target_hr_modal.to(self.device)
            #print("target",target.shape)

            self.optimizer.zero_grad()
            self.lr_Miss_SC_scheduler.zero_grad()
            self.lr_Miss_MSEx_scheduler.zero_grad()
            self.lr_Miss_MSEh_scheduler.zero_grad()
            #self.AMCS_optimizer.zero_grad()
            #x, x_hr_g, x_xyz_g, xx_hr_xyz
            #x, x_hr_g, x_xyz_g, mu, logvar, xx_hr_xyz
            output, x_hr_g, x_xyz_g, mu, logvar, xx_hr_xyz = self.model(data_hr, data_xyz, target_hr_miss.flatten(end_dim=1), target_xyz_miss.flatten(end_dim=1))
            output = output.to(self.device)
            x_hr_g = x_hr_g.to(self.device)
            x_xyz_g = x_xyz_g.to(self.device)
            mu = mu.to(self.device)
            logvar = logvar.to(self.device)
            xx_hr_xyz = xx_hr_xyz.to(self.device)

            output = output.flatten(end_dim=1)
            x_hr_g = x_hr_g.flatten(end_dim=1)
            x_xyz_g = x_xyz_g.flatten(end_dim=1)
            mu = mu.flatten(end_dim=1)
            logvar = logvar.flatten(end_dim=1)

            xx_hr_xyz = xx_hr_xyz.flatten(end_dim=1)
            #print(xx_xyz.shape)
            #print(xx_xyz.flatten(end_dim=1).shape)
            #projection_xyz_hr = torch.cat([xx_xyz, xx_hr], dim=0).to(self.device)
            target_xyz_hr = torch.cat([target_xyz_modal.flatten(end_dim=1), target_hr_modal.flatten(end_dim=1)], dim=0).to(self.device)
            target_cat = torch.cat([target, target], dim=0).to(self.device)
            #print(target_xyz_hr.shape)
            #print("output.shape", output.shape)
            #output = output.to(self.device)
            #output = output + self.adjustment
            #loss = self.criterion(output, target, self.class_weights, self.device)
            #print(output.shape)
            #print(target.shape)
            #self.criterion.set_epsilon(A)
            loss_ce = self.criterion(output, target.flatten()).to(self.device)
            loss_MSEh = self.criterion_Miss_MSE_h(x_hr_g, data_hr.flatten(end_dim=1), mu, logvar).to(self.device)
            loss_MSEx = self.criterion_Miss_MSE_x(x_xyz_g, data_xyz.flatten(end_dim=1), mu, logvar).to(self.device)
            loss_SC = self.criterion_Miss_SC(xx_hr_xyz, labels=target_cat, labelsM=target_xyz_hr).to(self.device)
            loss = loss_ce + 0.1 * loss_MSEh + 0.1 * loss_MSEx + 0.1 * loss_SC
            #print(projection_xyz_hr.flatten(end_dim=1).unsqueeze(dim=1).shape)
            #loss_AMCS = self.criterion_AMCS(projection_xyz_hr.flatten(end_dim=1).unsqueeze(dim=1), labels=target_xyz_hr, mask=None).to(self.device)
            #loss = loss + 0.02 * loss_AMCS
            loss.backward()
            self.optimizer.step()
            self.lr_Miss_SC_scheduler.step()
            self.lr_Miss_MSEx_scheduler.step()
            self.lr_Miss_MSEh_scheduler.step()
            #self.AMCS_optimizer.step()
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target.flatten()))

            if batch_idx % self.log_step == 0:
                #print(self.criterion.epsilon.grad)
                # 查看 epsilon 的梯度
                #epsilon_grad = self.criterion.epsilon.grad.item()
                #print(epsilon_grad)

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs, probs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
                selected_d["probs"] = probs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])
                overall_probs.extend(selected_d["probs"])
            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001


        return log, overall_outs, overall_trgs, overall_probs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            probs = np.array([])
            for batch_idx, (data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal,\
                            target_hr_modal) in enumerate(self.valid_data_loader):
                data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal, target_hr_modal = \
                    data_xyz.float().to(self.device), data_hr.float().to(self.device), data_xyz_miss.float().to(self.device), data_hr_miss.float().to(self.device), \
                    target.to(self.device), target_xyz_miss.to(self.device), target_hr_miss.to(self.device), target_xyz_modal.to(self.device), target_hr_modal.to(self.device)
                # print("target",target.shape)
                output, _, _, _ = self.model(data_hr, data_xyz, target_hr_miss.flatten(end_dim=1), target_xyz_miss.flatten(end_dim=1))
                output = output.to(self.device)
                #xx_xyz = xx_xyz.to(self.device)
                #xx_hr = xx_hr.to(self.device)
                output = output.flatten(end_dim=1)

                #output = output.to(self.device)
                #output = output + self.adjustment
                #loss = self.criterion(output, target, self.class_weights, self.device)
                #self.criterion.set_epsilon(A)
                loss = self.criterion(output, target.flatten()).to(self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target.flatten()))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                probs = np.append(probs, output.data.cpu().numpy())
                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())


        return self.valid_metrics.result(), outs, trgs,probs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, criterion_Miss_SC, criterion_Miss_MSE_x, criterion_Miss_MSE_h, metrics_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metrics_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.criterion_Miss_SC = criterion_Miss_SC
        self.criterion_Miss_MSE_x = criterion_Miss_MSE_x
        self.criterion_Miss_MSE_h = criterion_Miss_MSE_h
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.device = torch.device('cuda:0')
        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        overall_probs = []
        for batch_idx, (data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal, target_hr_modal) in enumerate(self.data_loader):
            data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal, target_hr_modal = \
                data_xyz.float().to(self.device), data_hr.float().to(self.device), data_xyz_miss.float().to(self.device), data_hr_miss.float().to(self.device), \
                target.to(self.device), target_xyz_miss.to(self.device), target_hr_miss.to(self.device), target_xyz_modal.to(self.device), target_hr_modal.to(self.device)
            #print("target",target.shape)

            self.optimizer.zero_grad()
            #self.AMCS_optimizer.zero_grad()
            #x, x_hr_g, x_xyz_g, xx_hr_xyz
            output, x_hr_g, x_xyz_g, mu, logvar, xx_hr_xyz = self.model(data_hr, data_xyz, target_hr_miss.flatten(end_dim=1), target_xyz_miss.flatten(end_dim=1))
            output = output.to(self.device)
            x_hr_g = x_hr_g.to(self.device)
            x_xyz_g = x_xyz_g.to(self.device)
            mu = mu.to(self.device)
            logvar = logvar.to(self.device)

            xx_hr_xyz = xx_hr_xyz.to(self.device)
            output = output.flatten(end_dim=1)
            x_hr_g = x_hr_g.flatten(end_dim=1)
            x_xyz_g = x_xyz_g.flatten(end_dim=1)
            mu = mu.flatten(end_dim=1)
            logvar = logvar.flatten(end_dim=1)
            xx_hr_xyz = xx_hr_xyz.flatten(end_dim=1)
            #print(xx_xyz.shape)
            #print(xx_xyz.flatten(end_dim=1).shape)
            #projection_xyz_hr = torch.cat([xx_xyz, xx_hr], dim=0).to(self.device)
            target_xyz_hr = torch.cat([target_xyz_modal.flatten(end_dim=1), target_hr_modal.flatten(end_dim=1)], dim=0).to(self.device)
            target_cat = torch.cat([target, target], dim=0).to(self.device)
            #print(target_xyz_hr.shape)
            #print("output.shape", output.shape)
            #output = output.to(self.device)
            #output = output + self.adjustment
            #loss = self.criterion(output, target, self.class_weights, self.device)
            #print(output.shape)
            #print(target.shape)
            #self.criterion.set_epsilon(A)
            loss_ce = self.criterion(output, target.flatten()).to(self.device)
            loss_MSEh = self.criterion_Miss_MSE_h(x_hr_g, data_hr_miss.flatten(end_dim=1), mu, logvar).to(self.device)
            loss_MSEx = self.criterion_Miss_MSE_x(x_xyz_g, data_xyz_miss.flatten(end_dim=1), mu, logvar).to(self.device)
            loss_SC = self.criterion_Miss_SC(xx_hr_xyz, labels=target_cat, labelsM=target_xyz_hr).to(self.device)
            loss = loss_ce + 0.001 * loss_MSEh + 0.001 * loss_MSEx + 0.01 * loss_SC
            #print(loss)
            #print(loss_MSEh)
            #print(loss_MSEx)
            #print(loss_SC)

            #print(projection_xyz_hr.flatten(end_dim=1).unsqueeze(dim=1).shape)
            #loss_AMCS = self.criterion_AMCS(projection_xyz_hr.flatten(end_dim=1).unsqueeze(dim=1), labels=target_xyz_hr, mask=None).to(self.device)
            #loss = loss + 0.02 * loss_AMCS
            loss.backward()
            self.optimizer.step()
            #self.AMCS_optimizer.step()
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target.flatten()))

            if batch_idx % self.log_step == 0:
                #print(self.criterion.epsilon.grad)
                # 查看 epsilon 的梯度
                #epsilon_grad = self.criterion.epsilon.grad.item()
                #print(epsilon_grad)

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs, probs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
                selected_d["probs"] = probs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])
                overall_probs.extend(selected_d["probs"])
            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.001


        return log, overall_outs, overall_trgs, overall_probs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            probs = np.array([])
            for batch_idx, (data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal,\
                            target_hr_modal) in enumerate(self.valid_data_loader):
                data_xyz, data_hr, data_xyz_miss, data_hr_miss, target, target_xyz_miss, target_hr_miss, target_xyz_modal, target_hr_modal = \
                    data_xyz.float().to(self.device), data_hr.float().to(self.device), data_xyz_miss.float().to(self.device), data_hr_miss.float().to(self.device), \
                    target.to(self.device), target_xyz_miss.to(self.device), target_hr_miss.to(self.device), target_xyz_modal.to(self.device), target_hr_modal.to(self.device)
                # print("target",target.shape)
                output, _, _, _, _, _ = self.model(data_hr, data_xyz, target_hr_miss.flatten(end_dim=1), target_xyz_miss.flatten(end_dim=1))
                output = output.to(self.device)
                #xx_xyz = xx_xyz.to(self.device)
                #xx_hr = xx_hr.to(self.device)
                output = output.flatten(end_dim=1)

                #output = output.to(self.device)
                #output = output + self.adjustment
                #loss = self.criterion(output, target, self.class_weights, self.device)
                #self.criterion.set_epsilon(A)
                loss = self.criterion(output, target.flatten()).to(self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target.flatten()))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                probs = np.append(probs, output.data.cpu().numpy())
                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())


        return self.valid_metrics.result(), outs, trgs,probs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)