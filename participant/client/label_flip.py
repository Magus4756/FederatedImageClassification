# encode=utf-8

import torch

from .normal import Client


class LabelFlippingClient(Client):
    def __init__(self, name, train_set, sample_set, label_map, logger):
        super(LabelFlippingClient, self).__init__(name, train_set, sample_set, logger)
        self.compromise = True
        self.map_func = label_map

    def _train_one_epoch(self, train_loader, optimizer) -> [float, float]:
        train_loss = 0
        train_correct = 0
        for iter_, (inputs, targets) in enumerate(train_loader):
            if self.model.cfg.SOLVER.CUDA:
                inputs, targets = inputs.cuda(), targets.cuda()
            targets = self.map_func(targets)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
            # 数据统计
            _, predicted = torch.max(outputs.data, 1)
            train_correct += int(predicted.eq(targets).sum())
        train_acc = train_correct / len(self.sample_set)
        train_loss = train_loss / len(self.sample_set)
        return train_loss, train_acc
