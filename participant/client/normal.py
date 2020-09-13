# encode=utf-8
import copy
from tqdm import tqdm
import random

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler, BatchSampler

from models.naive_fed_model import *


class Client(object):
    def __init__(self, name, train_set, sample_set: list, logger):
        """

        :param name: 客户端名字
        :param train_set: 训练集
        :param sample_set: 训练集数据索引
        """
        self.name = str(name)
        self.compromise = False
        self.train_set = train_set
        self.model = None
        self.cfg = None
        self.sample_set = sample_set
        self.train_loader = None
        self.loss_func = CrossEntropyLoss()
        self.logger = logger
        self.logger.logger.info('{}\n\tsample size: {}\n\tsample idx: {}'.format(self, len(sample_set), self.sample_set))

    def train(self, lr=0.1, batch_size=0, epoch_num=1) -> [dict, float, float]:
        """
        完成一次训练
        :param lr:
        :param batch_size:
        :param epoch_num:
        :return: 模型参数，损失，训练集准确率
        """
        if len(self.sample_set) == 0:
            return None, 0, 0

        self.model.train()
        if self.cfg.SOLVER.CUDA:
            self.model.cuda()
        train_loader = self._init_train_loader(batch_size)
        optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.5)
        train_loss = 0
        train_acc = 0

        for epoch in range(epoch_num):
            loss, acc = self._train_one_epoch(train_loader, optimizer)
            self.logger.logger.info('{} Train\tloss: {}\tacc: {}'.format(self, loss, acc))
            train_loss += loss
            train_acc += acc
            cuda.empty_cache()
        self.model.cpu()
        model_params = copy.deepcopy(self.model.state_dict())
        train_loss /= epoch_num
        train_acc /= epoch_num
        cuda.empty_cache()
        return model_params, train_loss, train_acc

    def update_model(self, model, cfg=None):
        """
        更新本地参数
        :param model: 新的参数
        :param cfg:
        :return:
        """
        self.model = copy.deepcopy(model)
        if cfg and self.cfg is None:  # TODO: 是否去掉后半句？
            self.cfg = cfg

    def _init_train_loader(self, batch_size=0) -> DataLoader:
        """
        创建训练用的DataLoader
        :param batch_size:
        :return:
        """
        random.shuffle(self.sample_set)
        sampler = SubsetRandomSampler(self.sample_set)
        if batch_size:
            sampler = BatchSampler(sampler, batch_size, drop_last=False)
        else:
            sampler = BatchSampler(sampler, len(sampler), drop_last=False)
        return DataLoader(self.train_set, batch_sampler=sampler, num_workers=2)

    def _train_one_epoch(self, train_loader, optimizer) -> [float, float]:
        train_loss = 0
        train_correct = 0
        for iter_, (inputs, targets) in enumerate(train_loader):
            if self.model.cfg.SOLVER.CUDA:
                inputs, targets = inputs.cuda(), targets.cuda()
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

    def eval(self, test_loader: DataLoader) -> [float, float]:
        """

        :param test_loader:
        :return:
        """
        self.model.eval()
        if self.cfg.SOLVER.CUDA:
            self.model.cuda()
        loss = 0
        correct = 0
        criterion = CrossEntropyLoss()
        pbar = tqdm(range(len(test_loader)))
        pbar.set_description('Client {} eval'.format(self.name))
        for iter_, _, (inputs, targets) in enumerate(zip(pbar, test_loader)):
            if self.cfg.SOLVER.CUDA:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            loss_ = criterion(outputs, targets)
            loss += float(loss_)
            _, predicted = torch.max(outputs.data, 1)
            correct += int(predicted.eq(targets).sum())
            pbar.set_postfix({
                'loss': '{}'.format(loss / ((iter_ + 1) * test_loader.batch_size)),
                'acc': '{}'.format(correct / ((iter_ + 1) * test_loader.batch_size))
            })
        self.model.cpu()
        loss = loss / (len(test_loader) * test_loader.batch_size)
        acc = correct / (len(test_loader) * test_loader.batch_size)
        self.logger.logger.info('{} Eval\tloss: {}\tacc: {}'.format(self, loss, acc))
        return loss, acc

    def __repr__(self):
        return '{} {}'.format(self.__class__.__name__, self.name)
