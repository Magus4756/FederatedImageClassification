# encode=utf-8
import copy
from tqdm import tqdm
from datetime import datetime
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import cuda
import torchvision
from torch.utils.data import *
from torchvision import transforms

from models.imageclassifier import ImageClassifier
from models.naive_fed_model import *
from config.config import cfg
from .client import Client

matplotlib.use('Agg')


class Server(object):
    def __init__(self, config, dataset_name, aggregator, logger, iid=True):
        """

        :param config: 配置文件，在config文件夹中
        :param dataset_name: 训练/测试数据集
        :param aggregator: 聚合函数
        :param iid: 训练集是否独立同分布
        """
        self.clients = []
        self.client_num = 0
        self.cfg = config
        self.dataset_name = dataset_name
        self.model = self._load_model()
        self._aggregator = aggregator
        self.iid = iid
        self.test_batch_size = 64
        self.logger = logger
        trans = transforms.Compose([transforms.Resize(self.cfg.MODEL.BACKBONE.INPUT_RESOLUTION),
                                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_set = torchvision.datasets.MNIST(
            root='data',
            train=False,
            download=True,
            transform=trans
        )
        self.test_loader = DataLoader(test_set, batch_size=self.test_batch_size, shuffle=False)
        self.logger.logger.info('Server init\n\tmodel: \n{}\n\tdataset: {}\n\taggregator: {}\n\tiid: {}\n\tcfg:\n{}'.format(
            self.model, self.dataset_name,
            self._aggregator.__class__.__name__, self.iid, self.cfg
        ))

    def append_clients(self, clients: Client):
        """
        添加参与方
        :param clients: client的list或者单个client
        :return:
        """
        if isinstance(clients, list):
            for client in clients:
                self.append_clients(client)
        elif isinstance(clients, Client):
            self.clients.append(clients)
            self.client_num += 1
            clients.update_model(copy.deepcopy(self.model), self.cfg)
        self.clients.sort(key=lambda x: x.name)

    def train(self, C=1.0, epoch_num=1, lr=0.1, local_eval=False):
        """
        训练联邦模型
        :param C: 每轮训练的参与比例
        :param epoch_num: 总训练轮次
        :param lr: 学习率
        :param local_eval: 参与方训练完是否测试
        :return:
        """
        self.logger.logger.info('Start training\n\tmax_epoch: {}\t\tlr: {}\tbatch: {}'.format(
            self.cfg.SOLVER.MAX_EPOCH, lr,
            self.cfg.DATASETS.TRAIN_BATCH_SIZE))

        train_loss, train_acc = [], []
        eval_loss, eval_acc = [], []
        for epoch in range(self.cfg.SOLVER.MAX_EPOCH):
            # 选择客户端
            chosen_c = self._choose_client(C)
            self.logger.logger.info('Train {}/{}\tchoose: {}'.format(epoch, self.cfg.SOLVER.MAX_EPOCH, chosen_c))
            # 更新选中的客户端
            self._update_clients(chosen_c)
            # 训练
            local_w, loss, acc = self._train_one_step(
                epoch,
                chosen_c,
                lr=lr,
                batch_size=self.cfg.DATASETS.TRAIN_BATCH_SIZE,
                epoch_num=epoch_num,
                local_eval=local_eval
            )
            self.logger.logger.info('Train {}/{}\tloss: {}\tacc: {}'.format(epoch, self.cfg.SOLVER.MAX_EPOCH, loss, acc))
            train_loss.append(loss)
            train_acc.append(acc)
            # 聚合参数
            new_w = self._aggregator(local_w)
            # 更新模型
            self.model.load_state_dict(new_w)
            if epoch % 1 == 0:
                loss, acc = self.eval(epoch, self.cfg.SOLVER.MAX_EPOCH)
                eval_loss.append(loss)
                eval_acc.append(acc)
        self.eval(self.cfg.SOLVER.MAX_EPOCH, self.cfg.SOLVER.MAX_EPOCH)
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss)
        plt.ylabel('train_loss')
        plt.savefig('{}/train_loss.png'.format(self.logger.file_root))
        plt.figure()
        plt.plot(range(len(train_acc)), train_acc)
        plt.ylabel('train_acc')
        plt.savefig('{}/train_acc.png'.format(self.logger.file_root))
        plt.figure()
        plt.plot(range(len(eval_loss)), eval_loss)
        plt.ylabel('eval_loss')
        plt.savefig('{}/eval_loss.png'.format(self.logger.file_root))
        plt.figure()
        plt.plot(range(len(eval_acc)), eval_acc)
        plt.ylabel('eval_acc')
        plt.savefig('{}/eval_acc.png'.format(self.logger.file_root))

    def _choose_client(self, C=1.0):
        choose_num = min(max(round(self.client_num * C), 1), self.client_num)
        chosen = random.sample(self.clients, choose_num)
        chosen.sort(key=lambda x: x.name)
        return chosen

    def _load_model(self):
        if 'CNN_2_Layer' in self.cfg.MODEL.NAME:
            model = CNN_2_Layer(self.cfg)
        elif 'MNIST_2NN' in self.cfg.MODEL.NAME:
            model = MNIST_2NN(self.cfg)
        elif 'SingleLayer' in self.cfg.MODEL.NAME:
            model = SingleLayer(self.cfg)
        elif 'MLP' in cfg.MODEL.NAME:
            model = MLP(self.cfg)
        else:
            model = ImageClassifier(self.cfg)
        return model

    def _train_one_step(self, epoch: int, chosen_clients: list, lr: float, batch_size=0, epoch_num=1, local_eval=False):
        """
        训练一个step
        一个step有多个epoch，每个epoch有多个iter
        :param epoch: 本轮名称
        :param chosen_clients: 选择的客户端 [client_1, client_2, ...]
        :param lr:
        :param batch_size:
        :param epoch_num:
        :param local_eval:
        :return:
        """
        local_weights = []
        train_loss = 0
        train_acc = 0
        pbar = tqdm(range(len(chosen_clients)))
        pbar.set_description('Train {}/{}'.format(epoch, self.cfg.SOLVER.MAX_EPOCH))
        for i, (_, client) in enumerate(zip(pbar, chosen_clients)):
            w, loss, acc = client.train(lr, batch_size, epoch_num=epoch_num)
            local_weights.append(copy.deepcopy(w))
            train_loss += float(loss)
            train_acc += float(acc)
            pbar.set_postfix({
                'loss': '{:.8f}'.format(train_loss / ((i + 1) * cfg.DATASETS.TRAIN_BATCH_SIZE)),
                'acc': '{:.8f}'.format(train_acc / (i + 1)),
            })
            if local_eval:
                client.eval(self.test_loader)
        train_loss /= len(chosen_clients)
        train_acc /= len(chosen_clients)
        return local_weights, train_loss, train_acc

    def _update_clients(self, chosen):
        for client in chosen:
            client.update_model(copy.deepcopy(self.model))

    def eval(self, cur_epoch: int, max_epoch: int) -> [float, float]:
        """
        云端测试
        :param cur_epoch: 当前epoch
        :param max_epoch: 最大epoch
        :return: 损失，准确率
        """
        self.model.eval()
        criterion = CrossEntropyLoss()
        loss = 0
        correct = 0
        pbar = tqdm(range(len(self.test_loader)))
        pbar.set_description('Eval {}/{}'.format(cur_epoch, max_epoch))
        if self.cfg.SOLVER.CUDA:
            self.model.cuda()
        for iter_, (_, (inputs, targets)) in enumerate(zip(pbar, self.test_loader)):
            if self.cfg.SOLVER.CUDA:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            batch_l = criterion(outputs, targets)
            loss += float(batch_l)
            _, predicted = torch.max(outputs.data, 1)
            correct += int(predicted.eq(targets).sum())
            acc = correct / ((iter_ + 1) * self.test_batch_size)
            pbar.set_postfix({
                'loss': '{:.8f}'.format(loss / ((iter_ + 1) * self.test_batch_size)),
                'acc': '{:.8f}'.format(acc)
            })
        self.model.cpu()
        acc = correct / (len(self.test_loader) * self.test_batch_size)
        self.logger.logger.info('{} Eval {}/{}\tloss: {}\tacc: {}'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cur_epoch, max_epoch, loss, acc))
        cuda.empty_cache()
        return loss, acc

