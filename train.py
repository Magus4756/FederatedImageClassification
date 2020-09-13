# encode=utf-8

from datetime import datetime
import torchvision
from torchvision import transforms

from config.config import cfg
from participant.client import Client, GaussianClient, LabelFlippingClient
from participant.server import Server
from solver.aggregator import *
from solver.data_loader import iid_sampler, non_iid_sampler
from utils.logging import Logger


def main(client_num, model, C=1.0, lr=0.1, epoch=1, aggregator=FedAvg(), iid=True, client_eval=False):
    logger = Logger('output/{}'.format(datetime.now().strftime('%y%m%d%H%M%S')), level='debug')

    # 读取配置
    cfg.merge_from_file('config/{}.yaml'.format(model))
    if C == 0:
        cfg.SOLVER.MAX_EPOCH = 2000
    cfg.freeze()

    # 加载数据
    transforms_train = transforms.Compose([transforms.Resize(cfg.MODEL.BACKBONE.INPUT_RESOLUTION),
                                           transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms_train
    )

    # 分配数据
    if iid:
        sample_idx = iid_sampler(train_set, client_num)
    else:
        sample_idx = non_iid_sampler(train_set, client_num)

    # 初始化联邦
    s = Server(
        config=cfg,
        dataset_name='MNIST',
        aggregator=aggregator,
        logger=logger
    )

    def flip(tensor):
        return (tensor + 2) % 10

    for i, sample in enumerate(sample_idx):
        if i < client_num - 10:
            c = Client(i, train_set, sample_idx[i], logger=logger)
        else:
            c = LabelFlippingClient(i, train_set, sample_idx[i], label_map=flip, logger=logger)
        s.append_clients(c)
    s.train(
        C=C,
        epoch_num=epoch,
        lr=lr,
        local_eval=client_eval
    )
    cfg.defrost()


def main2(client_num, model, C=1.0, lr=0.1, epoch=1, aggregator=FedAvg(), iid=True, client_eval=False):
    logger = Logger('output/{}'.format(datetime.now().strftime('%y%m%d%H%M%S')), level='debug')

    # 读取配置
    cfg.merge_from_file('config/{}.yaml'.format(model))
    if C == 0:
        cfg.SOLVER.MAX_EPOCH = 2000
    cfg.freeze()

    # 加载数据
    transforms_train = transforms.Compose([transforms.Resize(cfg.MODEL.BACKBONE.INPUT_RESOLUTION),
                                           transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms_train
    )

    # 分配数据
    if iid:
        sample_idx = iid_sampler(train_set, client_num)
    else:
        sample_idx = non_iid_sampler(train_set, client_num)

    # 初始化联邦
    s = Server(
        config=cfg,
        dataset_name='MNIST',
        aggregator=aggregator,
        logger=logger
    )
    for i, sample in enumerate(sample_idx):
        c = Client(i, train_set, sample_idx[i], logger=logger)
        s.append_clients(c)
    s.train(
        C=C,
        epoch_num=epoch,
        lr=lr,
        local_eval=client_eval
    )
    cfg.defrost()


if __name__ == '__main__':
    main(model='single_layer', iid=True, client_num=100, C=0.1, epoch=2, lr=0.1, aggregator=FedAvg(),
         client_eval=False)
    # main2(model='single_layer', iid=True, client_num=100, C=0.1, epoch=2, lr=0.1, aggregator=FedAvg(),
    #      client_eval=False)
