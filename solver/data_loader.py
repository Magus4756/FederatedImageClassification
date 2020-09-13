# encode=utf-8

import random

from torch.utils.data.dataset import Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def non_iid_sampler(train_set, client_num=100, shards=2):
    shards_num = client_num * shards
    size = len(train_set) // shards_num
    # 标签排序
    targets = train_set.targets
    _, idx = targets.sort()
    idx = idx.tolist()
    shards_idx = list(range(shards_num))
    random.shuffle(shards_idx)
    sampler = [list() for _ in range(client_num)]
    # 分配
    for i in range(client_num):
        shard1, shard2 = shards_idx[i * 2], shards_idx[i * 2 + 1]
        sampler[i] += idx[shard1 * size: (shard1 + 1) * size]
        sampler[i] += idx[shard2 * size: (shard2 + 1) * size]
        random.shuffle(sampler[i])
    return sampler


def iid_sampler(train_set, client_num=100):
    num_per_c = len(train_set) // client_num
    samplers = [list() for _ in range(client_num)]
    all_id = list(range(len(train_set)))
    random.shuffle(all_id)
    for i in range(len(samplers)):
        samplers[i] = all_id[num_per_c * i: num_per_c * (i + 1)]
    return samplers
