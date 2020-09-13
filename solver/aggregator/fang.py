import copy
import math
import torch

from .fed_avg import FedAvg


class Fang:
    def __init__(self, krum, compromised: list, client_num: int, cfg):
        self.krum = krum
        self.compromised = set(compromised)
        self.c = len(self.compromised)
        self.m = client_num
        self.d = 0
        self.lambda_ = 0
        self.min_lambda = 0.5e-5
        self.epoch = 1
        self.cfg = cfg
        self.previous_global = None

    def _get_before_attack_model_mean(self, models: list):
        mean_m = copy.deepcopy(models[0])
        for key in mean_m:
            for model in models[1:]:
                mean_m[key] += model[key]
            mean_m[key] /= len(models)
        return mean_m

    def _m2v(self, m):
        for k in m:
            m[k] = m[k].reshape([-1]).type(torch.FloatTensor)
        return torch.cat(list(m.values()))

    def _get_lambda(self, aim, global_model):
        """
        计算λ
        :param aim: 目标模型
        :param global_model: 本次的全局模型  # todo: 为什么常量λ需要该变量？
        """
        if self.d == 0:
            for key in aim:
                num_param = 1
                for n in aim[key].shape:
                    num_param *= n
                self.d += num_param
        # 计算minD^2
        # 转换为模型矩阵
        vectors = copy.deepcopy(global_model)
        for i, v in enumerate(vectors):
            vectors[i] = self._m2v(v)
        matrix_e = torch.FloatTensor(vectors)
        # 计算minD^2
        matrix_min = torch.zeros([matrix_e.shape[-1], matrix_e.shape[-1]])
        for i, v_i in enumerate(matrix_e):
            for j, v_j in enumerate(matrix_e[i:]):
                temp = v_i - v_j
                matrix_min[i][j + i] = matrix_min[j + i][i] = torch.matmul(temp, temp.T)
        matrix_min, _ = matrix_min.sort(axis=1)
        min_d = matrix_min[:, 1:-2].sum(axis=1).min()
        # 计算maxD
        matrix_e -= self._m2v(self.previous_global)
        max_d = torch.sqrt((matrix_e ** 2).sum(axis=1).max())
        # 计算λ
        lambda_ = math.sqrt(min_d / ((self.m - 2 * self.c - 1) * self.d)) + max_d / math.sqrt(self.d)
        return lambda_

    def __call__(self, chosen: list, weights: list):
        # 训练本轮空闲的恶意客户端
        inactivate_c = self.compromised - set(chosen)
        estimate_m = []
        for c in inactivate_c:
            m, _, _ = c.train()
            estimate_m.append(m)
        # 计算恶意客户端的平均模型，视为本次的全局模型的近似，即论文中的 w~
        activate_c = self.compromised - inactivate_c
        for idx, w in zip(chosen, weights):
            if idx in activate_c:
                estimate_m.append(w)
        compromised_mean = FedAvg()(estimate_m)
        # 计算模型改变的方向和大小，取反即为恶意的模型方向，即论文中的s~
        compromised_aim = copy.deepcopy(self.previous_global)
        for key in compromised_aim:
            compromised_aim[key] *= 2
            compromised_aim[key] -= compromised_mean[key]

        if self.lambda_ == 0:
            self.lambda_ = self._get_lambda(compromised_aim, estimate_m)
        # 计算恶意模型，即论文中的w_1'
        poisoned_m = copy.deepcopy(compromised_mean)
        for key in poisoned_m:
            poisoned_m[key] -= self.lambda_ * compromised_mean[key]
        # 在原参数列表中加入恶意模型，聚合
        for idx, c in enumerate(chosen):
            if c in self.compromised:
                weights[idx] = poisoned_m  # todo: 需要deepcopy吗？
        krum_global, krum_chosen = self.krum(chosen, weights)

        if ((self._m2v(krum_global) - self._m2v(poisoned_m)) ** 2).sum() > 0.1:  # 攻击失败
            self.lambda_ /= 2
        if self.lambda_ < self.min_lambda:
            print('攻击失败!')

        return krum_global, krum_chosen







