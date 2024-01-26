import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from BNN.Gaussian import Gaussian
from BNN.MixtureGaussian import ScaleMixtureGaussian

# 超参数
PI = 0.4
SIGMA_1 = torch.tensor([math.exp(-0)])
SIGMA_2 = torch.tensor([math.exp(-4)])


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        # 初始化
        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(
            out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        # 初始化
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Set Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, inputs, sample=False, calculate_log_probs=False):
        # 训练或采样时，抽样决定权重
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_posterior = 0, 0
        return F.linear(inputs, weight, bias)



