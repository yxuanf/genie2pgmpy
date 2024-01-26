import torch

# 无需在优化过程中更新先验的参数


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, inputs):
        inputs = inputs.cpu()
        prob1 = torch.exp(self.gaussian1.log_prob(inputs))
        prob2 = torch.exp(self.gaussian2.log_prob(inputs))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
