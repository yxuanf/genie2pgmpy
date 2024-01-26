import torch
from torch._prims_common import RETURN_TYPE
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.distributions import Normal
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms

CLASSES = 10
BATCH_SIZE = 256
TEST_BATCH_SIZE = 64


class Linear_BNN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        # 初始化输入输出维度
        self.input_features = input_features
        self.output_features = output_features

        # 初始化权重采样，mu：均值；rho：方差
        self.w_mu = nn.Parameter(
            torch.zeros(output_features, input_features).uniform_(-0.1, 0.1)
        )
        self.w_rho = nn.Parameter(
            torch.zeros(output_features, input_features).uniform_(-10, -9)
        )

        # 初始化偏置采样
        self.b_mu = nn.Parameter(torch.zeros(output_features).uniform_(-0.1, 0.1))
        self.b_rho = nn.Parameter(torch.zeros(output_features).uniform_(-10, -9))

        # 初始化权重和偏置
        self.w = None
        self.b = None

        # 初始化先验分布
        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, input):
        # 从标准正态分布中采样权重
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(device)
        # 获得服从均值为mu，方差为delta的正态分布的样本
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon
        # 采样偏置
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(device)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # 计算先验概率log p(w)，用于后续计算loss
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # 计算后验概率log p(w|\theta)，用于后续计算loss
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        # 后验概率
        self.log_post = (
            self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        )

        # 权重确定后，和BP网络层一样使用
        return F.linear(input, self.w, self.b)


class BNN(nn.Module):
    def __init__(self, hidden_unit):
        super().__init__()
        self.linear1 = Linear_BNN(512, hidden_unit)
        self.linear2 = Linear_BNN(hidden_unit, 10)

    def forward(self, x):
        # 激活函数选用sigmoid
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.softmax(x)
        return x

    def log_prior(self):
        # 计算先验概率
        return self.linear1.log_prior + self.linear2.log_prior

    def log_post(self):
        # 计算后验概率
        return self.linear1.log_post + self.linear2.log_post


# 计算loss
def sample_elbo(net: BNN, inputs, target, samples):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    # 初始化张量
    outputs = torch.zeros(samples, target.shape[0], 10)
    log_priors = torch.zeros(samples)
    log_posts = torch.zeros(samples)
    log_likes = torch.zeros(samples)
    # 预测(抽样取均值)
    for i in range(samples):
        # 预测
        outputs[i] = net(inputs)
        # 获得先验知识
        log_priors[i] = net.log_prior()
        # 获得后验概率
        log_posts[i] = net.log_post()
        # 计算似然函数
        log_likes[i] = criterion(net(inputs), target)

    # print(outputs)
    # 计算蒙特卡洛近似
    log_prior = log_priors.mean()
    log_post = log_posts.mean()
    log_like = log_likes.mean()
    # 计算最大证据下界，损失函数
    loss = log_post - log_prior + log_like
    return loss, log_like


def train(net: BNN, optimizer, epochs, samples=100):
    net.train()
    for _ in range(epochs):
        losses = 0
        for _, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss, log_like = sample_elbo(net, data, target, samples)
            loss.backward()
            optimizer.step()
            losses += log_like
        print(losses / len(train_loader))
    torch.save(net.state_dict(), "./model/bnn.pth")


def predict(net, testloader):
    pass


def dataset(filepath):
    df0 = pd.read_csv(filepath, index_col=0)
    feature = df0.iloc[:, :-1].values
    label = df0.iloc[:, -1].values
    data_tensor = torch.from_numpy(feature).float()
    label_tensor = torch.from_numpy(label)
    dataset = TensorDataset(data_tensor, label_tensor)
    train_size = int(len(dataset) * 0.65)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    # fileName = "./轴承数据集.csv"
    # df0 = pd.read_csv(fileName, index_col=0)
    # feature = df0.iloc[:, :-1].values
    # label = df0.iloc[:, -1].values
    # data_tensor = torch.from_numpy(feature).float()
    # label_tensor = torch.from_numpy(label)
    # dataset = TensorDataset(data_tensor, label_tensor)
    # train_size = int(len(dataset) * 0.65)
    # test_size = len(dataset) - train_size
    # # 划分训练集与测试集
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, test_size]
    # )
    # # 构建训练集
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = BNN(600).to(device)
    print(net)
    # epochs = 5
    # optimizer = optim.Adam(net.parameters(), lr=5e-3, weight_decay=5e-3)

    # train(net, optimizer, epochs)

    # loadnet = BNN(600).to(device)
    # loadnet.load_state_dict(torch.load("./model/bnn.pth"))
    # # 将模型设置为评估模式
    # loadnet.eval()
    # # 使用测试集进行推断
    # with torch.no_grad():
    #     loss_function = nn.CrossEntropyLoss(reduction="sum")
    #     correct_test = 0
    #     test_loss = 0
    #     for test_data, test_label in test_loader:
    #         test_data, test_label = test_data.to(device), test_label.to(device)
    #         test_output = loadnet(test_data)
    #         probabilities = F.softmax(test_output, dim=1)
    #         predicted_labels = torch.argmax(probabilities, dim=1)
    #         correct_test += (predicted_labels == test_label).cpu().sum().item()
    #         loss = loss_function(test_output, test_label)
    #         test_loss += loss.item()

    # test_accuracy = correct_test / len(test_loader.dataset)
    # test_loss = test_loss / len(test_loader.dataset)
    # print(f"Test Accuracy: {test_accuracy: .4f} Test Loss: {test_loss: .8f}")
