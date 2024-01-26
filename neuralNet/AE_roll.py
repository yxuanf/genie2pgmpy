import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
    classification_report,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, encoding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, encoding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        output = self.decoder(encoded)
        return output


def get_recon_err(X):
    return torch.mean((model(X) - X) ** 2, dim=1).detach().numpy()


def make_data_labels(dataframe):
    """
    分离数据与标签
    dataframe:数据
    x_data:数据集
    y_data:对应标签
    """
    # 信号值
    x_data = dataframe.iloc[:, 1:-1]
    # 标签值
    y_label = dataframe.iloc[:, -1]

    x_data = torch.tensor(x_data.values.astype("float")).float()
    y_label = torch.tensor(y_label.values.astype("int64"))

    return x_data, y_label


class visualization:
    labels = ["Nomal", "Abnormal"]

    def draw_confusion_matrix(self, y, ypred):
        matrix = confusion_matrix(y, ypred)
        plt.figure(figsize=(10, 8))
        colors = ["orange", "green"]
        sns.heatmap(
            matrix,
            xticklabels=self.labels,
            yticklabels=self.labels,
            cmap=colors,
            annot=True,
            fmt="d",
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

    def draw_abnormal(self, y, error, threshold):
        groupSDF = pd.DataFrame({"error": error, "true": y}).groupby("true")
        figure, axes = plt.subplots(figsize=(12, 8))

        for name, group in groupSDF:
            if name == 1:
                axes.plot(
                    group.index,
                    group.error,
                    marker="o",
                    linestyle="",
                    color="g",
                    label="Abnormal",
                    alpha=0.1,
                )
            else:
                axes.plot(
                    group.index,
                    group.error,
                    marker="x",
                    linestyle="",
                    color="r",
                    label="Normal",
                )
        axes.hlines(
            threshold,
            axes.get_xlim()[0],
            axes.get_xlim()[1],
            colors="b",
            zorder=100,
            label="Threshold",
        )
        axes.legend()
        plt.title("Abnormal")
        plt.ylabel("Error")
        plt.xlabel("Data")
        plt.show()

    def draw_error(self, error, threshold):
        plt.plot(error, marker="o", ms=3.5, linestyle="", label="Point")
        plt.hlines(
            threshold,
            xmin=0,
            xmax=len(error) - 1,
            colors="b",
            zorder=100,
            label="Threshold",
        )
        plt.legend()
        plt.title("Reconstruction error")
        plt.ylabel("Error")
        plt.xlabel("Data")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("转子数据集.csv")
    x_train, x_test = train_test_split(df, train_size=0.7, random_state=1)

    x_train_data, y_train_data = make_data_labels(x_train)
    x_test_data, y_test_data = make_data_labels(x_test)

    # 加载数据
    train_loader = DataLoader(
        dataset=TensorDataset(x_train_data, y_train_data),
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(x_test_data, y_test_data),
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # 样本长度
    train_size = len(train_loader)

    encoding_dim = 1024
    input_dim = x_train_data.shape[1]
    # print(input_dim)

    # 记录训练集loss和acc变化
    train_loss = []
    train_acc = []

    # 构建模型
    model = AutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    loss_func = nn.CrossEntropyLoss(reduction="sum")

    # *******************************训练模型**************************
    num_epochs = 50
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loss_epoch = 0.
        correct_epoch = 0
        for data, label in train_loader:
            optimizer.zero_grad()
            x_recon = model(data)

            probabilities = F.softmax(x_recon, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_epoch += (predicted_labels == label).sum().item()

            loss = loss_func(x_recon, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        
        print('Epoch {}/{} : loss: {:.9f}'.format(
            epoch + 1, num_epochs, running_loss / len(train_loader)
        ))
    plt.plot(range(num_epochs), train_loss, color='b', label='train_loss')
    plt.plot(range(num_epochs), train_acc, color='g', label='train_acc')
    plt.legend()
    plt.show()  # 显示 lable

    #torch.save(model, 'final_model_ae.pt')
    """

    # ***************************************** 模型评估 *********************************
    # 加载模型
    model = torch.load("./model/AE_bear.pt")
    # 将模型设置为评估模式
    model.eval()
    # 使用测试集进行推断
    with torch.no_grad():
        correct_test = 0
        test_loss = 0

        for test_data, test_label in test_loader:
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_test += (predicted_labels == test_label).sum().item()
            loss = loss_func(test_output, test_label)
            test_loss += loss.item()

    test_accuracy = correct_test / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy: .4f} Test Loss: {test_loss: .8f}")

    # ***************************画图*************************
    # 使用测试集数据进行推断并计算每一类的分类准确率
    class_labels = []  # 存储类别标签
    predicted_labels = []  # 存储预测的标签

    with torch.no_grad():
        for test_data, test_label in test_loader:
            # 将模型设置为评估模式
            model.eval()
            test_data = test_data
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

            class_labels.extend(test_label.tolist())
            predicted_labels.extend(predicted.tolist())

    # 混淆矩阵
    confusion_mat = confusion_matrix(class_labels, predicted_labels)

    # 计算每一类的分类准确率
    report = classification_report(class_labels, predicted_labels, digits=4)
    print(report)

    # *********************绘制混淆矩阵**************************

    label_mapping = {
        0: "C1",
        1: "C2",
        2: "C3",
        3: "C4",
    }

    # 绘制混淆矩阵
    plt.figure()
    sns.heatmap(
        confusion_mat,
        xticklabels=label_mapping.values(),
        yticklabels=label_mapping.values(),
        annot=True,
        fmt="d",
        cmap="summer",
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
