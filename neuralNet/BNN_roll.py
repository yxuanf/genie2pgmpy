import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BNNRoll(nn.Module):
    def __init__(self, in_features, hidden_size, out_class):
        super(BNNRoll, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 4))
        self.fc3 = nn.Linear(int(hidden_size / 4), out_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x, torch.argmax(x, dim=1)


def train_loop(model: BNNRoll, dataloader, loss_fun, optimizer, epochs):
    model.train()
    model.to(device)
    lossList = []
    accuracyList = []
    for _ in tqdm(range(epochs)):
        losses = 0
        correct_train = 0
        accuracy_train = 0
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output, labels = model(x.to(torch.float32))
            count = (labels == y).cpu().sum().item()
            correct_train += count
            loss = loss_fun(output, y.long())
            loss.backward()
            optimizer.step()
            losses += loss.item()
        accuracy_train = correct_train / len(dataloader.dataset)
        # 每个epoch的loss
        lossList.append(losses)
        accuracyList.append(accuracy_train)

    return lossList, accuracyList


def dataset(filepath):
    df0 = pd.read_csv(filepath, index_col=0)
    feature = df0.iloc[:, :-1].values
    label = df0.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(
        feature, label, train_size=0.7, random_state=0
    )
    x_train, x_test = (
        torch.from_numpy(x_train).float(),
        torch.from_numpy(x_test).float(),
    )
    y_train, y_test = (
        torch.from_numpy(y_train),
        torch.from_numpy(y_test),
    )
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return train_dataset, test_dataset


if __name__ == "__main__":
    fileName = "./转子数据集.csv"
    train_dataset, test_dataset = dataset(fileName)

    # 构建训练集
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    # net = BNNRoll(2048, 1024, 4).to(device)
    # 设置训练参数
    epochs = 1000
    learning_rateAdam = 3e-4
    criterion = nn.CrossEntropyLoss(reduction="mean")
    # optimizer = torch.optim.Adam(
    #     net.parameters(), lr=learning_rateAdam, weight_decay=0.035
    # )
    # train_loss_List, train_accuracy_List = train_loop(
    #     net, train_loader, criterion, optimizer, epochs
    # )
    net = torch.load("./model/转子模型/BNN_roll.pt")
    net.eval()
    # *******************使用测试集进行推断*******************
    label_class = []
    predicted_label = []
    with torch.no_grad():
        loss_function = nn.CrossEntropyLoss(reduction="sum")
        correct_test = 0
        test_loss = 0
        for test_data, test_label in test_loader:
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_output, predicted_labels = net(test_data)
            # 判断准确率
            count = (predicted_labels == test_label).cpu().sum().item()
            correct_test += count
            loss = loss_function(test_output, test_label)
            test_loss += loss.item()

            label_class.extend(test_label.tolist())
            predicted_label.extend(predicted_labels.tolist())

    test_accuracy = correct_test / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy: .4f} Test Loss: {test_loss: .8f}")

    # # *********************保存模型**************************#
    # if test_accuracy > 0.6:
    #     #     torch.save(net.state_dict(), "./model/bp_roll.pth")
    #     torch.save(net, "./model/转子模型/BNN_roll.pt")
    # # *********************绘制训练损失**************************#
    # f, axe = plt.subplots(1, 2)
    # axe[0].plot(
    #     range(len(train_loss_List)),
    #     train_loss_List,
    #     color="green",
    #     label="training loss",
    # )
    # axe[0].set_xlabel("epoch")
    # axe[0].set_ylabel("training loss")
    # axe[0].set_title("BNN模型训练集损失", font={"family": "SimHei", "size": 16})

    # axe[1].plot(
    #     range(len(train_accuracy_List)),
    #     train_accuracy_List,
    #     color="red",
    #     label="training accuracy",
    # )
    # axe[1].set_xlabel("epoch")
    # axe[1].set_ylabel("training accuracy")
    # axe[1].set_title("BNN模型训练集精确度", font={"family": "SimHei", "size": 16})
    # plt.tight_layout()

    # # *********************绘制混淆矩阵**************************#
    # confusion_mat = confusion_matrix(label_class, predicted_label)
    # # 计算每一类的分类准确率
    # report = classification_report(
    #     label_class, predicted_label, digits=4, zero_division=1
    # )
    # print(report)

    # label_mapping = {
    #     0: "C1",
    #     1: "C2",
    #     2: "C3",
    # }

    # _, axe = plt.subplots(figsize=(9, 6))
    # sns.heatmap(
    #     confusion_mat,
    #     xticklabels=label_mapping.values(),
    #     yticklabels=label_mapping.values(),
    #     annot=True,
    #     fmt="d",
    #     cmap=sns.color_palette("Blues", as_cmap=True),
    # )
    # axe.set_xlabel("Predicted Labels", fontsize=12)
    # axe.set_ylabel("True Labels", fontsize=12)
    # plt.title("Confusion Matrix", font={"family": "Arial", "size": 16})
    # plt.show()
