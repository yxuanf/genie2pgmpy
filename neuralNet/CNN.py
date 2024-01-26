import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import warnings
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BearingCNN(nn.Module):
    def __init__(self, in_features, hidden_size, out_class):
        super(BearingCNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), out_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x, torch.argmax(x, dim=1)


def train_loop(model: BearingCNN, dataloader, loss_fun, optimizer, epochs):
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
    fileName = "CNN_轴承.csv"
    train_dataset, test_dataset = dataset(fileName)
    # 构建训练集
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    net = BearingCNN(512, 256, 4)
    # 设置训练参数
    epochs = 1000
    batch_size = 32
    learning_rateAdam = 3.5e-4
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rateAdam, weight_decay=0.002
    )
    train_loss_List, train_accuracy_List = train_loop(
        net, train_loader, criterion, optimizer, epochs
    )

    net.eval()
    # 使用测试集进行推断
    with torch.no_grad():
        loss_function = nn.CrossEntropyLoss(reduction="sum")
        correct_test = 0
        test_loss = 0
        for test_data, test_label in test_loader:
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_output, predicted_labels = net(test_data)
            count = (predicted_labels == test_label).cpu().sum().item()
            correct_test += count
            loss = loss_function(test_output, test_label)
            test_loss += loss.item()

    test_accuracy = correct_test / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy: .4f} Test Loss: {test_loss: .8f}")
    if test_accuracy > 0.6:
        torch.save(net, "./model/转子模型/CNN_bear.pt")
    f, axe = plt.subplots(1, 2)
    axe[0].plot(
        range(len(train_loss_List)),
        train_loss_List,
        color="green",
        label="training loss",
    )
    axe[0].set_xlabel("epoch")
    axe[0].set_ylabel("training loss")
    axe[0].set_title("CNN模型训练集损失", font={"family": "SimHei", "size": 16})

    axe[1].plot(
        range(len(train_accuracy_List)),
        train_accuracy_List,
        color="red",
        label="training accuracy",
    )
    axe[1].set_xlabel("epoch")
    axe[1].set_ylabel("training accuracy")
    axe[1].set_title("CNN模型训练集精确度", font={"family": "SimHei", "size": 16})
    plt.tight_layout()
    plt.show()
