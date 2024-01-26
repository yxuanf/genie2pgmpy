import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns


class LSTMclassifier(nn.Module):
    def __init__(
        self, batch_size, input_dim, hidden_layer_sizes, output_dim, dropout_rate=0.5
    ):
        """
        LSTM分类任务： params:
        batch_size:批处理大小
        input_dim:输入数据的维度
        hidden_layer_sizes:隐藏层的数目和维度
        output_dim:输出的维度
        dropout_rate:随机丢弃神经元的概率
        """
        super().__init__()
        # 批处理大小
        self.batch_size = batch_size

        # lstm层数
        self.num_layers = len(hidden_layer_sizes)
        self.lstm_layers = nn.ModuleList()  # 保存lstm层的列表

        # 定义第一层LSTM
        self.lstm_layers.append(
            nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True)
        )

        # 定义后续的LSTM
        for i in range(1, self.num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True
                )
            )

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim),
        )

    # 前向传播
    def forward(self, input_seq):
        input_seq = input_seq.view(self.batch_size, 32, 32)
        lstm_out = torch.transpose(input_seq, 1, 2)  # 反转维度，适应网络输入形状
        for m in self.lstm_layers:
            lstm_out, _ = m(lstm_out)
        out = self.classifier(lstm_out[:, -1, :])
        return out


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

    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype("int64"))

    return x_data, y_label


# def model_train(batch_size, epochs, train_loader, model):
#     # 样本长度
#     train_size = len(train_loader) * batch_size

#     # 最高准确率  最佳模型
#     best_accuracy = 0.0
#     best_model = model

#     train_loss = []  # 记录在训练集上每个epoch的loss的变化情况
#     train_acc = []  # 记录在训练集上每个epoch的准确率的变化情况

#     # 计算模型运行时间
#     start_time = time.time()
#     for epoch in range(epochs):
#         # 训练
#         model.train()

#         loss_epoch = 0.0  # 保存当前epoch的loss和
#         correct_epoch = 0  # 保存当前epoch的正确个数和
#         for seq, labels in train_loader:
#             # print(seq.size(), labels.size()) torch.Size([32, 7, 1024]) torch.Size([32])
#             # 每次更新参数前都梯度归零和初始化
#             optimizer.zero_grad()
#             # 前向传播
#             y_pred = model(seq)  # torch.Size([16, 10])
#             # 对模型输出进行softmax操作，得到概率分布
#             probabilities = F.softmax(y_pred, dim=1)
#             # 得到预测的类别
#             predicted_labels = torch.argmax(probabilities, dim=1)
#             # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
#             correct_epoch += (predicted_labels == labels).sum().item()
#             # 损失计算
#             loss = loss_function(y_pred, labels)
#             loss_epoch += loss.item()
#             # 反向传播和参数更新
#             loss.backward()
#             optimizer.step()
#         #     break
#         # break
#         # 计算准确率
#         train_Accuracy = correct_epoch / train_size
#         train_loss.append(loss_epoch / train_size)
#         train_acc.append(train_Accuracy)
#         print(
#             f"Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}"
#         )

# # 保存最后的参数
# torch.save(model, "final_model_lstm_1.pt")

# matplotlib.rc("font", family="Microsoft YaHei")
# print(f"\nDuration: {time.time() - start_time:.0f} seconds")
# plt.plot(range(epochs), train_loss, color="b", label="train_loss")
# plt.plot(range(epochs), train_acc, color="g", label="train_acc")
# plt.legend()
# plt.show()  # 显示 lable


if __name__ == "__main__":
    df = pd.read_csv("LSTM_轴承.csv")
    x_train, x_test = train_test_split(df, train_size=0.7, random_state=1)

    x_train_data, y_train_data = make_data_labels(x_train)
    x_test_data, y_test_data = make_data_labels(x_test)

    # 定义模型参数
    batch_size = 32
    input_dim = 32
    hidden_layer_sizes = [256, 128, 64]
    output_dim = 4

    # 加载数据
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_train_data, y_train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_test_data, y_test_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    # # 构建模型
    # model = LSTMclassifier(batch_size, input_dim, hidden_layer_sizes, output_dim)
    # # print(model)

    # # 定义损失函数和优化函数
    # loss_function = nn.CrossEntropyLoss(reduction="sum")
    # learn_rate = 0.0001
    # optimizer = torch.optim.Adam(model.parameters(), learn_rate)

    # # ***************************************模型训练*******************************

    # # 开始训练
    # epochs = 1000
    # model_train(batch_size, epochs, train_loader, model)

    # *********************************************模型评估************************************************

    # 模型评估
    # 加载模型
    model = torch.load("./model/LSTM_bear.pt")
    print(model)

    # 使用测试集进行推断
    with torch.no_grad():
        correct_test = 0
        test_loss = 0
        for test_data, test_label in test_loader:
            # 将模型设置为评估模式
            model.eval()

            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_test += (predicted_labels == test_label).sum().item()
            loss = loss_function(test_output, test_label)
            test_loss += loss.item()

    test_accuracy = correct_test / (len(test_loader) * batch_size)
    test_loss = test_loss / (len(test_loader) * batch_size)
    print(f"Test Accuracy: {test_accuracy: .4f} Test Loss: {test_loss: .8f}")

    # *********************************得出每一类的分类准确率************************************
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
