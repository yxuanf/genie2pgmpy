import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import time


class BearingCNN(nn.Module):
    def __init__(self, conv_archs, num_classes, batch_size, input_channels=1):
        """
        conv_archs:网络结构
        num_classes:分类类别
        batch_size:批处理个数
        input_channels:输入通道数
        """
        super(BearingCNN, self).__init__()
        self.batch_size = batch_size

        # CNN参数
        self.conv_arch = conv_archs
        self.input_channels = input_channels
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool1d(9)  # 1D池化

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, num_classes),
        )

    # CNN卷积池化结构
    def make_layers(self):
        layers = []
        for num_convs, out_channels in self.conv_arch:
            for _ in range(num_convs):
                layers.append(
                    nn.Conv1d(
                        self.input_channels, out_channels, kernel_size=3, padding=1
                    )
                )
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, input_seq):  # [32, 512]
        input_seq = input_seq.view(self.batch_size, 1, 512)
        features = self.features(input_seq)  #
        x = self.avgpool(features)  # [32, 128, 9]
        flat_tensor = x.view(self.batch_size, -1)  #
        output = self.classifier(flat_tensor)  # [32, 10]
        return output


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


if __name__ == "__main__":
    df = pd.read_csv("CNN_轴承.csv")
    x_train, x_test = train_test_split(df, train_size=0.7, random_state=1)

    x_train_data, y_train_data = make_data_labels(x_train)
    x_test_data, y_test_data = make_data_labels(x_test)

    # 定义模型
    batch_size = 32

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

    #模型结构
    conv_arch = ((2, 32), (1, 64), (1, 128))  # 卷积结构
    input_channels = 1  # 输入通道
    num_classes = 4  # 类别
    model = BearingCNN(conv_arch, num_classes, batch_size)

    # 定义损失函数和优化函数,优化器
    device = torch.device("cuda")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    learn_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)

    # 样本长度
    train_size = len(train_loader) * batch_size
    test_size = len(test_loader) * batch_size

    # 最高准确率，最佳模型
    best_accuracy = 0.0
    best_model = model

    # 记录训练集loss和acc变化
    train_loss = []
    train_acc = []

    # 记录训练时间
    # start_time = time.time()
    # ***************************************模型训练*******************************

    #开始训练
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loss_epoch = 0.
        correct_epoch = 0
        for data, label in train_loader:
            optimizer.zero_grad()
            #前向传播
            outputs = model(data)

            probabilities = F.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_epoch += (predicted_labels == label).sum().item()

            loss = loss_function(outputs, label)
            loss_epoch += loss.item()

            #后向传播和优化

            loss.backward()
            optimizer.step()

            #计算损失
            running_loss += loss.item()

        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)

        print(f'Epoch[{epoch+1} / {epochs}], Loss: {running_loss / len(train_loader):.4f}')

    #保存模型
    torch.save(model, 'final_model_cnn1d_1.pt')
    plt.plot(range(epochs), train_loss, color='b', label='train_loss')
    plt.plot(range(epochs), train_acc, color='g', label='train_acc')
    plt.legend()
    plt.show()  # 显示 lable

    # *********************************************模型评估************************************************

    # 模型评估
    # 加载模型
    model = torch.load("./model/CNN_bear.pt")
    print(model)
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
            loss = loss_function(test_output, test_label)
            test_loss += loss.item()

    test_accuracy = correct_test / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy: .4f} Test Loss: {test_loss: .8f}")

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
        4: "C5",
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
