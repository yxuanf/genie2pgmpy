import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from Ui_GUI import Ui_Form
from genie2pgm.simplemodel import SimpleDiscreteModel, add_cpd
from neuralNet.AE_roll import AutoEncoder
from neuralNet.BNN_roll import BNNRoll
from neuralNet.CNN import BearingCNN
from neuralNet.LSTM import LSTMclassifier
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import OrderedDict

random.seed(0)


class Fusion(QWidget, Ui_Form):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        # 观测值
        self.observation = dict()
        # 监测值
        self.monitor = OrderedDict()
        self.virtual = list()
        self.variable = None
        self.canvas = None
        self.bear_prob = None
        self.bear_prob = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.roll = {"转子": "高压涡轮轴8_2_1"}
        self.bear = {"轴承": "前后支撑滚珠轴承6_3_3_3"}
        self.roll_mapping = {0: "正常", 1: "碰撞摩擦", 2: "不平衡", 3: "不对中"}
        self.bear_mapping = {0: "正常", 1: "滚珠故障", 2: "内圈故障", 3: "外圈故障"}
        self.roll_state = dict()
        self.bear_state = dict()
        self.fusionNode = ["高压涡轮轴8_2_1", "前后支撑滚珠轴承6_3_3_3"]
        self.BNNresult = {
            0: [1, 0, 0, 0],
            1: [0, 6 / 12, 2 / 12, 4 / 12],
            2: [0, 0, 1, 0],
            3: [1 / 16, 2 / 16, 0, 13 / 16],
        }
        self.AEresult = {
            0: [1, 0, 0, 0],
            1: [1 / 12, 5 / 12, 0, 6 / 12],
            2: [0, 0, 1, 0],
            3: [2 / 10, 1 / 10, 0, 7 / 10],
        }
        self.LSTMresult = {
            0: [53 / 67, 14 / 67, 0, 0],
            1: [12 / 57, 39 / 57, 6 / 57, 0],
            2: [0, 3 / 70, 57 / 70, 10 / 70],
            3: [0, 0, 9 / 62, 53 / 62],
        }
        self.CNNresult = {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 121 / 131, 10 / 131],
            3: [0, 0, 0, 1],
        }
        self.loadBN()
        self.NNSelectMenu()
        self.LoadmonitorData.clicked.connect(self.LoadData)
        self.clearEvidence.clicked.connect(self.clearEvidenceAction)
        self.queryCombobox.activated.connect(self.queryCombobox_selected)
        self.inferbutton.clicked.connect(self.query_button_clicked)
        self.roll_state[self.fusionNode[0]] = self.statename[self.fusionNode[0]]
        self.bear_state[self.fusionNode[1]] = self.statename[self.fusionNode[1]]

    def getBayesianModel(self, path):
        Model = SimpleDiscreteModel(xmlpath=path)
        return (
            add_cpd(model=Model.model, CPDs=Model.getcpd()),
            Model.state_names,
            Model.evidence_card,
        )

    # 导入贝叶斯模型
    def loadBN(self):
        self.BN_Path, _ = QFileDialog.getOpenFileName(
            self, "读取xml文件", "./model", "(*.xml *.xdsl)"
        )
        try:
            self.bnModel, self.statename, self.cards = self.getBayesianModel(
                self.BN_Path
            )
            if self.bnModel.check_model():
                QMessageBox.information(QWidget(), "导入成功", "贝叶斯网络导入成功")
                self.querynodes = list(self.bnModel.nodes)[:-39]
                self.queryCombobox.clear()
                self.queryCombobox.addItems(self.querynodes)
                # 默认首选第一个
                self.queryCombobox.setCurrentIndex(0)
                self.LoadObservation()
            else:
                QMessageBox.critical(QWidget(), "错误！", "无法正确导入模型的先验概率!,请检查导入模型的正确性！")
                return
        except:
            QMessageBox.critical(QWidget(), "错误！", "无法导入贝叶斯模型")

    def NNSelectMenu(self):
        self.NNselectMeau = QMenu(self.NeuralButton)
        self.action1 = QAction("转子嵌入神经网络", self.NNselectMeau)
        self.action2 = QAction("轴承嵌入神经网络", self.NNselectMeau)
        self.NNselectMeau.addAction(self.action1)
        self.NNselectMeau.addAction(self.action2)
        self.NNselectMeau.setStyleSheet(
            "QMenu {background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);selection-color: rgb(255, 0, 0);}"
        )
        self.NeuralButton.setMenu(self.NNselectMeau)
        self.action1.triggered.connect(lambda _: self.fusionRollNet())
        self.action2.triggered.connect(lambda _: self.fusionBearNet())

    def LoadData(self):
        rollindex = random.randint(0, len(self.rollData) - 1)
        bearindex = random.randint(0, len(self.bearData) - 1)
        rolldata = (
            torch.from_numpy(self.rollData.iloc[:, :-1].values[rollindex])
            .float()
            .view(-1, 2048)
            .to(self.device)
        )
        if isinstance(self.rollNet, BNNRoll):
            _, roll_predict = self.rollNet(rolldata)
            roll_predict = roll_predict.cpu().item()
        else:
            roll_predict = (
                torch.argmax(F.softmax(self.rollNet(rolldata), dim=1), dim=1)
                .cpu()
                .item()
            )
        roll_prob = np.array(self.BNNresult[roll_predict])
        self.monitor[self.roll["转子"]] = roll_prob
        roll_value = np.reshape(roll_prob, (roll_prob.shape[0], 1))
        roll_cpt = TabularCPD(
            variable=self.roll["转子"],
            variable_card=self.cards[self.roll["转子"]],
            values=roll_value,
            state_names=self.roll_state,
        )
        self.virtual.append(roll_cpt)

        if isinstance(self.bearNet, BearingCNN):
            beardata = (
                torch.from_numpy(self.bearData.iloc[:, :-1].values[bearindex])
                .float()
                .view(-1, 512)
                .to(self.device)
            )
            _, bear_predict = self.bearNet(beardata)
            bear_predict = bear_predict.cpu().item()
            bear_prob = np.array(self.CNNresult[bear_predict])
        else:
            beardata = (
                torch.from_numpy(self.bearData.iloc[:, :-1].values[bearindex])
                .float()
                .view(-1, 1024)
                .to(self.device)
            )
            _, bear_predict = self.bearNet(beardata)
            bear_predict = bear_predict.cpu().item()
            bear_prob = np.array(self.LSTMresult[bear_predict])
        self.monitor[self.bear["轴承"]] = bear_prob
        bear_value = np.reshape(bear_prob, (bear_prob.shape[0], 1))
        bear_cpt = TabularCPD(
            variable=self.bear["轴承"],
            variable_card=self.cards[self.bear["轴承"]],
            values=bear_value,
            state_names=self.bear_state,
        )
        self.virtual.append(bear_cpt)

        # 显示
        self.EvidenceBrowser.clear()
        for name, values in self.monitor.items():
            self.EvidenceBrowser.append(f"{name}的预测结果为")
            for index, value in enumerate(values):
                if name == self.fusionNode[0]:
                    text = f"{self.roll_mapping[index]}:{value:.5f}"
                else:
                    text = f"{self.bear_mapping[index]}:{value:.5f}"
                self.EvidenceBrowser.append(text)

    # 观测数据设置
    def LoadObservation(self):
        self.observedMenu = QMenu(self.ObserveData)
        nodes = list(self.bnModel.nodes)[-39:]
        for node in nodes:
            subMeau = QMenu(node, self.observedMenu)
            for state in self.statename[node]:
                action = QAction(state, subMeau)
                action.setCheckable(True)
                action.triggered.connect(
                    lambda _, node=node, state=state: self.add_observedState_action(
                        node, state
                    )
                )
                subMeau.addAction(action)
            self.observedMenu.addMenu(subMeau)
        self.observedMenu.setStyleSheet(
            "QMenu {background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);selection-color: rgb(255, 0, 0);}"
        )
        self.ObserveData.setMenu(self.observedMenu)

    # 融合转子节点
    def fusionRollNet(self):
        QMessageBox.warning(QWidget(), "温馨提示", "选择需要与转子融合的神经网络模型")
        self.rollNN_Path, _ = QFileDialog.getOpenFileName(
            self, "导入转子模型", "./model/转子模型", "(*.pt)"
        )
        self.rollNet = torch.load(self.rollNN_Path).to(self.device)
        self.rollData = pd.read_csv("./转子数据集.csv", index_col=0)

    # 融合轴承节点
    def fusionBearNet(self):
        QMessageBox.warning(QWidget(), "温馨提示", "选择需要与轴承融合的神经网络模型")
        self.bearNN_Path, _ = QFileDialog.getOpenFileName(
            self, "导入轴承模型", "./model/轴承模型", "(*.pt)"
        )
        self.bearNet = torch.load(self.bearNN_Path).to(self.device)
        if isinstance(self.bearNet, BearingCNN):
            self.bearData = pd.read_csv("./CNN_轴承.csv", index_col=0)
        else:
            self.bearData = pd.read_csv("./LSTM_轴承.csv", index_col=0)

    # 处理观测值数据
    def add_observedState_action(self, node, state):
        self.observation[node] = state
        text = f"{node}---------------------->{state}"
        self.EvidenceBrowser.append(text)

    # 清除证据
    def clearEvidenceAction(self):
        self.observation = dict()
        self.virtual = list()
        self.EvidenceBrowser.clear()
        self.LoadObservation()

    # 选择查询变量
    def queryCombobox_selected(self):
        self.variable = [self.queryCombobox.currentText()]
        print(self.variable)

    # 融合推理
    def query_button_clicked(self):
        try:
            if self.canvas != None:
                self.gridLayout.removeWidget(self.canvas)
            self.infer = VariableElimination(model=self.bnModel)
            self.query = self.infer.query(
                variables=self.variable,
                evidence=self.observation,
                virtual_evidence=self.virtual,
            )
            self.showpic()
            print(self.query)
        except TypeError:
            QMessageBox.critical(QWidget(), "Error", "请选择需要查询的节点！")

    # 画图
    def showpic(self):
        # seaborn样式
        sns.set(palette="muted", color_codes=True)
        # 解决Seaborn中文显示问题
        sns.set(font="Microsoft YaHei", font_scale=0.8)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
        # 解决无法显示符号的问题
        plt.rcParams["axes.unicode_minus"] = False
        # 获取变量名称
        variable = self.query.variables[0]
        if variable not in self.fusionNode or not self.monitor:
            data = self.query.values
        else:
            data = self.monitor[variable]
        state = self.statename[variable]
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))
        plt.rcParams["font.size"] = 12
        self.fig.suptitle(f"{variable}", font={"family": "Microsoft YaHei", "size": 12})
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.axes[0].bar(state, data, color=colors[: len(data)])
        self.axes[0].tick_params(axis="x", labelsize=12)
        self.axes[0].tick_params(axis="y", labelsize=12)
        for i, j in enumerate(data):
            self.axes[0].text(i, 1.01 * j, str(round(j, 4)), ha="center", va="bottom")
        self.axes[1].pie(data, labels=[""] * len(state), autopct=None, shadow=True)
        self.axes[1].get_xaxis().set_visible(False)
        self.axes[1].get_yaxis().set_visible(False)
        self.axes[1].legend(state, prop={"size": 12})
        plt.tight_layout()
        self.canvas = FigureCanvas(self.fig)
        self.gridLayout.addWidget(self.canvas)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    msg_box = QMessageBox.warning(QWidget(), "温馨提示", "请导入贝叶斯网络")
    myWin = Fusion()
    myWin.setWindowTitle("贝叶斯融合工具")
    myWin.show()
    sys.exit(app.exec_())
