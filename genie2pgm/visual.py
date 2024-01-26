"""
@Description: 概率可视化
@Author  : yxuanf
@Time    : 2023/8/31
@Site    : yxuanf@nudt.edu.cn
@File    : noisymax.py 
"""
import seaborn as sns
import matplotlib.pyplot as plt


class Visual:
    def __init__(self, state):
        self.state = state
        # seaborn样式
        sns.set(palette="muted", color_codes=True)
        # 解决Seaborn中文显示问题
        sns.set(font='Microsoft YaHei', font_scale=0.8)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        # 解决无法显示符号的问题
        plt.rcParams['axes.unicode_minus'] = False

    def show_barchart(self, query):
        print(query)
        for _, value in query.items():
            # get probability
            data = value.values
            state = self.state[value.variables[0]]
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
            plt.rcParams['font.size'] = 16
            fig.suptitle(f"{value.variables[0]}", font={'family': 'Microsoft YaHei', 'size': 16})
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            axes[0].bar(state, data, color=colors[:len(data)])
            axes[0].tick_params(axis='x', labelsize=16)
            axes[0].tick_params(axis='y', labelsize=16)
            for i, j in enumerate(data):
                axes[0].text(i, 1.01*j, str(round(j, 4)),ha='center', va='bottom')
            _, l_text, p_text = axes[1].pie(data, labels=state, autopct='%.4f%%')
            for t in p_text:
                t.set_size(16)
            for t in l_text:
                t.set_size(16)
            axes[1].legend(prop={'size': 16})
            axes[1].tick_params(axis='x', labelsize=16)
            plt.tight_layout()

