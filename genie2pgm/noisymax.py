"""
@Description: Noisymax model
@Author  : yxuanf
@Time    : 2023/8/30
@Site    : yxuanf@nudt.edu.cn
@File    : noisymax.py 
@note    : noisymax节点的所有父节点的状态数必须一致,否则无法运行！
"""
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from genie2pgm.simplemodel import SimpleDiscreteModel
np.seterr(divide='ignore', invalid='ignore')


class NoisyMax(SimpleDiscreteModel):
    def __init__(self, xmlpath):
        super(NoisyMax, self).__init__(xmlpath)
        self.cpd = self.getcpd()

    # Convert decimal to any base
    def decimal2base(self, decimal, base, num):
        if decimal == 0:
            return '0'.zfill(num)
        digits = []
        while decimal > 0:
            remainder = decimal % base
            digits.append(str(remainder))
            decimal = decimal // base
        digits.reverse()
        return ''.join(digits).zfill(num)

    # list -> matrix
    def list2matrix(self, nest_list):
        if isinstance(nest_list, list):
            return np.array([self.list2matrix(a) for a in nest_list], dtype=np.float32)
        else:
            return nest_list

    # Probability table processing
    def prob_tabel_process(self, param, parent_num, p_state, node_state):
        copy = np.zeros((parent_num, p_state, node_state))
        for i in range(node_state):
            # status 父节点状态组合
            for j in range(p_state**parent_num):
                # p_num 父节点个数
                for k in range(parent_num):
                    copy[k][int(self.decimal2base(j, p_state, parent_num)[k])][i] = param[k][int(self.decimal2base(
                        j, p_state, parent_num)[k])][i]/sum(param[k][int(self.decimal2base(j, p_state, parent_num)[k])][i:])
        return copy

    def calculate_cpd(self, param, leak, state_num, p_state, parent_num):
        tabel = np.zeros((state_num, p_state**parent_num))
        for i in range(state_num - 1):
            for j in range(p_state**parent_num):
                prob = 1
                for k in range(parent_num):
                    prob = prob * \
                        (1 - param[k][int(self.decimal2base(j,
                         p_state, parent_num)[k])][i])
                prob = 1 - prob * (1 - leak[i])
                if i == 0:
                    tabel[i][j] = prob
                else:
                    tabel[i][j] = (1 - sum(tabel[:i, j]))*prob

        for m in range(len(tabel[-1])):
            prob = 0
            tabel[-1][m] = 1 - sum(tabel[:-1, m])
        return tabel

    def noisymax2cpd(self, node):
        # Probability Table Parameters
        param = list(map(float, node.find("parameters").text.split()))
        # The num of parents nodes
        parent_num = len(node.find("parents").text.split())
        # The num of the nodes state
        state_num = self.evidence_card[node.attrib.get("id")]
        # The num of the parent nodes state
        p_state_num = self.evidence_card[node.find("parents").text.split()[0]]
        # prob tabel -> matrix
        param = self.list2matrix([param[i: i + state_num]
                                 for i in range(0, len(param), state_num)])
        # leaked parameters
        leak = param[-1]
        # delete leak from prob tabel
        param = np.delete(param, -1, axis=0).reshape(parent_num,
                                                     p_state_num, state_num)
        # Noisymax probaility process
        for i in range(len(leak)):
            leak[i] = leak[i]/sum(leak[i:])
        param = np.nan_to_num(self.prob_tabel_process(
            param, parent_num, p_state_num, state_num))
        tabel = self.calculate_cpd(
            param, leak, state_num, p_state_num, parent_num)
        return tabel.tolist()

    def get_nosiy_message(self):
        TotalCPD = []
        for node in self.nodes.findall("noisymax"):
            tabel, name, card = {}, {}, []
            # Get id
            variable = node.attrib.get('id')
            # Get the state number of the parent node
            variable_card = self.evidence_card[variable]
            # Find all parents
            evidence = node.find('parents').text.split()
            name[variable] = self.state_names[variable]
            for evid in evidence:
                # Get evidence_cards
                card.append(self.evidence_card[evid])
                # Get state_names
                name[evid] = self.state_names[evid]
            tabel['variable'] = variable
            tabel['variable_card'] = variable_card
            tabel['state_names'] = name
            tabel['evidence'] = evidence
            tabel['evidence_card'] = card
            tabel['values'] = self.noisymax2cpd(node)
            TotalCPD.append(tabel)
        CPD = []
        for i in TotalCPD:
            cpd = TabularCPD(variable=i['variable'],
                             variable_card=i['variable_card'],
                             values=i['values'],
                             evidence=i["evidence"],
                             evidence_card=i['evidence_card'],
                             state_names=i['state_names'])
            CPD.append(cpd)
        return CPD

    # add cpd
    def add_noisymax_cpd(self):
        noisymax_cpd = self.get_nosiy_message()
        for i in noisymax_cpd:
            self.cpd.append(i)
        self.add_cpd(model=self.model, CPDs=self.cpd)
