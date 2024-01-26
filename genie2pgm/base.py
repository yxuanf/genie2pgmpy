"""
@Description: 基本操作（提取结构,状态名称等）
@Author  : yxuanf
@Time    : 2023/8/30
@Site    : yxuanf@nudt.edu.cn
@File    : genie2pgm.py 
"""
import xml.etree.ElementTree as ET
from pgmpy.models import BayesianNetwork


class BasicOperation:
    def __init__(self, xmlpath):
        self.path = xmlpath
        self.tree = ET.parse(xmlpath)
        self.root = self.tree.getroot()
        self.nodes = self.root.find('nodes')
        self.extensions  = self.root.find('extensions')

    # Get BayesianNetwork structure
    def GetStructure(self):
        structure = []
        # Find 'nodes' in Xml
        for node in self.nodes:
            variable = node.attrib.get('id')
            if node.findall("parents"):
                evidence = node.findall('parents')[0].text.split()
                for parent in evidence:
                    structure.append((parent, variable))
        model = BayesianNetwork(structure)
        return model


    # Get state_names and Card_Num
    def GetState(self):
        state_names, evidence_card = {}, {}
        # Find node that labled "nodes"
        for node in self.nodes:
            # The id of node
            variable = node.attrib.get('id')
            # The num of state
            variable_card = len(node.findall('state'))
            evidence_card[variable] = variable_card
            state_name = []
            # 
            for name in node.findall("state"):
                state_name.append(name.attrib.get('id'))
            state_names[variable] = state_name
        return state_names, evidence_card
