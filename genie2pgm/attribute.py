"""
@Description: Get nodes Arribution
@Author  : yxuanf
@Time    : 2023/9/1
@Site    : yxuanf@nudt.edu.cn
@File    : attribute.py 
"""
import os
import json
from genie2pgm.simplemodel import SimpleDiscreteModel


class Attributes(SimpleDiscreteModel):
    def __init__(self, xmlpath):
        super(Attributes, self).__init__(xmlpath)
        self.filename = os.path.splitext(self.path)[0].split("/")[-1]
        self.genie = self.extensions.find('genie')
        self.nodes = self.genie.findall('node')

    def getInformation(self):
        information = []
        for nodes in self.nodes:
            temp = {}
            node = nodes.find('name')
            id = nodes.attrib.get("id")
            name = node.text
            temp['id'] = id
            temp['name'] = name
            temp['state'] = self.state_names[id]
            information.append(temp)
        with open(f"./information/{self.filename}.json", "w", encoding='utf-8') as js:
            json.dump(information, js, ensure_ascii=False)
