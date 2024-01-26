"""
@Description: Genie xdsl解析
@Author  : yxuanf
@Time    : 2023/8/24
@Site    : yxuanf@nudt.edu.cn
@File    : genie2pgm.py 
"""
from genie2pgm.base import BasicOperation
from pgmpy.factors.discrete import TabularCPD


# For the basic model
class SimpleDiscreteModel(BasicOperation):
    def __init__(self, xmlpath):
        super(SimpleDiscreteModel, self).__init__(xmlpath=xmlpath)
        self.model = self.GetStructure()
        self.state_names, self.evidence_card = self.GetState()

    # Get CPD
    def getcpd(self):
        TotalCPD = []
        #  Get the state name of the node and the number of its parent nodes
        for node in self.nodes:
            """
            cpd : Store the information of the CPT table in a dictionary
            name: store the id of the node and the parent node in a dictionary
            vaules: store the conditional probability table in a list
            """
            cpd, name, values = {}, {}, []
            # get node id
            variable = node.attrib.get("id")
            # Get the state number of the parent node
            variable_card = self.evidence_card[variable]
            value_dict = {}

            # if the node type is deterministic
            if node.tag == "deterministic":
                # Obtain deterministic results for this type of node（list[str]）
                prob = node.find("resultingstates").text.split()
                i = 0
                # Store the state of the node as a dictionary
                for value in node.findall("state"):
                    value_dict[value.attrib.get("id")] = i
                    i += 1
            elif node.tag == "cpt":
                # Obtain conditional probability and complete type conversion（str -> float）
                prob = list(map(float, node.find("probabilities").text.split()))
            else:
                # pass if not these nodes
                continue

            length = len(prob)
            name[variable] = self.state_names[variable]

            # if has parent node
            if node.findall("parents"):
                # Find all parents
                evidence = node.findall("parents")[0].text.split()
                cpd["evidence"] = evidence
                # the num of parent node state
                card = []
                for evid in evidence:
                    # Get evidence_cards
                    card.append(self.evidence_card[evid])
                    # Get state_names
                    name[evid] = self.state_names[evid]
                cpd["evidence_card"] = card

                # Probability table processing
                # if not deterministic node
                if not value_dict:
                    for i in range(int(variable_card)):
                        values.append(prob[i : length : int(variable_card)])
                else:
                    # Initialize the probability table to 0
                    """The number of lists is the number of node states,
                    and the number of elements in the list is the number of permutations of parent node states
                    """
                    values = [[0] * len(prob) for _ in range(len(value_dict))]
                    j = 0
                    for i in prob:
                        value = value_dict[i]
                        values[value][j] = 1
                        j += 1
            else:
                if not value_dict:
                    values = [[value] for value in prob]
                else:
                    values = [[0] for _ in range(len(value_dict))]
                    values[value_dict[prob[0]]][0] = 1
            cpd["variable"] = variable
            cpd["variable_card"] = variable_card
            cpd["state_names"] = name
            cpd["values"] = values
            TotalCPD.append(cpd)
        # Store the CPT table of different variables into list
        CPD = []
        for i in TotalCPD:
            if "evidence" in list(i.keys()):
                cpd = TabularCPD(
                    variable=i["variable"],
                    variable_card=i["variable_card"],
                    values=i["values"],
                    evidence=i["evidence"],
                    evidence_card=i["evidence_card"],
                    state_names=i["state_names"],
                )
            else:
                cpd = TabularCPD(
                    variable=i["variable"],
                    variable_card=i["variable_card"],
                    values=i["values"],
                    state_names=i["state_names"],
                )
            CPD.append(cpd)
        return CPD


# Add cpd tabel to Bayesian Networks
def add_cpd(model, CPDs):
    for cpd in CPDs:
        model.add_cpds(cpd)
    return model
