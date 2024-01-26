"""
@Description: query
@Author  : yxuanf
@Time    : 2023/8/31
@Site    : yxuanf@nudt.edu.cn
@File    : query.py 
"""
from pgmpy.inference import VariableElimination
from pgmpy.inference import ApproxInference

class Query:
    def __init__(self, model):
        self.model = model


    # Doing Inference using hard or virtual evidence
    def exact_query(self, variable, evidence, use_joint=True, virtual_evidence=None):
        # 采用变量消除法
        infer = VariableElimination(model = self.model)
        q = infer.query(variables=variable, evidence=evidence, joint=use_joint)
        if virtual_evidence is not None:
            q = infer.query(variables=variable,
                            evidence=evidence,
                            joint=use_joint,
                            virtual_evidence=virtual_evidence)
        # print(q)
        # if not use_joint:
        #     for factor in q.values():
        #         print(factor)
        return q

    # Compute MAP question
    def get_map(self, variable, evidence, virtual_evidence=None):
        infer = VariableElimination(model=self.model)
        q = infer.map_query(variables=variable,
                            evidence=evidence,
                            virtual_evidence=virtual_evidence)
        # print(q)
        return q

    # Approximate inference based on sampling
    def approximate_query(self, variables, n_samples=int(1e4), evidence=None,
                     virtual_evidence=None, use_joint=False, show_progress=True, seed=None):
        infer = ApproxInference(self.model)
        q = infer.query(variables, n_samples, evidence,
                        virtual_evidence, use_joint, show_progress)
        # print(q)
        return q

    # Get data distribution
    def get_distribution(self, samples, variable=None):
        """
        For marginal distribution, P(A): get_distribution(samples, variables=['A'])
        For joint distribution, P(A, B): get_distribution(samples, variables=['A', 'B'])
        """
        if variable is None:
            raise ValueError("variables must be specified")

        return samples.groupby(variable).size() / samples.shape[0]
