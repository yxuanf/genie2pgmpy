import unittest

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy import config
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TestEM(unittest.TestCase):
    def setUp(self):
        self.model1 = get_example_model("cancer")
        self.data1 = self.model1.simulate(int(1e4), seed=42)

        self.model2 = BayesianNetwork(self.model1.edges(), latents={"Smoker"})
        self.model2.add_cpds(*self.model1.cpds)
        self.data2 = self.model2.simulate(int(1e4), seed=42)

    def test_get_parameters(self):
        ## All observed
        est = EM(self.model1, self.data1)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        ## Latent variables
        est = EM(self.model2, self.data2)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

    def tearDown(self):
        del self.model1
        del self.model2
        del self.data1
        del self.data2

        get_reusable_executor().shutdown(wait=True)


class TestEMTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.model1 = get_example_model("cancer")
        self.data1 = self.model1.simulate(int(1e4), seed=42)

        self.model2 = BayesianNetwork(self.model1.edges(), latents={"Smoker"})
        self.model2.add_cpds(*self.model1.cpds)
        self.data2 = self.model2.simulate(int(1e4), seed=42)

    def test_get_parameters(self):
        est = EM(self.model1, self.data1)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        est = EM(self.model2, self.data2)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

    def tearDown(self):
        del self.model1
        del self.model2
        del self.data1
        del self.data2

        get_reusable_executor().shutdown(wait=True)

        config.set_backend("numpy")
