import unittest
from copy import deepcopy
import numpy as np
from rus_comsol.rus_rpr import RUSRPR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

class TestRUSRPR(unittest.TestCase):

    order = 12
    nb_freq = 10
    mass = 0.002132e-3 # mass in kg
    dimensions = np.array([1560e-6, 1033e-6, 210e-6])

    # initial elastic constants in Pa
    cij_dict = {"c11": 231e9,
                "c12": 132e9,
                "c13": 71e9,
                "c33": 186e9,
                "c44": 49e9,
                "c66": 95e9,
                }

    def test_resonances(self):
        rus = RUSRPR(TestRUSRPR.cij_dict, TestRUSRPR.mass,
                     TestRUSRPR.dimensions, TestRUSRPR.order, TestRUSRPR.nb_freq)
        rus.initialize()
        resonances = np.round(rus.compute_resonances()*1e-6, 8)
        resonances_ref = np.array([0.40762294, 0.4490465, 0.92114884,
                                   0.95872178, 0.99171676, 1.19959051,
                                   1.2624155, 1.45997466, 1.47416173, 1.69389203])
        self.assertEqual(resonances[0], resonances_ref[0])
        self.assertEqual(resonances[1], resonances_ref[1])
        self.assertEqual(resonances[2], resonances_ref[2])
        self.assertEqual(resonances[3], resonances_ref[3])
        self.assertEqual(resonances[-1], resonances_ref[-1])


if __name__ == '__main__':
    unittest.main()