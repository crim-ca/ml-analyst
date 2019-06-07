import os
import sys
import unittest
from analyze import run_analysis, get_parser, args2params

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestCirculationGraph(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_minimal(self):
        input_file = "iris.csv"
        output_dir = "/tmp/test_analyze"
        args = get_parser().parse_args(
            [input_file, "-ml", "LogisticRegression,SVC", "-prep", "RobustScaler,Normalizer", "-results", output_dir])
        params = args2params(args)
        res = run_analysis(**params)
        self.assertEqual(len(res), len(params['learners']))
        self.assertEqual(res.count(0), len(res), "Not all %d runs were successful" % (len(params['learners'])))

