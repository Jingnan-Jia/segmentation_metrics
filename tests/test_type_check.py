import glob
import importlib
import os
import unittest
import SimpleITK as sitk
import numpy as np
import sys
import seg_metrics.seg_metrics as sg
from parameterized import parameterized
import seg_metrics.seg_metrics as sg
import tempfile

# correct cases
TEST_CASE1 = [{
"gdth_path": '/data/gdth.mha',
"pred_path": '/data/pred.mha'
}]

TEST_CASE2 = [{
"gdth_path": ['/data/gdth.mha', 'aa'],
"pred_path": ['/data/pred.mha', 'aa']
}]

TEST_CASE3 = [{
"gdth_path": None,
"pred_path": None
}]

# wrong cased


class Test_seg_metrics(unittest.TestCase):
    @parameterized.expand([TEST_CASE1, TEST_CASE2, TEST_CASE3])
    def test_type_check(self, path):
        
        self.assertIsNone(sg.type_check(gdth_path=path['gdth_path'], pred_path=path['pred_path']))
        


if __name__ == '__main__':
    unittest.main()