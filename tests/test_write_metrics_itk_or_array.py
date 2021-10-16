import unittest
import collections
import numpy as np
from parameterized import parameterized

import seg_metrics.seg_metrics as sg

labels = [0, 1]
pred = np.ones((10, 10, 10))
gdth = np.ones((10, 10, 10))
pred[0,0,0] = 0

test_case1 = [{'labels': labels,
              'pred': pred,
              'gdth': gdth,
'metrics': None,
              'expected': {"msd": 0,
                           "mdsd": 0,
                           "stdsd": 0,
                           "hd95": 0,
                           "hd": 0,
                           "dice": 1,
                           "jaccard": 1,
                           "precision": 1,
                           "recall": 1,
                           "fnr": 0,
                           "fpr": 0,
                           "vs": 1
                           }
              }]
test_case2 = [{'labels': labels,
              'pred': np.pad(pred, ((5, 5), (5, 5), (5, 5))),
              'gdth': np.pad(gdth, ((0, 10), (0, 10), (0, 10))),
'metrics': None,
              'expected': {"msd": 0,
                           "mdsd": 0,
                           "stdsd": 0,
                           "hd95": 0,
                           "hd": 0,
                           "dice": 1,
                           "jaccard": 1,
                           "precision": 1,
                           "recall": 1,
                           "fnr": 0,
                           "fpr": 0,
                           "vs": 1
                           }}]
test_case3 = [{'labels': labels,
              'pred': np.pad(pred, ((10, 0), (10, 0), (10, 0))),
              'gdth': np.pad(gdth, ((0, 10), (0, 10), (0, 10))),
'metrics': None,
              'expected': {"msd": 0,
                           "mdsd": 0,
                           "stdsd": 0,
                           "hd95": 0,
                           "hd": 0,
                           "dice": 0,
                           "jaccard": 0,
                           "precision": 0,
                           "recall": 0,
                           "fnr": 1,
                           "fpr": 1,
                           "vs": 0
                           }}]

class Img_itk:
    def __init__(self, img, origin, spacing, orientation):
        self.img = img
        self.origin = origin
        self.spacing = spacing
        self.orientation = orientation

class Test_seg_metrics(unittest.TestCase):

    @parameterized.expand([test_case1, test_case2, test_case3])
    def test_computeQualityMeasures(self, case):
        out = sg.write_metrics(labels=case['labels'],
                               pred_img=case['pred'],
                               gdth_img=case['gdth'],
                               csv_file=None,
                               metrics=case['metrics'])
        for o, e in zip(collections.OrderedDict(sorted(out.items())).items(),
                        collections.OrderedDict(sorted(case['expected'].items())).items()):
            print(o, e)



            # self.assertAlmostEqual(o[1], e[1] )


if __name__ == '__main__':
    unittest.main()
