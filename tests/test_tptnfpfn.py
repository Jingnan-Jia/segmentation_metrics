import unittest
import collections
import numpy as np
from parameterized import parameterized
import SimpleITK as sitk
import seg_metrics.seg_metrics as sg

pred = np.ones((10, 10, 10))
gdth = np.ones((10, 10, 10))
pred[0,0,0] = 0


test_case0 = [{'labels': [1],
               'pred':  pred,
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
                            "vs": 1,
                            "TP": 999,
                            "TN": 0,
                            "FP": 0,
                            "FN": 1
                            }
               }]



class Img_itk:
    def __init__(self, img, origin, spacing, orientation):
        self.img = img
        self.origin = origin
        self.spacing = spacing
        self.orientation = orientation


class Test_seg_metrics(unittest.TestCase):

    @parameterized.expand([test_case0])
    def test_computeQualityMeasures(self, case):

        for data_type in ['np_array', 'itk_img']:
            if data_type=='itk_img':
                pred = sitk.GetImageFromArray(case['pred'].astype('int16'))
                gdth = sitk.GetImageFromArray(case['gdth'].astype('int16'))
            else:
                pred = case['pred']
                gdth = case['gdth']

            for file_nb in ['single_file', 'batch_files']:
                if file_nb == 'batch_files':
                    pred = [pred, pred, pred, pred]
                    gdth = [gdth, gdth, gdth, gdth]

                outs = sg.write_metrics(labels=case['labels'],
                                       pred_img=pred,
                                       gdth_img=gdth,
                                       csv_file=None,
                                       metrics=case['metrics'],
                                       TPTNFPFN=True)
                for out in outs:
                    for o, e in zip(collections.OrderedDict(sorted(out.items())).items(),
                                    collections.OrderedDict(sorted(case['expected'].items())).items()):
                        print(o, e)


if __name__ == '__main__':
    unittest.main()
