import unittest
import collections
import numpy as np
from parameterized import parameterized
import SimpleITK as sitk
import seg_metrics.seg_metrics as sg

spacing_1 = [1., 2., 3.]
spacing_2 = np.array([1., 2., 3.])
spacing_3 = [1, 2, 3]
spacing_4 = np.array([1, 2, 3])
spacing_5 = [1.5, 2.5, 3.5]
spacing_6 = np.array([1.5, 2.5, 3.5])


class Test_seg_metrics(unittest.TestCase):

    def test_spacing(self):
        labels = [0, 1, 2]
        gdth_img = np.array([[0, 0, 1],
                             [0, 1, 2]])
        pred_img = np.array([[0, 0, 1],
                             [0, 2, 2]])
        csv_file = 'metrics.csv'
        for spacing in [spacing_1, spacing_2, spacing_3, spacing_4, spacing_5, spacing_6]:
            # spacing = np.array([1., 2.])
            metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                       gdth_img=gdth_img,
                                       pred_img=pred_img,
                                       csv_file=csv_file,
                                       spacing=spacing,
                                       metrics=['dice', 'hd'])
            # for only one metrics
            metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                                       gdth_img=gdth_img,
                                       pred_img=pred_img,
                                       csv_file=csv_file,
                                       spacing=spacing,
                                       metrics='msd')


if __name__ == '__main__':
    unittest.main()
