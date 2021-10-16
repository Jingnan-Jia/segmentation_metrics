import glob
import importlib
import os
import unittest
import SimpleITK as sitk
import numpy as np
import sys
import seg_metrics.seg_metrics as sg
from parameterized import parameterized
from medutils.medutils import save_itk
import tempfile

SUFFIX_LS = {".mhd", ".mha", ".nrrd", ".nii", ".nii.gz"}

TEST_CASE1 = [{
"IMG": np.random.randint(low=-1500, high=1500, size=(512, 512, 200)),
"ORIGIN": np.array([-192.345, 129.023, 1100]),
"SPACING": np.array([0.602, 0.602, 0.3]),
"ORIENTATION": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])}]

TEST_CASE2 = [{
"IMG": np.random.randint(low=-1500, high=1500, size=(512, 512)),
"ORIGIN": np.array([-192.345, 129.023]),
"SPACING": np.array([0.602, 0.602]),
"ORIENTATION": np.array([1, 0, 0, 1])}]

class Test_seg_metrics(unittest.TestCase):
    @parameterized.expand([TEST_CASE1, TEST_CASE2])
    def test_save_and_load(self, image):
        with tempfile.TemporaryDirectory() as tempdir:
            for suffix in SUFFIX_LS:
                img_fpath = os.path.join(tempdir, 'test_img' + suffix)
                # print('img', image['IMG'].shape)
                save_itk(img_fpath, image['IMG'], image['ORIGIN'], image['SPACING'])   # save image
                load_img, load_origin, load_spacing = sg.load_itk(img_fpath, require_ori_sp=True)  # load image
                # print(SUFFIX_LS)
                # print('suffix', suffix)
                # print('load_origin', load_origin)
                # print('image[ORIGIN]', image['ORIGIN'])
                self.assertIsNone(np.testing.assert_allclose(load_img, image['IMG']))
                self.assertIsNone(np.testing.assert_allclose(load_origin, image['ORIGIN']))
                self.assertIsNone(np.testing.assert_allclose(load_spacing, image['SPACING']))


if __name__ == '__main__':
    unittest.main()