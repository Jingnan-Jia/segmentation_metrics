import glob
import importlib
import os
import unittest
import SimpleITK as sitk
import numpy as np
import sys
import seg_metrics.seg_metrics as sg
from myutil.myutil import save_itk

def save_multi_suffix_if_need(img_itk, dirname, prefix_list):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    for prefix in prefix_list:
        file_fpath = dirname + "/test_img" + prefix
        if not os.path.isfile(file_fpath):
            save_itk(file_fpath, img_itk.img, img_itk.origin, img_itk.spacing, img_itk.orientation)


class Img_itk:
    def __init__(self, img, origin, spacing, orientation):
        self.img = img
        self.origin = origin
        self.spacing = spacing
        self.orientation = orientation

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

class Test_load_itk(unittest.TestCase):

    def test_prefix(self):
        self.file_dir = "test_data"  # full path
        self.img = np.random.randint(low=-3000, high=3000, size=(512, 512, 800))
        self.origin = np.array([-192.345, 129.023, 1100])
        self.spacing = np.array([0.602, 0.602, 0.3])
        self.orientation = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

        self.img_itk = Img_itk(self.img, self.origin, self.spacing, self.orientation)
        self.prefix_list = [".mhd", ".mha", ".nrrd"]

        save_multi_suffix_if_need(self.img_itk, self.file_dir, self.prefix_list)

        file_names = []
        for prefix in self.prefix_list:
            file_names.extend(glob.glob(self.file_dir + '/*' + prefix))
        for file in file_names:
            load_img, load_origin, load_spacing = sg.load_itk(file)
            self.assertEqual(load_img.all(), self.img.all())
            self.assertEqual(load_origin.all(), self.origin.all())
            self.assertEqual(load_spacing.all(), self.spacing.all())

if __name__ == '__main__':
    unittest.main()
