# Segmentaion Metrics Package [![DOI](https://zenodo.org/badge/273067948.svg)](https://zenodo.org/badge/latestdoi/273067948)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/Ordgod/segmentation_metrics)
![publish workflow status](https://github.com/Jingnan-Jia/segmentation_metrics/actions/workflows/python-publish.yml/badge.svg)
[![codecov](https://codecov.io/gh/Jingnan-Jia/segmentation_metrics/branch/master/graph/badge.svg?token=UO1QSYBEU6)](https://codecov.io/gh/Jingnan-Jia/segmentation_metrics)
![test workflow status](https://github.com/Jingnan-Jia/segmentation_metrics/actions/workflows/test_and_coverage.yml/badge.svg?branch=master)
[![Documentation Status](https://readthedocs.org/projects/segmentation-metrics/badge/?version=latest)](https://segmentation-metrics.readthedocs.io/en/latest/?badge=latest)
[![OSCS Status](https://www.oscs1024.com/platform/badge/Jingnan-Jia/segmentation_metrics.svg?size=small)](https://www.oscs1024.com/project/Jingnan-Jia/segmentation_metrics?ref=badge_small)

This is a simple package to compute different metrics for **Medical** image segmentation(images with suffix `.mhd`, `.mha`, `.nii`, `.nii.gz` or `.nrrd` ), and write them to csv file.

*BTW, if you need the support for more suffix, just let me know by creating new issues*
## Summary
To assess the segmentation performance, there are several different methods. Two main methods are volume-based metrics and distance-based metrics.


## Metrics included
This library computes the following performance metrics for segmentation:
 
### Voxel based metrics
- Dice (F-1)
- Jaccard
- Precision
- Recall
- False positive rate
- False negtive rate
- [Volume similarity](https://github.com/Jingnan-Jia/segmentation_metrics/blob/master/README_appendix.md#explanication-of-volume-similarity)

The equations for these metrics can be seen in the [wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall).

### Surface Distance based metrics (with spacing as default)
- [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance)
- Hausdorff distance 95% percentile
- Mean (Average) surface distance
- Median surface distance
- Std surface distance

**Note**: These metrics are **symmetric**, which means the distance from A to B is the same as the distance from B to A. More detailed explanication of these surface distance based metrics could be found [here](/README_appendix.md).


## Installation

```shell
$ pip install seg-metrics
```

## Getting started
Tutorial is at the [Colab](https://colab.research.google.com/drive/1LUH9cozeeSdmn9W_WdwBjKnrhWb39dq_?usp=sharing).

API reference is available at [Documentation](https://segmentation-metrics.readthedocs.io/en/latest/seg_metrics.html#seg_metrics.write_metrics)

Examples could be found below.


## Usage
At first, import the package:
```python
import seg_metrics.seg_metrics as sg
```


### Evaluate two batches of images with same filenames from two different folders
```python
labels = [0, 4, 5 ,6 ,7 , 8]
gdth_path = 'data/gdth'  # this folder saves a batch of ground truth images
pred_path = 'data/pred'  # this folder saves the same number of prediction images
csv_file = 'metrics.csv'  # results will be saved to this file and prented on terminal as well. If not set, results 
# will only be shown on terminal.

metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)
print(metrics)  # a list of dictionaries which includes the metrics for each pair of image.
```
After runing the above codes, you can get a **list of dictionaries** `metrics` which contains all the metrics. **Also you can find a `.csv` file containing all metrics in the same directory.** If the `csv_file` is not given, the metrics results will not be saved to disk.

### Evaluate two images
```python
labels = [0, 4, 5 ,6 ,7 , 8]
gdth_file = 'data/gdth.mhd'  # ground truth image full path
pred_file = 'data/pred.mhd'  # prediction image full path
csv_file = 'metrics.csv'

metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file)
```
After runing the above codes, you can get a **dictionary** `metrics` which contains all the metrics. **Also you can find a `.csv` file containing all metrics in the same directory.**

**Note:** 
1. When evaluating one image, the returned `metrics` is a dictionary.
2. When evaluating a batch of images, the returned `metrics` is a list of dictionaries.

### Evaluate two images with specific metrics
```python
labels = [0, 4, 5 ,6 ,7 , 8]
gdth_file = 'data/gdth.mhd'
pred_file = 'data/pred.mhd'
csv_file = 'metrics.csv'

metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file,
                  metrics=['dice', 'hd'])
# for only one metric
metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file,
                  metrics='msd')  
```

By passing the following parameters to select specific metrics.

```python
- dice:         Dice (F-1)
- jaccard:      Jaccard
- precision:    Precision
- recall:       Recall
- fpr:          False positive rate
- fnr:          False negtive rate
- vs:           Volume similarity

- hd:           Hausdorff distance
- hd95:         Hausdorff distance 95% percentile
- msd:          Mean (Average) surface distance
- mdsd:         Median surface distance
- stdsd:        Std surface distance
```

For example:
```python
labels = [1]
gdth_file = 'data/gdth.mhd'
pred_file = 'data/pred.mhd'
csv_file = 'metrics.csv'

metrics = sg.write_metrics(labels, gdth_file, pred_file, csv_file, metrics=['dice', 'hd95'])
dice = metrics['dice']
hd95 = metrics['hd95']
```


### Evaluate two images in memory instead of disk
**Note:**
1. The two images must be both numpy.ndarray or SimpleITK.Image.
2. Input arguments are different. Please use `gdth_img` and `pred_img` instead of `gdth_path` and `pred_path`.
3. If evaluating `numpy.ndarray`, the default `spacing` for all dimensions would be `1.0` for distance based metrics.
4. If you want to evaluate `numpy.ndarray` with specific spacing, pass a sequence with the length of image dimension as `spacing`.

```python
labels = [0, 1, 2]
gdth_img = np.array([[0,0,1], 
                     [0,1,2]])
pred_img = np.array([[0,0,1], 
                     [0,2,2]])
csv_file = 'metrics.csv'
spacing = [1, 2]
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
```

#### About the calculation of surface distance
The default surface distance is calculated based on **fullyConnected** border. To change the default connected type, 
you can set argument `fullyConnected` as `False` as follows.
```python
metrics = sg.write_metrics(labels=[1,2,3],
                        gdth_img=gdth_img,
                        pred_img=pred_img,
                        csv_file=csv_file,
                        fully_connected=False) 
```                  
In 2D image, fullyconnected means 8 neighbor points, while faceconnected means 4 neighbor points.
In 3D image, fullyconnected means 26 neighbor points, while faceconnected means 6 neighbor points.


# How to obtain more metrics? like "False omission rate" or "Accuracy"?
A great number of different metrics, like "False omission rate" or "Accuracy", could be derived from some the [confusion matrics](https://en.wikipedia.org/wiki/Confusion_matrix). To calculate more metrics or design custom metrics, use `TPTNFPFN=True` to return the number of voxels/pixels of true positive (TP), true negative (TN), false positive (FP), false negative (FN) predictions. For example,
```python
metrics = sg.write_metrics(
                        gdth_img=gdth_img,
                        pred_img=pred_img,
                        TPTNFPFN=True) 
tp, tn, fp, fn = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
false_omission_rate = fn/(fn+tn)
accuracy = (tp + tn)/(tp + tn + fp + fn)
```          

# Comparision with medpy
`medpy` also provide functions to calculate metrics for medical images. But `seg-metrics`     
has several advantages.
1. **Faster**. `seg-metrics` is **10 times faster** calculating distance based metrics. This [jupyter 
notebook](https://colab.research.google.com/drive/1gLQghS1d_fWsaJs3G4Ip0GlZHEJFcxDr#scrollTo=mDWvyxW7VExd) could reproduce the results. 
2. **More convenient**. `seg-metrics` can calculate all different metrics in once in one function while 
`medpy` needs to call different functions multiple times which cost more time and code.
3. **More Powerful**. `seg-metrics` can calculate **multi-label** segmentation metrics and save results to 
`.csv` file in good manner, but `medpy` only provides binary segmentation metrics. Comparision can be found in this [jupyter 
notebook](https://colab.research.google.com/drive/1gLQghS1d_fWsaJs3G4Ip0GlZHEJFcxDr#scrollTo=mDWvyxW7VExd).
 


If this repository helps you in anyway, show your love ❤️ by putting a ⭐ on this project. 
I would also appreciate it if you cite the package in your publication. (**Note:** This package is **NOT** approved for clinical use and is intended for research use only. )

# Bibtex

    @misc{Jingnan,
        title  = {A package to compute segmentation metrics: seg-metrics},
        author = {Jingnan Jia},
        url    = {https://github.com/Ordgod/segmentation_metrics}, 
        year   = {2020}, 
        doi = {10.5281/zenodo.3995075}
    }





