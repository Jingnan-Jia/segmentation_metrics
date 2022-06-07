# Segmentaion Metrics Package [![DOI](https://zenodo.org/badge/273067948.svg)](https://zenodo.org/badge/latestdoi/273067948)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/Ordgod/segmentation_metrics)
![publish workflow status](https://github.com/Jingnan-Jia/segmentation_metrics/actions/workflows/python-publish.yml/badge.svg)
[![codecov](https://codecov.io/gh/Jingnan-Jia/segmentation_metrics/branch/master/graph/badge.svg?token=UO1QSYBEU6)](https://codecov.io/gh/Jingnan-Jia/segmentation_metrics)
![test workflow status](https://github.com/Jingnan-Jia/segmentation_metrics/actions/workflows/test_and_coverage.yml/badge.svg?branch=master)

This is a simple package to compute different metrics for **Medical** image segmentation(images with suffix `.mhd`, `.mha`, `.nii`, `.nii.gz` or `.nrrd`), and write them to csv file.

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
- Volume similarity

### Surface Distance based metrics (with spacing as default)
- Hausdorff distance
- Hausdorff distance 95% percentile
- Mean (Average) surface distance
- Median surface distance
- Std surface distance


## Installation

```shell
$ pip install seg-metrics
```

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
After runing the above codes, you can get a **list of dictionaries** `metrics` which contains all the metrics. **Also you can find a `.csv` file containing all metrics in the same directory.**

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
- dice:     Dice (F-1)
- jaccard:  Jaccard
- precision:    Precision
- recall:   Recall
- fpr:      False positive rate
- fnr:      False negtive rate
- vs:       Volume similarity

- hd:       Hausdorff distance
- hd95:     Hausdorff distance 95% percentile
- msd:      Mean (Average) surface distance
- mdsd:     Median surface distance
- stdsd:    Std surface distance
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
                        fullyConnected=False) 
```                  
In 2D image, fullyconnected means 8 neighbor points, while faceconnected means 4 neighbor points.
In 3D image, fullyconnected means 26 neighbor points, while faceconnected means 6 neighbor points.

If this repository helps you in anyway, show your love ❤️ by putting a ⭐ on this project. 
I would also appreciate it if you cite the package in your publication. (**Note:** This package is **NOT** approved for clinical use and is intended for research use only. )

#Bibtex

    @misc{Jingnan,
        title  = {A package to compute segmentation metrics: seg-metrics},
        author = {Jingnan Jia},
        url    = {https://github.com/Ordgod/segmentation_metrics}, 
        year   = {2020}, 
        doi = {10.5281/zenodo.3995075}
    }





