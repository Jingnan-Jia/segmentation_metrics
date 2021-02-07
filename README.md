# Segmentaion Metrics Package [![DOI](https://zenodo.org/badge/273067948.svg)](https://zenodo.org/badge/latestdoi/273067948)

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
- 

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


### Evaluate two batch of images with same filenames from two different folders
```python
labels = [0, 4, 5 ,6 ,7 , 8]
gdth_path = 'data/gdth'
pred_path = 'data/pred'
csv_file = 'metrics.csv'

metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)
print(metrics)
```
After runing the above codes, you can get a dict `metrics` which contains all the metrics. Also you can find a `.csv` file containing all metrics in the same directory.

### Evaluate two images
```python
labels = [0, 4, 5 ,6 ,7 , 8]
gdth_file = 'data/gdth.mhd'
pred_file = 'data/pred.mhd'
csv_file = 'metrics.csv'

metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file)
```

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

If this repository helps you in anyway, show your love ❤️ by putting a ⭐ on this project. I would also appreciate it if you cite the package in your publication.

#Bibtex

    @misc{Jingnan,
        title  = {A package to compute segmentation metrics: seg-metrics},
        author = {Jingnan Jia},
        url    = {https://github.com/Ordgod/segmentation_metrics}, 
        year   = {2020}, 
        doi = {10.5281/zenodo.3995075}
    }





