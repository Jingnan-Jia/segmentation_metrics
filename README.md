# Segmentaion Metrics Package

This is a simple package to compute different metrics for **Medical** image segmentation.

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

### Surface Distance based metrics
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
For simple usage examples, see `surface_distance_test.py`.
