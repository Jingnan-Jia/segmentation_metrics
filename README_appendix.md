## Explanication of  surface distance based metrics

For each contour voxel of the segmented volume `A`, the Euclidean distance from the closest contour voxel of the reference volume `B` is computed and stored as `list1`. This computation is also performed for the contour voxels of the reference volume `B`, stored as `list2`. `list1` and `list2` are merged to get `list3`.
- `Hausdorff distance` is the maximum value of `list3`. 
- `Hausdorff distance 95% percentile` is the 95% percentile of `list3`. 
- `Mean (Average) surface distance` is the mean value of `list3`.
- `Median surface distance` is the median value of `list3`.
- `Std surface distance` is the standard deviation of `list3`. 

**References:**
1. Heimann T, Ginneken B, Styner MA, et al. Comparison and Evaluation of Methods for Liver Segmentation From CT Datasets. IEEE Transactions on Medical Imaging. 2009;28(8):1251–1265.
2. Yeghiazaryan, Varduhi, and Irina D. Voiculescu. "Family of boundary overlap metrics for the evaluation of medical image segmentation." Journal of Medical Imaging 5.1 (2018): 015006.
3. Ruskó, László, György Bekes, and Márta Fidrich. "Automatic segmentation of the liver from multi-and single-phase contrast-enhanced CT images." Medical Image Analysis 13.6 (2009): 871-882.