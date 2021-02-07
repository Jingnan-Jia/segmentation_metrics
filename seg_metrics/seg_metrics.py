import numpy as np
import os
import SimpleITK as sitk
# import nibabel as nib
import pandas as pd
import copy
import PySimpleGUI as gui
import matplotlib.pyplot as plt
import glob
import sys
from myutil.myutil import load_itk, get_gdth_pred_names, one_hot_encode_3d


# %%
def show_itk(itk, idx):
    ref_surface_array = sitk.GetArrayViewFromImage(itk)
    plt.figure()
    plt.imshow(ref_surface_array[idx])
    plt.show()

    return None


def computeQualityMeasures(lP, lT, spacing, metrics_type=None):
    """

    :param lP: prediction, shape (x, y, z)
    :param lT: ground truth, shape (x, y, z)
    :param spacing: shape order (x, y, z)
    :return: quality: dict contains metircs
    """
    quality = {}
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelPred.SetSpacing(spacing)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    labelTrue.SetSpacing(spacing)  # spacing order (x, y, z)

    voxel_metrics = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs']
    distance_metrics = ['hd', 'hd95', 'msd', 'mdsd', 'stdsd']
    metrics_type = set([]) if metrics_type is None else set(metrics_type)
    # to save time, we need to determine which metrics we need to compute
    if set(voxel_metrics).intersection(metrics_type) or not metrics_type:
        pred = lP.astype(int)  # float data does not support bit_and and bit_or
        gdth = lT.astype(int)  # float data does not support bit_and and bit_or
        fp_array = copy.deepcopy(pred)  # keep pred unchanged
        fn_array = copy.deepcopy(gdth)
        gdth_sum = np.sum(gdth)
        pred_sum = np.sum(pred)
        intersection = gdth & pred
        union = gdth | pred
        intersection_sum = np.count_nonzero(intersection)
        union_sum = np.count_nonzero(union)

        tp_array = intersection

        tmp = pred - gdth
        fp_array[tmp < 1] = 0

        tmp2 = gdth - pred
        fn_array[tmp2 < 1] = 0

        tn_array = np.ones(gdth.shape) - union

        tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

        smooth = 0.001
        precision = tp / (pred_sum + smooth)
        recall = tp / (gdth_sum + smooth)

        false_positive_rate = fp / (fp + tn + smooth)
        false_negtive_rate = fn / (fn + tp + smooth)

        jaccard = intersection_sum / (union_sum + smooth)
        dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

        dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
        dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)

        quality["dice"] = dice
        quality["jaccard"] = jaccard
        quality["precision"] = precision
        quality["recall"] = recall
        quality["false_negtive_rate"] = false_negtive_rate
        quality["false_positive_rate"] = false_positive_rate
        quality["volume_similarity"] = dicecomputer.GetVolumeSimilarity()

    if set(distance_metrics).intersection(metrics_type) or not metrics_type:
        slice_idx = 300
        # Surface distance measures
        signed_distance_map = sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False,
                                                           useImageSpacing=True)  # It need to be adapted.
        # show_itk(signed_distance_map, slice_idx)

        ref_distance_map = sitk.Abs(signed_distance_map)
        # show_itk(ref_distance_map, slice_idx)

        ref_surface = sitk.LabelContour(labelTrue > 0.5, fullyConnected=True)
        # show_itk(ref_surface, slice_idx)
        ref_surface_array = sitk.GetArrayViewFromImage(ref_surface)

        statistics_image_filter = sitk.StatisticsImageFilter()
        statistics_image_filter.Execute(ref_surface > 0.5)

        num_ref_surface_pixels = int(statistics_image_filter.GetSum())

        signed_distance_map_pred = sitk.SignedMaurerDistanceMap(labelPred > 0.5, squaredDistance=False,
                                                                useImageSpacing=True)
        # show_itk(signed_distance_map_pred, slice_idx)

        seg_distance_map = sitk.Abs(signed_distance_map_pred)
        # show_itk(seg_distance_map, slice_idx)

        seg_surface = sitk.LabelContour(labelPred > 0.5, fullyConnected=True)
        # show_itk(seg_surface, slice_idx)
        seg_surface_array = sitk.GetArrayViewFromImage(seg_surface)

        seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
        # show_itk(seg2ref_distance_map, slice_idx)

        ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)
        # show_itk(ref2seg_distance_map, slice_idx)

        statistics_image_filter.Execute(seg_surface > 0.5)

        num_seg_surface_pixels = int(statistics_image_filter.GetSum())

        seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
        seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
        seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
        ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
        ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
        ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #

        all_surface_distances = seg2ref_distances + ref2seg_distances
        quality["mean_surface_distance"] = np.mean(all_surface_distances)
        quality["median_surface_distance"] = np.median(all_surface_distances)
        quality["std_surface_distance"] = np.std(all_surface_distances)
        quality["95_surface_distance"] = np.percentile(all_surface_distances, 95)
        quality["Hausdorff"] = np.max(all_surface_distances)

    return quality


def get_metrics_dict_all_labels(labels, gdth, pred, spacing, metrics_type=None):
    """

    :param metrics_type:
    :param labels: not include background, e.g. [4,5,6,7,8] or [1]
    :param gdth: shape: (x, y, z, channels), channels is equal to len(labels) or equal to len(labels)+1 (background)
    :param pred: the same as above
    :param spacing: spacing order should be (x, y, z) !!!
    :return: metrics_dict_all_labels a dict which contain all metrics
    """
    metrics_parameters_dict = {}

    Hausdorff_list = []
    Dice_list = []
    Jaccard_list = []
    Volume_list = []
    mean_surface_dis_list = []
    median_surface_dis_list = []
    std_surface_dis_list = []
    nine5_surface_dis_list = []
    precision_list = []
    recall_list = []
    false_positive_rate_list = []
    false_negtive_rate_list = []

    for i, label in enumerate(labels):
        print('start get metrics for label: ', label)
        pred_per = pred[..., i]  # select onlabel
        gdth_per = gdth[..., i]

        metrics = computeQualityMeasures(pred_per, gdth_per, spacing=spacing, metrics_type=metrics_type)
        print(metrics)

        Dice_list.append(metrics["dice"])
        Jaccard_list.append(metrics["jaccard"])
        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        false_negtive_rate_list.append(metrics["false_negtive_rate"])
        false_positive_rate_list.append(metrics["false_positive_rate"])
        Volume_list.append(metrics["volume_similarity"])

        mean_surface_dis_list.append(metrics["mean_surface_distance"])
        median_surface_dis_list.append(metrics["median_surface_distance"])
        std_surface_dis_list.append(metrics["std_surface_distance"])
        nine5_surface_dis_list.append(metrics["95_surface_distance"])
        Hausdorff_list.append(metrics["Hausdorff"])

    metrics_dict_all_labels = {'dice': Dice_list,
                               'jaccard': Jaccard_list,
                               'precision': precision_list,
                               'recall': recall_list,
                               'fpr': false_positive_rate_list,
                               'fnr': false_negtive_rate_list,
                               'vs': Volume_list,
                               'hd': Hausdorff_list,
                               'msd': mean_surface_dis_list,
                               'mdsd': median_surface_dis_list,
                               'stdsd': std_surface_dis_list,
                               'hd95': nine5_surface_dis_list}

    metrics_dict = {k: v for k, v in metrics_dict_all_labels.items() if v}  # remove empty values

    return metrics_dict


def write_metrics(labels, gdth_path, pred_path, csv_file, metrics=None):
    """

    :param labels:  exclude background
    :param gdth_path: a absolute directory path or file name
    :param pred_path: a absolute directory path or file name
    :param csv_file: filename to save the metrics
    :return: metrics_dict_all_labels: a dict which save metrics
    """
    print('start calculate all metrics (volume and distance) and write them to csv')
    if '/' not in gdth_path.split('.')[-1]:  # gdth is a file instead of a directory
        gdth_names, pred_names = [gdth_path], [pred_path]
    else:
        gdth_names, pred_names = get_gdth_pred_names(gdth_path, pred_path)

    for gdth_name, pred_name in zip(gdth_names, pred_names):
        gdth, gdth_origin, gdth_spacing = load_itk(gdth_name)
        pred, pred_origin, pred_spacing = load_itk(pred_name)

        gdth = one_hot_encode_3d(gdth, labels=labels)
        pred = one_hot_encode_3d(pred, labels=labels)
        print('start calculate all metrics for image: ', pred_name)
        metrics_dict_all_labels = get_metrics_dict_all_labels(labels, gdth, pred, spacing=gdth_spacing[::-1],
                                                              metrics_type=metrics)
        metrics_dict_all_labels['filename'] = pred_name  # add a new key to the metrics
        data_frame = pd.DataFrame(metrics_dict_all_labels)
        data_frame.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    return metrics_dict_all_labels


def main():
    labels = [0, 4, 5, 6, 7, 8]
    gdth_path = 'data/gdth'
    pred_path = 'data/pred'
    csv_file = 'metrics.csv'

    write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)


if __name__ == "__main__":
    main()
