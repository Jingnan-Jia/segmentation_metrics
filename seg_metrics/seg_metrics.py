import copy
import os
from typing import Dict, Union, Optional, Sequence
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from medutils.medutils import load_itk, get_gdth_pred_names, one_hot_encode_3d
import logging
from tqdm import tqdm

__all__ = ["write_metrics"]


def show_itk(img: sitk.SimpleITK.Image, idx: int) -> None:
    """Show a 2D slice of 3D ITK image.

    :param itk: ITK image
    :param idx: index of 2D slice
    """
    ref_surface_array = sitk.GetArrayViewFromImage(img)
    plt.figure()
    plt.imshow(ref_surface_array[idx])
    plt.show()

    return None


def computeQualityMeasures(lP: np.ndarray,
                           lT: np.ndarray,
                           spacing: np.ndarray,
                           metrics_names: Union[Sequence, set, None] = None,
                           fullyConnected=True):
    """

    :param lP: prediction, shape (x, y, z)
    :param lT: ground truth, shape (x, y, z)
    :param spacing: shape order (x, y, z)
    :return: metrics_names: container contains metircs names
    """
    quality = {}
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelPred.SetSpacing(spacing)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    labelTrue.SetSpacing(spacing)  # spacing order (x, y, z)

    voxel_metrics = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs']
    distance_metrics = ['hd', 'hd95', 'msd', 'mdsd', 'stdsd']
    if metrics_names is None:
        metrics_names = {'dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd', 'mdsd',
                         'stdsd'}
    else:
        metrics_names = set(metrics_names)
    # print('metrics0', metrics_names)

    # to save time, we need to determine which metrics we need to compute
    if set(voxel_metrics).intersection(metrics_names) or not metrics_names:
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

        fpr = fp / (fp + tn + smooth)
        fnr = fn / (fn + tp + smooth)

        jaccard = intersection_sum / (union_sum + smooth)
        dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

        dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
        dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)

        quality["dice"] = dice
        quality["jaccard"] = jaccard
        quality["precision"] = precision
        quality["recall"] = recall
        quality["fnr"] = fnr
        quality["fpr"] = fpr
        quality["vs"] = dicecomputer.GetVolumeSimilarity()
    # print('set(distance_metrics).intersection(metrics)', set(distance_metrics).intersection(metrics_names))
    # print('set(distance_metrics)', set(distance_metrics))
    # print('metrics', metrics_names)
    if set(distance_metrics).intersection(metrics_names) or not metrics_names:
        slice_idx = 300
        # Surface distance measures
        signed_distance_map = sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False,
                                                           useImageSpacing=True)  # It need to be adapted.
        # show_itk(signed_distance_map, slice_idx)

        ref_distance_map = sitk.Abs(signed_distance_map)
        # show_itk(ref_distance_map, slice_idx)

        ref_surface = sitk.LabelContour(labelTrue > 0.5, fullyConnected=fullyConnected)
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

        seg_surface = sitk.LabelContour(labelPred > 0.5, fullyConnected=fullyConnected)
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
        quality["msd"] = np.mean(all_surface_distances)
        quality["mdsd"] = np.median(all_surface_distances)
        quality["stdsd"] = np.std(all_surface_distances)
        quality["hd95"] = np.percentile(all_surface_distances, 95)
        quality["hd"] = np.max(all_surface_distances)
    return quality


def get_metrics_dict_all_labels(labels: Sequence,
                                gdth: np.ndarray,
                                pred: np.ndarray,
                                spacing: np.ndarray,
                                metrics_names: Union[Sequence, set, None] = None,
                                fullyConnected: bool = True) -> Dict[str, list]:
    """

    :param labels: not include background, e.g. [4,5,6,7,8] or [1]
    :param gdth: shape: (x, y, z, channels), channels is equal to len(labels) or equal to len(labels)+1 (background)
    :param pred: the same as above
    :param spacing: spacing order should be (x, y, z) !!!
    :param metrics_names: a list of metrics
    :param fullyConnected: if apply fully connected border during the calculation of surface distance.
    :return: metrics_dict_all_labels a dict which contain all metrics
    """
    if metrics_names is None:
        metrics_names = {'dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd', 'mdsd',
                         'stdsd'}
    if type(metrics_names) is str:
        metrics_names = [metrics_names]
    hd_list = []
    dice_list = []
    jaccard_list = []
    vs_list = []
    msd_list = []
    mdsd_list = []
    stdsd_list = []
    hd95_list = []
    precision_list = []
    recall_list = []
    fpr_list = []
    fnr_list = []

    label_list = [lb for lb in labels]

    metrics_dict_all_labels = {'label': label_list,
                               'dice': dice_list,
                               'jaccard': jaccard_list,
                               'precision': precision_list,
                               'recall': recall_list,
                               'fpr': fpr_list,
                               'fnr': fnr_list,
                               'vs': vs_list,
                               'hd': hd_list,
                               'msd': msd_list,
                               'mdsd': mdsd_list,
                               'stdsd': stdsd_list,
                               'hd95': hd95_list}

    for i, label in enumerate(labels):
        logging.info('\nstart to get metrics for label: ', label)
        pred_per = pred[..., i]  # select onlabel
        gdth_per = gdth[..., i]
        # print('metrics-1', metrics_names)
        metrics = computeQualityMeasures(pred_per, gdth_per,
                                         spacing=spacing,
                                         metrics_names=metrics_names,
                                         fullyConnected=fullyConnected)

        for k, v in metrics_dict_all_labels.items():
            if k in metrics_names:
                v.append(metrics[k])
    #     if "jaccard" in metrics.keys():
    #     jaccard_list.append(metrics["jaccard"])
    #     precision_list.append(metrics["precision"])
    #     recall_list.append(metrics["recall"])
    #     fnr_list.append(metrics["fnr"])
    #     fpr_list.append(metrics["fpr"])
    #     vs_list.append(metrics["vs"])
    #
    #     msd_list.append(metrics["msd"])
    #     mdsd_list.append(metrics["mdsd"])
    #     stdsd_list.append(metrics["stdsd"])
    #     hd95_list.append(metrics["hd95"])
    #     hd_list.append(metrics["hd"])
    #
    # label_list = [lb for lb in labels]
    #
    # metrics_dict_all_labels = {'label': label_list,
    #                            'dice': dice_list,
    #                            'jaccard': jaccard_list,
    #                            'precision': precision_list,
    #                            'recall': recall_list,
    #                            'fpr': fpr_list,
    #                            'fnr': fnr_list,
    #                            'vs': vs_list,
    #                            'hd': hd_list,
    #                            'msd': msd_list,
    #                            'mdsd': mdsd_list,
    #                            'stdsd': stdsd_list,
    #                            'hd95': hd95_list}

    metrics_dict = {k: v for k, v in metrics_dict_all_labels.items() if v}  # remove empty values

    return metrics_dict


def type_check(gdth_path: Union[str, pathlib.Path, Sequence, None],
               pred_path: Union[str, pathlib.Path, Sequence, None],
               gdth_img: Union[np.ndarray, sitk.SimpleITK.Image, Sequence, None],
               pred_img: Union[np.ndarray, sitk.SimpleITK.Image, Sequence, None]) -> None:
    if type(gdth_img) is not type(pred_img):  # gdth and pred should have the same type
        raise Exception(f"gdth_array is {type(gdth_img)} but pred_array is {type(pred_img)}. "
                        f"They should be the same type.")
    if type(gdth_path) is not type(pred_path):  # gdth_path and pred_path should have the same type
        raise Exception(f"gdth_array is {type(gdth_path)} but pred_array is {type(pred_path)}. "
                        f"They should be the same type.")
    if type(gdth_path) is type(gdth_img):
        raise Exception(f"gdth_array is {type(gdth_path)} but pred_array is {type(pred_path)}. "
                        f"Only one of them should be None, and the other should be assigned values.")

    assert any(isinstance(gdth_path, tp) for tp in [str, pathlib.Path, Sequence, type(None)])
    assert any(isinstance(gdth_img, tp) for tp in [np.ndarray, sitk.SimpleITK.Image, Sequence, type(None)])

    if isinstance(gdth_path, Sequence):
        assert any(isinstance(gdth_path, tp) for tp in [str, pathlib.Path])
    if isinstance(gdth_img, Sequence):
        if type(gdth_img[0]) not in [np.ndarray, sitk.SimpleITK.Image]:
            raise Exception(
                f"gdth_img[0]'s type should be ndarray or SimpleITK.SimpleITK.Image, but get {type(gdth_img)}")


def write_metrics(labels: Sequence,
                  gdth_path: Union[str, pathlib.Path, Sequence, None] = None,
                  pred_path: Union[str, pathlib.Path, Sequence, None] = None,
                  csv_file: Union[str, pathlib.Path, None] = None,
                  gdth_img: Union[np.ndarray, sitk.SimpleITK.Image, Sequence, None] = None,
                  pred_img: Union[np.ndarray, sitk.SimpleITK.Image, Sequence, None] = None,
                  metrics: Union[Sequence, set, None] = None,
                  verbose: bool = True,
                  spacing: Union[Sequence, np.ndarray, None] = None):
    """

    :param labels:  exclude background
    :param gdth_path: a absolute directory path or file name
    :param pred_path: a absolute directory path or file name
    :param gdth_img: np.ndarray for ground truth
    :param pred_img: np.ndarray for prediction
    :param csv_file: filename to save the metrics
    :return: metrics: a sequence which save metrics
    """
    type_check(gdth_path, pred_path, gdth_img, pred_img)
    logging.info('start to calculate metrics (volume or distance) and write them to csv')
    output_list = []
    metrics_dict_all_labels = None
    if metrics is None:
        metrics = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd', 'mdsd', 'stdsd']

    if gdth_path is not None:
        if os.path.isfile(gdth_path):  # gdth is a file instead of a directory
            gdth_names, pred_names = [gdth_path], [pred_path]
        else:
            gdth_names, pred_names = get_gdth_pred_names(gdth_path, pred_path)
        with tqdm(zip(gdth_names, pred_names), disable=not verbose) as pbar:
            for gdth_name, pred_name in pbar:
                pbar.set_description(f'Process {os.path.basename(pred_name)} ...')
                gdth, gdth_origin, gdth_spacing = load_itk(gdth_name, require_ori_sp=True)
                pred, pred_origin, pred_spacing = load_itk(pred_name, require_ori_sp=True)

                gdth = one_hot_encode_3d(gdth, labels=labels)
                pred = one_hot_encode_3d(pred, labels=labels)
                metrics_dict_all_labels = get_metrics_dict_all_labels(labels, gdth, pred, spacing=gdth_spacing[::-1],
                                                                      metrics_names=metrics)
                metrics_dict_all_labels['filename'] = pred_name  # add a new key to the metrics

                if csv_file:
                    data_frame = pd.DataFrame(metrics_dict_all_labels)
                    data_frame.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
                output_list.append(metrics_dict_all_labels)

    if gdth_img is not None:
        if type(gdth_img) in [sitk.SimpleITK.Image, np.ndarray]:  # gdth is a file instead of a list
            gdth_img, pred_img = [gdth_img], [pred_img]
        with tqdm(zip(gdth_img, pred_img), disable=not verbose) as pbar:
            img_id = 0
            for gdth, pred in pbar:
                img_id += 1
                if type(gdth) not in [sitk.SimpleITK.Image, np.ndarray]:
                    raise TypeError(f"image type should be sitk.SimpleITK.Image or np.ndarray, but is {type(gdth)}")
                if isinstance(gdth, sitk.SimpleITK.Image):
                    gdth_array = sitk.GetArrayFromImage(gdth)
                    pred_array = sitk.GetArrayFromImage(pred)

                    gdth_spacing = np.array(list(reversed(gdth.GetSpacing())))  # after reverseing, spacing =(z,y,x)
                    pred_spacing = np.array(list(reversed(pred.GetSpacing())))  # after reverseing, spacing =(z,y,x)
                    assert all(gdth_spacing == pred_spacing)
                    gdth_orientation = gdth.GetDirection()
                    if gdth_orientation[-1] == -1:
                        gdth_array = gdth_array[::-1]
                    pred_orientation = pred.GetDirection()
                    if pred_orientation[-1] == -1:
                        pred_array = pred_array[::-1]

                    gdth = gdth_array
                    pred = pred_array
                else:  # numpy.Ndarray
                    if spacing is None:
                        if gdth.ndim == 2:
                            gdth_spacing = np.array([1., 1.])  # spacing should be double
                        elif gdth.ndim == 3:
                            gdth_spacing = np.array([1., 1., 1.])  # spacing should be double
                        else:
                            raise Exception(f"The dimension of gdth should be 2 or 3, but it is {gdth.ndim}")
                    else:
                        gdth_spacing = np.array(spacing).astype(np.float32)
                        if len(gdth_spacing) not in (2, 3):
                            raise Exception(f"The length of spacing should be 2 or 3, but the spacing is {gdth_spacing} "
                                            f"with length of {len(gdth_spacing)}")


                gdth = one_hot_encode_3d(gdth, labels=labels)
                pred = one_hot_encode_3d(pred, labels=labels)
                metrics_dict_all_labels = get_metrics_dict_all_labels(labels, gdth, pred, spacing=gdth_spacing[::-1],
                                                                      metrics_names=metrics)
                # metrics_dict_all_labels['image_number'] = img_id  # add a new key to the metrics

                if csv_file:
                    data_frame = pd.DataFrame(metrics_dict_all_labels)
                    data_frame.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
                output_list.append(metrics_dict_all_labels)
    if csv_file:
        logging.info('Metrics were saved at : ', csv_file)

    if metrics_dict_all_labels is None:
        if gdth_path is not None:
            raise Exception(f"The metrics are None， because no files were detected in folder: {gdth_path} or folder: {pred_path}")
        # if gdth_img is not None:
        #     raise Exception(f"The metrics are None，because give image is None")
    if len(output_list)==0:
        return metrics_dict_all_labels
    else:
        return output_list


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
