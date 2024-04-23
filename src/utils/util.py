import logging
import os
import time
import torch
import shutil
import numpy as np
import nibabel as nib
import pandas
from typing import List, Tuple, Type, Union

def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)

    log_time = get_timestamp()
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, log_time))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg, log_time


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime



def save_csv(args, logger, patient_list,
             loss, loss_nsd,
             ):
    save_predict_dir = os.path.join(args.save_base_dir, 'csv_file')
    if not os.path.exists(save_predict_dir):
        os.makedirs(save_predict_dir)

    df_dict = {'patient': patient_list,
               'dice': loss,
               'nsd': loss_nsd,
               }

    df = pandas.DataFrame(df_dict)
    df.to_csv(os.path.join(save_predict_dir, 'prompt_' + str(args.num_prompts)
                           + '_' + str(args.save_name) + '.csv'), index=False)
    logger.info("- CSV saved")


def save_image(save_array, test_data, image_data, save_prediction_path):
    nib.save(nib.Nifti1Image(save_array[0, 0, :].permute(test_data.dataset.spatial_index).cpu().numpy(),
                             image_data.affine, image_data.header), save_prediction_path)


def _bbox_mask(mask_volume: torch.Tensor, diff=1, mode='train', dynamic=False, max_diff=10, return_extend=False) -> torch.Tensor:
    bbox_coords = []
    for volume in mask_volume:
        i_any = volume.any(dim=2).any(dim=1)
        j_any = volume.any(dim=2).any(dim=0)
        k_any = volume.any(dim=1).any(dim=0)

        i_min, i_max = torch.where(i_any)[0][[0, -1]]
        j_min, j_max = torch.where(j_any)[0][[0, -1]]
        k_min, k_max = torch.where(k_any)[0][[0, -1]]

        # i_max, j_max, k_max = i_max + diff, j_max + diff, k_max + diff
        # bb = torch.tensor([[i_min, j_min, k_min, i_max, j_max, k_max]])

        if dynamic and mode == 'train':
            # diff_ = np.random.choice(range(-max_diff, max_diff), size=6, replace=True)
            diff_ = np.random.choice(range(0, max_diff), size=6, replace=True)

            if max(0, i_min - diff_[0]) < min(i_max + diff_[1], 126):
                i_min, i_max = max(0, i_min - diff_[0]), min(i_max + diff_[1], 126)
            if max(0, j_min - diff_[2]) < min(j_max + diff_[3], 126):
                j_min, j_max = max(0, j_min - diff_[2]), min(j_max + diff_[3], 126)
            if max(0, k_min - diff_[4]) < min(k_max + diff_[5], 126):
                k_min, k_max = max(0, k_min - diff_[4]), min(k_max + diff_[5], 126)

        # delta_i = i_max - i_min + diff
        # delta_j = j_max - j_min + diff
        # delta_k = k_max - k_min + diff

        # diff_value = -5
        # i_min, i_max = max(0, i_min - diff_value), min(i_max + diff_value, 126)
        # j_min, j_max = max(0, j_min - diff_value), min(j_max + diff_value, 126)
        # k_min, k_max = max(0, k_min - diff_value), min(k_max + diff_value, 126)


        bb = torch.tensor([[i_min, j_min, k_min, i_max + 1, j_max + 1, k_max + 1]])
        # print(i_min, i_max + 1, j_min, j_max + 1, k_min, k_max + 1) # check dynamic box

        # bb = torch.tensor([[i_min, j_min, k_min, delta_i, delta_j, delta_k]])
        bbox_coords.append(bb)
        # print(torch.sum(volume), torch.sum(volume[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1]))
    bbox_coords = torch.stack(bbox_coords)
    return bbox_coords

