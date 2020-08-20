from __future__ import print_function, division
import os

import time
import argparse
from glob import glob

# from tensorflow import keras
import numpy as np
from keras.optimizers import Nadam

from quality_metrics import psnr, uiq, sam, sre

from data_utils import get_logger
from patches import recompose_images, OpenDataFilesTest, OpenDataFiles
from DSen2Net import s2model
logger = get_logger(__name__)

SCALE = 2000
lr = 1e-4
MODEL_PATH = "../models/"


def write_final_dict(metric, metric_dict):
    # Create a directory to save the text file of including evaluation values.
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    with open(os.path.join(predict_path, metric + '.txt'), 'w') as f:
        f.writelines('{}:{}\n'.format(k, v) for k, v in metric_dict.items())


def predict_downsampled_img(path, folder, dset, border, final_name):

    logger.info("Loading weight ...")
    if args.run_60:
        input_shape = ((4, None, None), (6, None, None), (2, None, None))  # type: ignore
    else:
        input_shape = ((4, None, None), (6, None, None))  # type: ignore
        # create model
    model = s2model(input_shape, num_layers=6, feature_size=128)
    print("Symbolic Model Created.")

    nadam = Nadam(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004
    )

    model.compile(
        optimizer=nadam, loss="mean_absolute_error", metrics=["mean_squared_error"]
    )
    print("Model compiled.")
    model.count_params()

    # model = keras.models.load_model(MODEL_PATH +"s2_030_lr_1e-05.hdf5")
    model.load_weights(MODEL_PATH +"s2_032_lr_1e-04.hdf5")

    start = time.time()
    print("Timer started.")
    print("Predicting: {}.".format(dset))
    train, image_size = OpenDataFilesTest(
        os.path.join(path, folder + dset), args.run_60, SCALE, False
    )
    logger.info("Predicting ...")
    prediction = model.predict(train, batch_size=8, verbose=1)

    images = recompose_images(prediction, border=border, size=image_size)
    import pdb;
    pdb.set_trace()
    print("Writing to file...")
    np.save(os.path.join(path, folder + dset + "/no_tiling/" + final_name), images * SCALE)
    end = time.time()
    logger.info(f"Elapsed time: {end - start}.")


def evaluation(org_img, pred_img, metric):
    org_img_array = np.load(org_img)
    pred_img_array = np.load(pred_img)

    result = eval(f"{metric}(org_img_array, pred_img_array)")
    return result


def process(path, metric):
    if args.l1c:
        prefix = "l1c"
    if args.l2a:
        prefix = "l2a"
    if args.run_60:
        folder = prefix + "test60/"
        border = 12
        final_name = "data60_predicted"
    else:
        folder = prefix + "test/"
        border = 4
        final_name = "data20_predicted"

    path_to_patches = os.path.join(path, folder)

    fileList = [
        os.path.basename(x) for x in sorted(glob(path_to_patches + "*SAFE"))
    ]

    gt_sr = []
    metric_dict = {}

    for dset in fileList:
        if args.run_60:
            org_img_path = os.path.join(path_to_patches, dset + "/no_tiling/data60_gt.npy")
            pred_img_path = os.path.join(path_to_patches, dset +  "/no_tiling/data60_predicted.npy")
        else:
            org_img_path = os.path.join(path_to_patches, dset + "/no_tiling/data20_gt.npy")
            pred_img_path = os.path.join(path_to_patches, dset + "/no_tiling/data20_predicted.npy")

        predict_downsampled_img(path, folder, dset, border, final_name)
        eval_value = evaluation(org_img_path, pred_img_path, metric)
        gt_sr.append(eval_value)
    metric_dict["GT_SR"] = sum(gt_sr) / len(gt_sr)

    write_final_dict(metric, metric_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluates an Image Super Resolution Model")
    parser.add_argument("--path", type=str, help="Path to ")
    parser.add_argument("--l1c", action="store_true", help="getting L1C samples")
    parser.add_argument("--l2a", action="store_true", help="getting L2A samples")
    parser.add_argument("--metric", type=str, default="psnr", help="use psnr, uiq, sam or sre as evaluation metric")
    parser.add_argument(
        "--run_60",
        action="store_true",
        help="Whether to run a 60->10m network. Default 20->10m.",
    )

    args = parser.parse_args()
    path = args.path
    metric = args.metric

    process(path, metric)

