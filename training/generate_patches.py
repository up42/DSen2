import argparse
from pathlib import Path

from create_patches import readS2fromFile

LOGGER = get_logger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser(
        description="Read Sentinel-2 data. The code was adapted from N. Brodu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_folder_path",
        help=(
            "Path to folder with S2 SAFE files."
        ),
    )
    parser.add_argument(
        "--test_data",
        default=False,
        action="store_true",
        help="Store test patches in a separate dir.",
    )
    parser.add_argument(
        "--rgb_images",
        default=False,
        action="store_true",
        help=(
            "If set, write PNG images for the original and the superresolved bands,"
            " together with a composite rgb image (first three 10m bands), all with a "
            "quick and dirty clipping to 99%% of the original bands dynamic range and "
            "a quantization of the values to 256 levels."
        ),
    )
    parser.add_argument(
        "--save_prefix",
        default="../data/",
        help=(
            "If set, speficies the name of a prefix for all output files. "
            "Use a trailing / to save into a directory. The default of no prefix will "
            "save into the current directory. Example: --save_prefix result/"
        ),
    )
    parser.add_argument(
        "--run_60",
        default=False,
        action="store_true",
        help="If set, it will create patches also from the 60m channels.",
    )
    parser.add_argument(
        "--true_data",
        default=False,
        action="store_true",
        help=(
            "If set, it will create patches for S2 without GT. This option is not "
            "really useful here, please check the testing folder for predicting S2 images."
        ),
    )
    parser.add_argument(
        "--train_data",
        default=False,
        action="store_true",
        help="Store train patches in a separate dir",
    )

    args = parser.parse_args()
    return args

def main(args):
    LOGGER.info(
        f"I will proceed with file {args.data_folder_path}"
    )  # pylint: disable=logging-fstring-interpolation

    for file_path in Path(args.data_folder_path).glob("S2*"):
        LOGGER.info(f"Processing {file_path}")
        readS2fromFile(
            str(file_path),
            "",
            args.save_prefix,
            args.rgb_images,
            args.run_60,
            args.true_data,
            args.test_data,
            args.train_data,
        ).process_patches()

if __name__ == "__main__":
    main(arg_parse())
