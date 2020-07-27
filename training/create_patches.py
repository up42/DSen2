from __future__ import division
import os
import sys
import re
import glob
import argparse
import json

from collections import defaultdict
from typing import List, Tuple
import numpy as np
import imageio

import rasterio
from rasterio.windows import Window
import pyproj as proj

from training_utils import get_logger


sys.path.append("../")
from utils.patches import (
    downPixelAggr,
    save_test_patches,
    save_random_patches,
    save_random_patches60,
    save_test_patches60,
)

LOGGER = get_logger(__name__)


class readS2fromFile:
    def __init__(
        self,
        data_file,
        clip_to_aoi=None,
        save_prefix="../data/",
        rgb_images=False,
        run_60=False,
        true_data=False,
        test_data=False,
        train_data=False,
    ):
        self.data_file = data_file
        self.test_data = test_data
        self.clip_to_aoi = clip_to_aoi
        self.save_prefix = save_prefix
        self.rgb_images = rgb_images
        self.run_60 = run_60
        self.true_data = true_data
        self.test_data = test_data
        self.train_data = train_data

    def get_data(self) -> list:
        """
        This method returns the raster data set of original image for
        all the available resolutions.
        """
        data_path = ""
        data_folder = "MTD*.xml"
        for file in glob.iglob(
            os.path.join(self.data_file, data_folder), recursive=True,
        ):
            data_path = file

        raster_data = rasterio.open(data_path)
        datasets = raster_data.subdatasets

        return datasets

    def checking_data_file(self):
        if self.data_file.endswith("/"):
            tmp = os.path.split(self.data_file)[0]
            self.data_file = os.path.split(tmp)[1]
        else:
            self.data_file = os.path.split(self.data_file)[1]
        return self.data_file

    @staticmethod
    def get_max_min(x_1: int, y_1: int, x_2: int, y_2: int, data) -> Tuple:
        """
        This method gets pixels' location for the region of interest on the 10m bands
        and returns the min/max in each direction and to nearby 60m pixel boundaries and the area
        associated to the region of interest.
        **Example**
        >>> get_max_min(0,0,400,400)
        (0, 0, 395, 395, 156816)

        """
        with rasterio.open(data) as d_s:
            d_width = d_s.width
            d_height = d_s.height

        tmxmin = max(min(x_1, x_2, d_width - 1), 0)
        tmxmax = min(max(x_1, x_2, 0), d_width - 1)
        tmymin = max(min(y_1, y_2, d_height - 1), 0)
        tmymax = min(max(y_1, y_2, 0), d_height - 1)
        # enlarge to the nearest 60 pixel boundary for the super-resolution
        tmxmin = int(tmxmin / 6) * 6
        tmxmax = int((tmxmax + 1) / 6) * 6 - 1
        tmymin = int(tmymin / 6) * 6
        tmymax = int((tmymax + 1) / 6) * 6 - 1
        area = (tmxmax - tmxmin + 1) * (tmymax - tmymin + 1)
        return tmxmin, tmymin, tmxmax, tmymax, area

    # pylint: disable-msg=too-many-locals
    def to_xy(self, lon: float, lat: float, data) -> Tuple:
        """
        This method gets the longitude and the latitude of a given point and projects it
        into pixel location in the new coordinate system.
        :param lon: The longitude of a chosen point
        :param lat: The longitude of a chosen point
        :return: The pixel location in the coordinate system of the input image
        """
        # get the image's coordinate system.
        with rasterio.open(data) as d_s:
            coor = d_s.transform
        a_t, b_t, xoff, d_t, e_t, yoff = [coor[x] for x in range(6)]

        # transform the lat and lon into x and y position which are defined in
        # the world's coordinate system.
        local_crs = self.get_utm(data)
        crs_wgs = proj.Proj(init="epsg:4326")  # WGS 84 geographic coordinate system
        crs_bng = proj.Proj(init=local_crs)  # use a locally appropriate projected CRS
        x_p, y_p = proj.transform(crs_wgs, crs_bng, lon, lat)
        x_p -= xoff
        y_p -= yoff

        # matrix inversion
        # get the x and y position in image's coordinate system.
        det_inv = 1.0 / (a_t * e_t - d_t * b_t)
        x_n = (e_t * x_p - b_t * y_p) * det_inv
        y_n = (-d_t * x_p + a_t * y_p) * det_inv
        return int(x_n), int(y_n)

    @staticmethod
    def get_utm(data) -> str:
        """
        This method returns the utm of the input image.
        :param data: The raster file for a specific resolution.
        :return: UTM of the selected raster file.
        """
        with rasterio.open(data) as d_s:
            data_crs = d_s.crs.to_dict()
        utm = data_crs["init"]
        return utm

    # pylint: disable-msg=too-many-locals
    def area_of_interest(self, data):
        """
        This method returns the coordinates that define the desired area of interest.
        """
        if self.clip_to_aoi:
            roi_lon1, roi_lat1, roi_lon2, roi_lat2 = [
                float(x) for x in re.split(",", args.roi__y)
            ]
            x_1, y_1 = self.to_xy(roi_lon1, roi_lat1, data)
            x_2, y_2 = self.to_xy(roi_lon2, roi_lat2, data)
        else:
            x_1, y_1, x_2, y_2 = 0, 0, 20000, 20000

        xmi, ymi, xma, yma, area = self.get_max_min(x_1, y_1, x_2, y_2, data)
        return xmi, ymi, xma, yma, area

    @staticmethod
    def validate_description(description: str) -> str:
        """
        This method rewrites the description of each band in the given data set.
        :param description: The actual description of a chosen band.

        **Example**
        >>> ds10.descriptions[0]
        'B4, central wavelength 665 nm'
        >>> validate_description(ds10.descriptions[0])
        'B4 (665 nm)'
        """
        m_re = re.match(r"(.*?), central wavelength (\d+) nm", description)
        if m_re:
            return m_re.group(1) + " (" + m_re.group(2) + " nm)"
        return description

    @staticmethod
    def get_band_short_name(description):
        """
        This method returns only the name of the bands at a chosen resolution.

        :param description: This is the output of the validate_description method.

        **Example**
        >>> desc = validate_description(ds10.descriptions[0])
        >>> desc
        'B4 (665 nm)'
        >>> get_band_short_name(desc)
        'B4'
        """
        if "," in description:
            return description[: description.find(",")]
        if " " in description:
            return description[: description.find(" ")]
        return description[:3]

    def validate(self, data) -> Tuple:
        """
        This method takes the short name of the bands for each
        separate resolution and returns three lists. The validated_
        bands and validated_indices contain the name of the bands and
        the indices related to them respectively.
        The validated_descriptions is a list of descriptions for each band
        obtained from the validate_description method.
        :param data: The raster file for a specific resolution.
        **Example**
        >>> validated_10m_bands, validated_10m_indices, \
        >>> dic_10m = validate(ds10)
        >>> validated_10m_bands
        ['B4', 'B3', 'B2', 'B8']
        >>> validated_10m_indices
        [0, 1, 2, 3]
        >>> dic_10m
        defaultdict(<class 'str'>, {'B4': 'B4 (665 nm)',
         'B3': 'B3 (560 nm)', 'B2': 'B2 (490 nm)', 'B8': 'B8 (842 nm)'})
        """
        input_select_bands = "B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12"  # type: str
        select_bands = re.split(",", input_select_bands)  # type: List[str]
        validated_bands = []  # type: list
        validated_indices = []  # type: list
        validated_descriptions = defaultdict(str)  # type: defaultdict
        with rasterio.open(data) as d_s:
            for i in range(0, d_s.count):
                desc = self.validate_description(d_s.descriptions[i])
                name = self.get_band_short_name(desc)
                if name in select_bands:
                    select_bands.remove(name)
                    validated_bands += [name]
                    validated_indices += [i]
                    validated_descriptions[name] = desc
        return validated_bands, validated_indices, validated_descriptions

    @staticmethod
    # pylint: disable-msg=too-many-arguments
    def data_final(
        data, term: List, x_mi: int, y_mi: int, x_ma: int, y_ma: int, n_res
    ) -> np.ndarray:
        """
        This method takes the raster file at a specific
        resolution and uses the output of get_max_min
        to specify the area of interest.
        Then it returns an numpy array of values
        for all the pixels inside the area of interest.
        :param data: The raster file for a specific resolution.
        :param term: The validate indices of the
        bands obtained from the validate method.
        :return: The numpy array of pixels' value.
        """
        if term:
            LOGGER.info(term)
            with rasterio.open(data) as d_s:
                d_final = np.rollaxis(
                    d_s.read(
                        window=Window(
                            col_off=x_mi,
                            row_off=y_mi,
                            width=x_ma - x_mi + n_res,
                            height=y_ma - y_mi + n_res,
                        )
                    ),
                    0,
                    3,
                )[:, :, term]
        return d_final

    def save_band(self, data, name, percentile_data=None):
        # The percentile_data argument is used to plot superresolved and original data
        # with a comparable black/white scale
        if percentile_data is None:
            percentile_data = data
        mi, ma = np.percentile(percentile_data, (1, 99))
        band_data = np.clip(data, mi, ma)
        band_data = (band_data - mi) / (ma - mi)
        imageio.imsave(
            self.save_prefix + name + ".png", band_data
        )  # img_as_uint(band_data))

    # chan3 = data10[:, :, 0]
    # vis = (chan3 < 1).astype(np.int)
    # if np.sum(vis) > 0:
    #     print("The selected image has some blank pixels")
    #     # sys.exit()
    @staticmethod
    def check_size(dims):
        xmin, ymin, xmax, ymax = dims
        if xmax < xmin or ymax < ymin:
            LOGGER.error("Invalid region of interest / UTM Zone combination")
            sys.exit(1)

        if (xmax - xmin) < 192 or (ymax - ymin) < 192:
            LOGGER.error(
                "AOI too small. Try again with a larger AOI (minimum pixel width or heigh of 192)"
            )
            sys.exit(1)

    def get_original_image(self) -> Tuple:

        data_list = self.get_data()
        for dsdesc in data_list:
            if "10m" in dsdesc:
                xmin, ymin, xmax, ymax, interest_area = self.area_of_interest(dsdesc)
                LOGGER.info("Selected pixel region:")
                LOGGER.info("xmin = %s", xmin)
                LOGGER.info("ymin = %s", ymin)
                LOGGER.info("xmax = %s", xmax)
                LOGGER.info("ymax = %s", ymax)
                LOGGER.info("The area of selected region = %s", interest_area)
            self.check_size(dims=(xmin, ymin, xmax, ymax))

        for dsdesc in data_list:
            if "10m" in dsdesc:
                LOGGER.info("Selected 10m bands:")
                _, validated_10m_indices, _ = self.validate(dsdesc)
                data10 = self.data_final(
                    dsdesc, validated_10m_indices, xmin, ymin, xmax, ymax, 1
                )
            if "20m" in dsdesc:
                LOGGER.info("Selected 20m bands:")
                _, validated_20m_indices, _ = self.validate(dsdesc)
                data20 = self.data_final(
                    dsdesc,
                    validated_20m_indices,
                    xmin // 2,
                    ymin // 2,
                    xmax // 2,
                    ymax // 2,
                    1 // 2,
                )
            if "60m" in dsdesc:
                LOGGER.info("Selected 60m bands:")
                _, validated_60m_indices, _ = self.validate(dsdesc)
                data60 = self.data_final(
                    dsdesc,
                    validated_60m_indices,
                    xmin // 6,
                    ymin // 6,
                    xmax // 6,
                    ymax // 6,
                    1 // 6,
                )

        return data10, data20, data60, xmin, ymin, xmax, ymax

    def get_downsampled_images(self, data10, data20, data60):
        if self.run_60:
            data10_lr = downPixelAggr(data10, SCALE=6)
            data20_lr = downPixelAggr(data20, SCALE=6)
            data60_lr = downPixelAggr(data60, SCALE=6)
            return data10_lr, data20_lr, data60_lr
        else:
            data10_lr = downPixelAggr(data10, SCALE=2)
            data20_lr = downPixelAggr(data20, SCALE=2)

            return data10_lr, data20_lr

    def process_patches(self):
        if self.run_60:
            scale = 6
        else:
            scale = 2

        self.data_file = self.checking_data_file()
        data10, data20, data60, xmin, ymin, xmax, ymax = self.get_original_image()

        if self.test_data:
            out_per_image = self.saving_test_data(data10, data20, data60)
            with open(out_per_image + "roi.json", "w") as f:
                json.dump(
                    [
                        xmin // scale,
                        ymin // scale,
                        (xmax + 1) // scale,
                        (ymax + 1) // scale,
                    ],
                    f,
                )

        if self.rgb_images:
            self.create_rgb_images(data10, data20, data60)

        if self.true_data:
            out_per_image = self.saving_true_data(data10, data20, data60)
            with open(out_per_image + "roi.json", "w") as f:
                json.dump(
                    [
                        xmin // scale,
                        ymin // scale,
                        (xmax + 1) // scale,
                        (ymax + 1) // scale,
                    ],
                    f,
                )

        if self.train_data:
            self.saving_train_data(data10, data20, data60)

        LOGGER.info("Success.")

    def saving_test_data(self, data10, data20, data60):
        # if test_data:
        if self.run_60:
            data10_lr, data20_lr, data60_lr = self.get_downsampled_images(
                data10, data20, data60
            )
            out_per_image0 = self.save_prefix + "test60/"
            out_per_image = self.save_prefix + "test60/" + self.data_file + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)

            LOGGER.info(f"Writing files for testing to:{out_per_image}")
            save_test_patches60(data10_lr, data20_lr, data60_lr, out_per_image)

        else:
            data10_lr, data20_lr = self.get_downsampled_images(data10, data20, data60)
            out_per_image0 = self.save_prefix + "test/"
            out_per_image = self.save_prefix + "test/" + self.data_file + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)

            LOGGER.info(
                f"Writing files for testing to:{out_per_image}"
            )  # pylint: disable=logging-fstring-interpolation
            save_test_patches(data10_lr, data20_lr, out_per_image)

        if not os.path.isdir(out_per_image + "no_tiling/"):
            os.mkdir(out_per_image + "no_tiling/")

        LOGGER.info("Now saving the whole image without tiling...")
        if self.run_60:
            np.save(
                out_per_image + "no_tiling/" + "data60_gt", data60.astype(np.float32)
            )
            np.save(
                out_per_image + "no_tiling/" + "data60", data60_lr.astype(np.float32)
            )
        else:
            np.save(
                out_per_image + "no_tiling/" + "data20_gt", data20.astype(np.float32)
            )
            self.save_band(data10_lr[:, :, 0:3], "/test/" + self.data_file + "/RGB")
        np.save(out_per_image + "no_tiling/" + "data10", data10_lr.astype(np.float32))
        np.save(out_per_image + "no_tiling/" + "data20", data20_lr.astype(np.float32))
        return out_per_image

    def create_rgb_images(self, data10, data20, data60):
        # elif write_images
        data10_lr, data20_lr = self.get_downsampled_images(data10, data20, data60)
        LOGGER.info("Creating RGB images...")
        self.save_band(data10_lr[:, :, 0:3], "/raw/rgbs/" + self.data_file + "RGB")
        self.save_band(data20_lr[:, :, 0:3], "/raw/rgbs/" + self.data_file + "RGB20")

    def saving_true_data(self, data10, data20, data60):
        # elif true_data:
        out_per_image0 = self.save_prefix + "true/"
        out_per_image = self.save_prefix + "true/" + self.data_file + "/"
        if not os.path.isdir(out_per_image0):
            os.mkdir(out_per_image0)
        if not os.path.isdir(out_per_image):
            os.mkdir(out_per_image)

        LOGGER.info(
            f"Writing files for testing to:{out_per_image}"
        )  # pylint: disable=logging-fstring-interpolation
        save_test_patches60(
            data10, data20, data60, out_per_image, patchSize=384, border=12
        )

        if not os.path.isdir(out_per_image + "no_tiling/"):
            os.mkdir(out_per_image + "no_tiling/")

        LOGGER.info("Now saving the whole image without tiling...")
        np.save(out_per_image + "no_tiling/" + "data10", data10.astype(np.float32))
        np.save(out_per_image + "no_tiling/" + "data20", data20.astype(np.float32))
        np.save(out_per_image + "no_tiling/" + "data60", data60.astype(np.float32))
        return out_per_image

    def saving_train_data(self, data10, data20, data60):
        # if train_data
        if self.run_60:
            out_per_image0 = self.save_prefix + "train60/"
            out_per_image = self.save_prefix + "train60/" + self.data_file + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)
            LOGGER.info(
                f"Writing files for training to:{out_per_image}"
            )  # pylint: disable=logging-fstring-interpolation
            data10_lr, data20_lr, data60_lr = self.get_downsampled_images(
                data10, data20, data60
            )
            save_random_patches60(
                data60, data10_lr, data20_lr, data60_lr, out_per_image
            )
        else:
            out_per_image0 = self.save_prefix + "train/"
            out_per_image = self.save_prefix + "train/" + self.data_file + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)
            LOGGER.info(
                f"Writing files for training to:{out_per_image}"
            )  # pylint: disable=logging-fstring-interpolation
            data10_lr, data20_lr = self.get_downsampled_images(data10, data20, data60)
            save_random_patches(data20, data10_lr, data20_lr, out_per_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read Sentinel-2 data. The code was adapted from N. Brodu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file",
        help=(
            "An input Sentinel-2 data file. This can be either the original ZIP file,"
            " or the S2A[...].xml file in a SAFE directory extracted from that ZIP."
        ),
    )
    parser.add_argument(
        "--clip_to_aoi",
        default="",
        help=(
            "Sets the region of interest to extract as pixels locations on the 10m"
            'bands. Use this syntax: x_1,y_1,x_2,y_2. E.g. --roi_x_y "2000,2000,3200,3200"'
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

    LOGGER.info(
        f"I will proceed with file {args.data_file}"
    )  # pylint: disable=logging-fstring-interpolation
    readS2fromFile(
        args.data_file,
        args.clip_to_aoi,
        args.save_prefix,
        args.rgb_images,
        args.run_60,
        args.true_data,
        args.test_data,
        args.train_data,
    ).process_patches()
