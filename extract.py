import cv2
from skimage import io
from barvocuc import ImageAnalyzer
import os
import pandas as pd
import numpy as np
import utils
from pathlib import Path

import utils.directory_utils
import utils.yaml_utils


class LuminanceExtraction:
    def __init__(self, config):
        self.config = utils.yaml_utils.read_config(config)

    # results = {}

    def automatic_color_segmentation(self):
        folders_to_process, save_folder = utils.directory_utils.search_existing_directories(self.config,
                                                                            "auto_segment_results",
                                                                            "background_segmented")
        for folder in folders_to_process:
            subfolder = os.path.join(save_folder, os.path.basename(folder))
            os.makedirs(subfolder)

            overall_luminance = []
            light_values = []
            dark_values = []
            light_amount = []
            dark_amount = []

            for file in os.listdir(folder):
                analysis = ImageAnalyzer(Path(os.path.join(folder, file)))
                overall_lum = np.mean(analysis.arrays["l"]) / np.max(analysis.arrays["l"])
                rel_mel_lum = np.mean(analysis.arrays["l"][np.where(analysis.arrays["black"])]) / np.max(
                    analysis.arrays["l"])
                rel_non_mel_lum = np.mean(analysis.arrays["l"][np.where(~analysis.arrays["black"])]) / np.max(
                    analysis.arrays["l"])

                non_mel_area = analysis.results["colorful"] + analysis.results["white"] + analysis.results["gray"]
                mel_area = analysis.results["black"]

                light_values.append(rel_non_mel_lum)
                dark_values.append(rel_mel_lum)
                light_amount.append(non_mel_area)
                dark_amount.append(mel_area)
                overall_luminance.append(overall_lum)

            luma_values = pd.DataFrame(list(zip(overall_luminance, light_values, dark_values, light_amount, dark_amount)),
                                       columns=["Overall Lum", "Light lum", "Dark lum", "Light prop", "Dark prop"])

            luma_values.to_csv(Path(os.path.join(subfolder, "luma_values.csv")), sep=',')
            print("Finished! results saved to: " + subfolder + "luma_values.csv")

    # TODO: Masked image is based off sklearn, which segments image with recolorization
    # to correct, get unique values from masked array (image with 4 clusters should have 4 unique colors).
    # color to mask, in this case melanistic color, will have lowest value

    def extract_manual_segmentations(self):
        folders_to_process, save_folder = utils.directory_utils.search_existing_directories(self.config,
                                                                            "manual_segment_results",
                                                                            "manually_segmented")

        modified_images_folder = Path(self.config["project_path"]) / "modified"

        def extract(original, masked):
            original_mask = np.array(original[:, :, 0], copy=True, dtype=bool).astype(float)
            masked_mask = np.array(masked[:, :, 0], copy=True, dtype=bool).astype(float)
            masked_mask[original_mask == 0] = np.nan

            '''might switch to luminance calculation here, need to see what other packages do/recommend'''
            # def lum_convert(arr):
            #     red = 0.2126
            #     green = 0.7152
            #     blue = 0.0722
            #     red_ch = np.multiply(arr[:, :, 0], red)
            #     green_ch = np.multiply(arr[:, :, 1], green)
            #     blue_ch = np.multiply(arr[:, :, 2], blue)
            #     lum_arr = np.add(red_ch, green_ch, blue_ch)
            #     return lum_arr

            img_yuv = cv2.cvtColor(original[:, :, [2, 1, 0]], cv2.COLOR_BGR2YUV)
            luma, u, v = cv2.split(img_yuv)
            # calc_luma = lum_convert(original)
            light = luma[np.where(masked_mask == 1)]
            dark = luma[np.where(masked_mask == 0)]

            light_luma = np.mean(light) / np.max(luma)
            dark_luma = np.mean(dark) / np.max(luma)

            image_size = len(original_mask[original_mask == 1])

            light_proportion = len(light) / image_size
            dark_proportion = len(dark) / image_size

            return light_luma, dark_luma, light_proportion, dark_proportion

        for folder in folders_to_process:
            subfolder = os.path.join(save_folder, os.path.basename(folder))
            os.makedirs(subfolder)
            modified_subdirectory = modified_images_folder / os.path.basename(folder)
            modified_images = os.listdir(modified_subdirectory)

            light_values = []
            dark_values = []
            light_amount = []
            dark_amount = []

            for file in os.listdir(folder):
                mod_image_path = [i for i in modified_images if i == file]
                if len(mod_image_path) == 1:
                    mod_image = io.imread(Path(os.path.join(modified_images_folder, mod_image_path[0]))).astype(np.uint8)
                    masked_image = io.imread(Path(os.path.join(folder, file))).astype(np.uint8)
                    light_luma, dark_luma, light_proportion, dark_proportion = extract(mod_image, masked_image)

                    light_values.append(light_luma)
                    dark_values.append(dark_luma)
                    light_amount.append(light_proportion)
                    dark_amount.append(dark_proportion)

                elif len(mod_image_path) > 1:
                    print("multiple files found to match")
                elif len(mod_image_path) < 1:
                    print("No matching images found")

            luma_values = pd.DataFrame(list(zip(light_values, dark_values, light_amount, dark_amount)),
                                       columns=["Light lum", "Dark lum", "Light prop", "Dark prop"])
            if len(luma_values) > 0:
                luma_values.to_csv(Path(os.path.join(subfolder, "luma_values.csv")), sep=',')
                print("Finished! luma_values.csv results saved to: {}".format(save_folder))
            else:
                print("Error: No values calculated for folder {}, output file not saved".format(folder))
