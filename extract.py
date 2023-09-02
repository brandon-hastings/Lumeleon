import cv2
from skimage import io
import matplotlib.pyplot as plt
import segment
from barvocuc import ImageAnalyzer
from tkinter import simpledialog
import os
import fnmatch
import pandas as pd
import numpy as np
import utils
from pathlib import Path


class LuminanceExtraction:
    def __init__(self, config):
        self.config = config

    results = {}

    def automatic_color_segmentation(self):
        folders_to_process, save_folder = utils.search_existing_directories(self.config, self.config["image_folders"],
                                                                            "auto_segment_results",
                                                                            "background_segmented")
        for folder in folders_to_process:
            for file in os.listdir(folder):
        # for file in fnmatch.filter(sorted(os.listdir(self.folder)), '*e?.png'):
                analysis = ImageAnalyzer(Path(os.path.join(folder, file)))
                overall_lum = np.mean(analysis.arrays["l"]) / np.max(analysis.arrays["l"])
                rel_mel_lum = np.mean(analysis.arrays["l"][np.where(analysis.arrays["black"])]) / np.max(
                    analysis.arrays["l"])
                rel_non_mel_lum = np.mean(analysis.arrays["l"][np.where(~analysis.arrays["black"])]) / np.max(
                    analysis.arrays["l"])

                non_mel_area = analysis.results["colorful"] + analysis.results["white"] + analysis.results["gray"]
                mel_area = analysis.results["black"]


    def extract_manual_segmentations(self):
        folders_to_process, save_folder = utils.search_existing_directories(self.config, self.config["image_folders"],
                                                                            "manual_segment_results",
                                                                            "manually_segmented")

        modified_images_folder = Path(self.config["project_path"]) / "modified"

        def extract(original, masked):
            original_mask = np.array(original[:, :, 0], copy=True, dtype=bool).astype(float)
            masked_mask = np.array(masked[:, :, 0], copy=True, dtype=bool).astype(float)
            masked_mask[original_mask == 0] = np.nan

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

            light_values.append(light_luma)
            dark_values.append(dark_luma)
            light_amount.append(light_proportion)
            dark_amount.append(dark_proportion)

        # luma_values = pd.DataFrame(list(zip(light_values, dark_values, light_amount, dark_amount)),
        #                            columns=["Light lum", "Dark lum", "Light prop", "Dark prop"])
        # if len(luma_values) > 0:
        #     luma_values.to_csv(save_folder / "luma_values.csv", sep=',')
        #     print("Finished! results saved to: " + save_folder+ "luma_values.csv")
        # else:
        #     print("Error: Make sure selected file has both original and masked images.")

        for folder in folders_to_process:
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
                    extract(mod_image, masked_image)

                    luma_values = pd.DataFrame(list(zip(light_values, dark_values, light_amount, dark_amount)),
                                               columns=["Light lum", "Dark lum", "Light prop", "Dark prop"])
                    if len(luma_values) > 0:
                        luma_values.to_csv(Path(os.path.join(save_folder, os.path.basename(folder), "luma_values.csv")), sep=',')
                        print("Finished! results saved to: " + save_folder + "luma_values.csv")
                    else:
                        print("Error: Make sure selected file has both original and masked images.")

                elif len(mod_image_path) > 1:
                    print("multiple files found to match")
                elif len(mod_image_path) < 1:
                    print("No matching images found")
