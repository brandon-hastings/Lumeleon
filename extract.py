import cv2
from skimage import io
import matplotlib.pyplot as plt
import os
import fnmatch
import pandas as pd
import numpy as np
import utils


def main(folder, body_part=""):
    print("Extracting luminance values...")
    folder = utils.correctPath(folder)

    masked = fnmatch.filter(sorted(os.listdir(folder)), '*_?.png')
    original = fnmatch.filter(sorted(os.listdir(folder)), '*e?.png')

    def extract(mask, raw):
        light_values = []
        dark_values = []
        light_amount = []
        dark_amount = []
        for image in mask:
            masked = io.imread(folder+'/'+image).astype(np.uint8)
            original = io.imread(folder+'/'+raw[mask.index(image)]).astype(np.uint8)
            original_mask = np.array(original[:, :, 0], copy=True, dtype=bool).astype(float)
            masked_mask = np.array(masked[:, :, 0], copy=True, dtype=bool).astype(float)
            masked_mask[original_mask == 0] = np.nan

            def lum_convert(arr):
                red = 0.2126
                green = 0.7152
                blue = 0.0722
                red_ch = np.multiply(arr[:, :, 0], red)
                green_ch = np.multiply(arr[:, :, 1], green)
                blue_ch = np.multiply(arr[:, :, 2], blue)
                lum_arr = np.add(red_ch, green_ch, blue_ch)
                return lum_arr


            img_yuv = cv2.cvtColor(original[:,:,[2,1,0]], cv2.COLOR_BGR2YUV)
            luma, u, v = cv2.split(img_yuv)
            # calc_luma = lum_convert(original)
            light = luma[np.where(masked_mask == 1)]
            dark = luma[np.where(masked_mask == 0)]
            # print(light)
            # print(luma[masked_mask == 1])
            # print(dark)
            # print(luma[masked_mask == 0])

            light_luma = np.mean(light) / np.max(luma)
            dark_luma = np.mean(dark) / np.max(luma)

            image_size = len(original_mask[original_mask == 1])

            light_proportion = len(light) / image_size
            dark_proportion = len(dark) / image_size

            light_values.append(light_luma)
            dark_values.append(dark_luma)
            light_amount.append(light_proportion)
            dark_amount.append(dark_proportion)

        luma_values = pd.DataFrame(list(zip(light_values, dark_values, light_amount, dark_amount)),
                                   columns=["Light lum", "Dark lum", "Light prop", "Dark prop"])
        if len(luma_values) > 0:
            luma_values.to_csv(folder + "/"+body_part+"_luma_values.csv", sep=',')
            print("Finished! results saved to: "+folder+"/"+body_part+"_luma_values.csv")
        else:
            print("Error: Make sure selected file has both original and masked images.")

    extract(masked, original)
