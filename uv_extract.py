import cv2
from skimage import io
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
        uv_values = []
        norm_values = []
        uv_amount = []
        norm_amount = []
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


            # img_yuv = cv2.cvtColor(original[:,:,[2,1,0]], cv2.COLOR_BGR2YUV)
            # red, u, v = cv2.split(img_yuv)
            red = original[:, :, 0]
            # calc_luma = lum_convert(original)
            light = red[np.where(masked_mask == 1)]
            dark = red[np.where(masked_mask == 0)]
            # print(light)
            # print(luma[masked_mask == 1])
            # print(dark)
            # print(luma[masked_mask == 0])

            light_luma = np.mean(light) / np.max(red)
            dark_luma = np.mean(dark) / np.max(red)

            image_size = len(original_mask[original_mask == 1])

            light_proportion = len(light) / image_size
            dark_proportion = len(dark) / image_size

            uv_values.append(light_luma)
            norm_values.append(dark_luma)
            uv_amount.append(light_proportion)
            norm_amount.append(dark_proportion)

        uv_extraction = pd.DataFrame(list(zip(uv_values, norm_values, uv_amount, norm_amount)),
                                   columns=["Light lum", "Dark lum", "Light prop", "Dark prop"])
        print(uv_extraction)
        if len(uv_extraction) > 0:
            uv_extraction.to_csv(folder + "/uv_values.csv", sep=',')
            print("Finished! results saved to: "+folder+"/uv_values.csv")
        else:
            print("Error: Make sure selected file has both original and masked images.")

    extract(masked, original)
