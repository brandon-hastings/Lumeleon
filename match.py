import numpy as np
import cv2
import rawpy
import os
from pathlib import Path
import utils


class IntensityMatch:
    def __init__(self, config):
        self.config = utils.read_config(config)
        # self.project_folder = project_folder
        self.image_type = self.config["image_type"]
        # self.sub_directory = utils.make_folder(self.project_folder, folder_name="modified")

    def reference_image(self):
        folders_to_process, save_folder = utils.search_existing_directories(self.config, self.config["image_folders"],
                                                                            "intensity_matched", "images")
        points = []
        lum_values = []
        for folder in folders_to_process:
            for file in os.listdir(folder):
                if file.endswith(self.image_type):
                    raw = rawpy.imread(Path(os.path.join(folder, file)))
                    img = raw.postprocess()
                    clone = img.copy()
                    WinCoords = utils.SavePoints(img)
                    points.append(WinCoords)
                    Y = [points[-1][0][1], points[-1][1][1]]
                    X = [points[-1][0][0], points[-1][1][0]]
                    # convert to HLS color channel
                    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2HLS)
                    # max amount of pixels in luminance channel of cropped section (standard)
                    crop = clone[min(Y):max(Y), min(X):max(X), 1]
                    # take mean of standard and append to list L
                    lum_values.append(crop.mean())
                    if len(lum_values) > 1:
                        # difference between luminance of first (ref image) and last image in L
                        delta = lum_values[-1] - lum_values[0]
                        # adjust pixels in images based on delta value
                        clone[:, :, 1] = clone[:, :, 1] - delta
                        # scale "overwhite" pixels (>255) back to white
                        clone[:, :, 1][clone[:, :, 1] > 255] = 255
                        # convert image back to BGR channel
                        img = cv2.cvtColor(clone, cv2.COLOR_HLS2BGR)
                        # save image in RGB not BGR
                        cv2.imwrite(Path(os.path.join(save_folder, os.path.basename(folder), file[:-4], 'modified.png')), img[:, :, [2, 1, 0]])

                    if len(lum_values) == 1:
                        img = cv2.cvtColor(clone, cv2.COLOR_HLS2BGR)
                        cv2.imwrite(Path(os.path.join(save_folder, os.path.basename(folder), file[:-4], 'ref.png')), img[:, :, [2, 1, 0]])

            # os.chdir("../")

    def scale_image_intensity(self):
        folders_to_process, save_folder = utils.search_existing_directories(self.config, self.config["image_folders"],
                                                                            "intensity_matched", "images")
        points = []
        lum_values = []
        image_dict = {}
        for folder in folders_to_process:
            for file in os.listdir(folder):
                if file.endswith(self.image_type):
                    raw = rawpy.imread(Path(os.path.join(folder, file)))
                    img = raw.postprocess()
                    clone = img.copy()
                    WinCoords = utils.SavePoints(img)
                    points.append(WinCoords)
                    Y = [points[-1][0][1], points[-1][1][1]]
                    X = [points[-1][0][0], points[-1][1][0]]
                    # convert to HLS color channel
                    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2HLS)
                    # max amount of pixels in luminance channel of cropped section (standard)
                    crop = clone[min(Y):max(Y), min(X):max(X), 1]
                    # take mean of standard and append to list L
                    lum_values.append(crop.mean())

                    image_dict[file] = clone

            '''CHANGE TO ADJUST VALUE BASED ON MEAN OF LUMINANCE'''
            delta = np.mean(lum_values)
            for file_name, image in image_dict.items():
            # file_names = list(image_dict.keys())
            # images = list(image_dict.values())
            # for i in range(len(image_dict.values())):
                clone = image
                # adjust pixels in images based on delta value
                clone[:, :, 1] = clone[:, :, 1] - delta
                # scale "overwhite" pixels (>255) back to white
                clone[:, :, 1][clone[:, :, 1] > 255] = 255
                # convert image back to BGR channel
                img = cv2.cvtColor(clone, cv2.COLOR_HLS2BGR)
                # save image in RGB not BGR
                cv2.imwrite(Path(os.path.join(save_folder, os.path.basename(folder), file_name[:-4], 'modified.png')), img[:, :, [2, 1, 0]])

            # os.chdir("../")
