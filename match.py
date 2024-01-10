import numpy as np
import cv2
import rawpy
import os
from pathlib import Path
import utils.directory_utils
import utils.yaml_utils

'''functions used for area selection in image intesity matching'''


def click_and_crop(event, x, y, flags, param):
    global img,refPt, cropping
    if len(refPt) == 2:
        return
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            cropping = False
            cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 5)
            cv2.imshow("image", img[:,:,[2,1,0]])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Proceed? [y/n]', (50,150),
                  font, 5, (255, 0, 0), 8)


def SavePoints(I):
    global img,refPt, cropping
    refPt = []
    cropping = False
    img = I
    clone = img.copy()

    scale = 0.25
    h = int(scale * img.shape[0])
    w = int(scale * img.shape[1])

    while True:
        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('image', img[:,:,[2,1,0]])
        cv2.resizeWindow('image', w, h)

        cv2.setMouseCallback('image', click_and_crop)


        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            img = clone.copy()
            refPt = []

        elif key == ord("y"):
            cv2.destroyAllWindows()
            break
    return refPt


class IntensityMatch:
    def __init__(self, config):
        self.config = utils.yaml_utils.read_config(config)
        # self.project_folder = project_folder
        self.image_type = self.config["image_type"]
        # self.sub_directory = utils.make_folder(self.project_folder, folder_name="modified")

    def reference_image(self):
        folders_to_process, save_folder = utils.directory_utils.search_existing_directories(self.config, "intensity_matched", "original_images")
        print(folders_to_process)
        print(save_folder)
        points = []
        lum_values = []
        for folder in folders_to_process:
            lum_values.clear()
            subfolder = os.path.join(save_folder, os.path.basename(folder))
            os.makedirs(subfolder)
            for file in os.listdir(folder):
                points.clear()
                print(file)
                if file.lower().endswith(self.image_type):
                    raw = rawpy.imread(str(Path(os.path.join(folder, file))))
                    img = raw.postprocess()
                    clone = img.copy()
                    WinCoords = SavePoints(img)
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
                        cv2.imwrite(str(Path(os.path.join(subfolder, file[:-4]+'modified.png'))), img[:, :, [2, 1, 0]])

                    if len(lum_values) == 1:
                        img = cv2.cvtColor(clone, cv2.COLOR_HLS2BGR)
                        cv2.imwrite(str(Path(os.path.join(subfolder, file[:-4]+'ref.png'))), img[:, :, [2, 1, 0]])

            # os.chdir("../")

    def scale_image_intensity(self):
        folders_to_process, save_folder = utils.directory_utils.search_existing_directories(self.config,
                                                                            "intensity_matched", "original_images")
        points = []
        lum_values = []
        image_dict = {}
        for folder in folders_to_process:
            lum_values.clear()
            subfolder = os.path.join(save_folder, os.path.basename(folder))
            os.makedirs(subfolder)
            for file in os.listdir(folder):
                points.clear()
                if file.lower().endswith(self.image_type):
                    raw = rawpy.imread(str(Path(os.path.join(folder, file))))
                    img = raw.postprocess()
                    clone = img.copy()
                    WinCoords = SavePoints(img)
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
                cv2.imwrite(str(Path(os.path.join(subfolder, file_name[:-4]+'modified.png'))), img[:, :, [2, 1, 0]])

            # os.chdir("../")
