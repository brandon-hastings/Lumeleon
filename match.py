import cv2
import rawpy
import os
import shutil
import sys
import numpy as np
# import custom utils file
import utils


def main(folder=None, ref_image=True):
    # check if argument is called from gui
    if folder is not None:
        # TODO: change to reflect "ref_image" argument in GUI call
        argument = utils.correctPath(folder)
    # if not, get command line argument as folder path
    else:
        # TODO: change to reflect "ref_image" argument in command line call
        argument = sys.argv[1]
        if len(argument) != 2:
            print("usage:python match.py ImageFolder")
            return

    os.chdir(argument)
    if 'modified' in os.listdir():
        shutil.rmtree('modified')
    os.makedirs('modified')
    
    points = []
    L=[]
    image_dict = {}
    for file in os.listdir():
        if file.endswith('.CR2'):
            raw = rawpy.imread(file)
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
            L.append(crop.mean())

            image_dict[file] = clone

            if ref_image is True:
                if len(L)>1:
                    # difference between luminance of first (ref image) and last image in L
                    delta = L[-1] - L[0]
                    # adjust pixels in images based on delta value
                    clone[:, :, 1] = clone[:, :, 1] - delta
                    # scale "overwhite" pixels (>255) back to white
                    clone[:, :, 1][clone[:, :, 1]>255] = 255
                    # convert image back to BGR channel
                    img = cv2.cvtColor(clone, cv2.COLOR_HLS2BGR)
                    # save image in RGB not BGR
                    cv2.imwrite('modified/' + file[:-4] + 'modified.png', img[:, :, [2, 1, 0]])

                if len(L)==1:
                    img = cv2.cvtColor(clone, cv2.COLOR_HLS2BGR)
                    cv2.imwrite('modified/' + file[:-4] + 'ref.png', img[:, :, [2, 1, 0]])

    if ref_image is False:
        '''CHANGE TO ADJUST VALUE BASED ON MEAN OF LUMINANCE'''
        delta = np.mean(L)
        file_names = list(image_dict.keys())
        images = list(image_dict.values())
        for i in range(len(image_dict.values())):
            clone = images[i]
            # adjust pixels in images based on delta value
            clone[:, :, 1] = clone[:, :, 1] - delta
            # scale "overwhite" pixels (>255) back to white
            clone[:, :, 1][clone[:, :, 1] > 255] = 255
            # convert image back to BGR channel
            img = cv2.cvtColor(clone, cv2.COLOR_HLS2BGR)
            # save image in RGB not BGR
            cv2.imwrite('modified/' + file_names[i][:-4] + 'modified.png', img[:, :, [2, 1, 0]])

    os.chdir("../")


if __name__ == '__main__':
    main()
