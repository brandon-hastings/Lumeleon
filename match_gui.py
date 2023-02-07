import cv2
import rawpy
import os
import shutil
import sys

import utils


def main(folder):
    folder = utils.correctPath(folder)

    os.chdir(folder)
    if 'modified' in os.listdir():
        shutil.rmtree('modified')
    os.makedirs('modified')
    
    points = []
    L=[]
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
            
    os.chdir("../")


if __name__ == '__main__':
    main()
