import os.path

import cv2

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


'''remove trailing slash in case of user input, not a problem with file selection'''
def correctPath(path):
    return os.path.normpath(path)
