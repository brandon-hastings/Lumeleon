import os
import sys
import tkinter as tk
import cv2
import fnmatch
from pathlib import Path
from skimage import io
from sklearn.cluster import MiniBatchKMeans
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import utils
import segment


def main(folder, N_cluster, toplevel, uv=False):
    folder = utils.correctPath(folder)
    if os.path.exists(Path(folder) / ".DS_Store"):
        os.remove(Path(folder) / ".DS_Store")

    # for every file in folder run a function that returns the modified image after input to a different function to
    # save
    def choose_clusters(file_path):
        subroot = tk.Toplevel(toplevel)
        subroot.title(str(file)+" ORIGINAL")

        image = io.imread(file_path).astype(np.uint32)
        if uv:
            image[image == np.nan] = 0

        image = image[:, :, :3]

        # segment.cv_segment(image=image, N_cluster=N_cluster)
        # image = image[:, :, :3]
        # pixel_vals = image.reshape((-1, 3))

        # fig = Figure(figsize=(8, 8), dpi=100)
        # axs = fig.subplots(N_cluster - 1, 2)
        # # list for use in elbow graph
        # # wcss = []
        # for i in range(2, N_cluster+1):
        #     pixel_vals = np.float32(pixel_vals)

        #     # using cv2 kmeans clustering here as it gives better visual representation of image clustering to user than sklearn
        #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #     compactness, labels, centers = cv2.kmeans(pixel_vals, i, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #     # wcss.append(compactness)
        #     # convert data into 8-bit values
        #     centers = np.uint8(centers)
        #     segmented_data = centers[labels.flatten()]
        #     # reshape data into the original image dimensions
        #     segmented_image = segmented_data.reshape(image.shape)
        #     i = i - 2
        #     axs[i, 0].imshow(image)
        #     axs[i, 0].axis('off')

        #     axs[i, 1].imshow(segmented_image)
        #     axs[i, 1].axis('off')
        #     axs[i, 1].text(50, 200, str(i+2), c='r', fontsize=10)


        def choose_segments(n_cluster):
            seg_root = tk.Toplevel(subroot)
            seg_root.title(str(file)+" CLUSTERED")
            # image = io.imread(Path(folder) / file).astype(np.uint32)
            n_cluster = int(n_cluster) + 2
            fig, p, m, n = segment.main(file_path, N_cluster)

            # button_quit, toolbar, canvas = utils.pop_up(fig, seg_root)


            def save_mask(selection):
                i = int(selection)
                b = np.reshape((p == i) * 1, (m, n))
                savefile = file[:-4] + '_' + str(selection) + '.png'
                io.imsave(Path(folder) / savefile, (image * np.repeat(b[:, :, np.newaxis], 3, axis=2)).astype(np.uint8))
                # canvas.draw()
                seg_root.quit()
                # seg_root.destroy()

            utils.image_selection(seg_root, n_cluster, save_mask, utils.pop_up(fig, seg_root), end=-1)


            tk.mainloop()
            seg_root.destroy()
            subroot.quit()

        utils.image_selection(subroot, N_cluster, choose_segments, utils.pop_up(segment.cv_segment(image=image[:, :, :3], N_cluster=N_cluster), subroot), start=2, end=1)

        tk.mainloop()
        subroot.destroy()


    for file in fnmatch.filter(sorted(os.listdir(folder)), '*e?.png'):
        file_path = Path(folder) / file
        choose_clusters(file_path)
