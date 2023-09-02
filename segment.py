import os
from skimage import io
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from tkinter import simpledialog
from pathlib import Path
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import numpy as np
import fnmatch
import cv2
import tkinter as tk
import shutil

import utils

'''MAIN SEGMENTATION METHOD VIA KMEANS SKIMAGE, CALLABLE FROM COMMAND LINE OR GUI'''


class Segmentation:
    def __init__(self, config, folder=None, n_cluster=None, toplevel=None):
        self.folder = folder if folder is not None else TypeError
        self.toplevel = toplevel if toplevel is not None else TypeError
        if n_cluster is None:
            self.n_cluster = simpledialog.askinteger("Input", "Number of clusters to use:",
                                                     parent=self.toplevel, minvalue=0, maxvalue=10)
        elif type(n_cluster) is int:
            self.n_cluster = n_cluster
        else:
            print("n_cluster setting error")
        self.child_root = tk.Toplevel(toplevel)

        self.config = utils.read_config(config)
        # self.image_directories = self.config["image_folders"]

        self.file_list = None
        self.file = None
        if os.path.isdir(folder):
            self.directory = Path(folder)
            self.file_list = [Path(folder) / i for i in fnmatch.filter(sorted(os.listdir(folder)), '*e?.png')]
        elif os.path.isfile(folder):
            self.directory = os.path.dirname(folder)
            self.file = [folder]

    # segmented_image_dict = {}

    # def save_images(self, save_folder):
    #     for key, value in self.segmented_image_dict.items():
    #         base_name = os.path.basename(key)
    #         savefile = os.path.join(save_folder, base_name[:-4]) + '_' + str(self.n_cluster) + '.png'
    #         io.imsave(Path(savefile), value)


    '''create figure canvas to be used with tkinter window'''

    @staticmethod
    def pop_up(fig, subroot):
        canvas = FigureCanvasTkAgg(fig, master=subroot)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, subroot, pack_toolbar=False)
        toolbar.update()

        canvas.mpl_connect(
            "key_press_event", lambda event: print(f"you pressed {event.key}"))
        canvas.mpl_connect("key_press_event", key_press_handler)

        button_quit = tk.Button(master=subroot, text="Quit", command=subroot.destroy)
        return [button_quit, toolbar, canvas]

    '''toolbar functions for selecting image'''

    @staticmethod
    def image_selection(subroot, n_cluster, command, canvas, start=1, end=1):
        options = list(range(start, n_cluster + end))
        # convert to strings
        options = [str(x) for x in options]
        #
        variable = tk.StringVar(subroot)
        variable.set(options[0])
        selector = tk.OptionMenu(subroot, variable, *options, command=command)
        canvas[0].pack(side=tk.BOTTOM)
        selector.pack(side=tk.BOTTOM)
        canvas[1].pack(side=tk.BOTTOM, fill=tk.X)
        canvas[2].get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def background_segmentation(self):
        folders_to_process, save_folder = utils.search_existing_directories(self.config, self.config["image_folders"],
                                                                            "background_segmented", "intensity_matched")

        for folder in folders_to_process:
            for file in os.listdir(folder):
                image = io.imread(os.path.join(folder, file)).astype(np.uint8)

                image = image[:, :, :3]

                m = image.shape[0]
                n = image.shape[1]

                # reshape image and apply kMeans
                x = np.reshape(image, (m * n, 3))
                model = MiniBatchKMeans(n_clusters=self.n_cluster, init='k-means++', max_iter=100, batch_size=2048, verbose=0,
                                        compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None,
                                        n_init=5, reassignment_ratio=0.01).fit(x)

                p = model.predict(x)

                # plot images side by side, original on left, masked image on right. One set for each cluster
                levels = np.unique(p)
                fig = plt.figure(figsize=(8, 8), dpi=100)
                axs = fig.subplots(len(levels), 2)
                for i in levels:
                    b = np.reshape((p == i) * 1, (m, n))

                    axs[i, 0].imshow(b)
                    # axs[i, 0].imshow(image)
                    axs[i, 0].axis('off')

                    axs[i, 1].imshow(image * np.repeat(b[:, :, np.newaxis], 3, axis=2))
                    axs[i, 1].axis('off')
                    axs[i, 1].text(50, 200, str(i+1), c='r', fontsize=10)

                def save_mask(selection):
                    # new_folder = "background_segmented"
                    # if new_folder in os.listdir(self.directory):
                    #     shutil.rmtree(new_folder)
                    # new_dir = os.path.join(self.directory, new_folder)
                    i = int(selection)-1
                    b = np.reshape((p == i) * 1, (m, n))
                    # folder_name = os.path.basename(folder)
                    savefile = os.path.join(save_folder, os.path.basename(folder), os.path.basename(file[:-4]), '_'
                                            + str(self.n_cluster) + '_' + str(selection) + '.png')
                    io.imsave(Path(savefile), (image * np.repeat(b[:, :, np.newaxis], 3, axis=2)).astype(np.uint8))
                    self.child_root.quit()

                self.image_selection(self.child_root, self.n_cluster, save_mask, self.pop_up(fig, self.child_root))

    #         add tkinter question to ask if user would like to proceed with segmentation of next folder
    #           if yes, countinue. if no, break

    '''using cv2 kmeans clustering here as it gives better visual representation of
        image clustering to user than sklearn'''

    def manual_pattern_segmentation(self):
        folders_to_process, save_folder = utils.search_existing_directories(self.config, self.config["image_folders"],
                                                                            "manually_segmented", "intensity_matched")

        # dictionary to collect chosen segmented images. Saved in one process at end of function
        segmented_image_dict = {}

        # function to save images from segmented image dictionary
        def save_images():
            for key, value in segmented_image_dict.items():
                # commented out as basename is taken as key into dictionary
                # base_name = os.path.basename(key)
                savefile = os.path.join(save_folder, key[:-4], '_' + str(self.n_cluster) + '.png')
                io.imsave(Path(savefile), value)

        for folder in folders_to_process:
            files = os.listdir(folder)
            for j in range(len(files)):
                image = io.imread(Path(os.path.join(folder, files[j]))).astype(np.uint8)
                image = image[:, :, :3]

                # list for use in elbow graph
                # wcss = []

                m = image.shape[0]
                n = image.shape[1]

                # reshape image and apply kMeans
                x = np.reshape(image, (m * n, 3))
                pixel_vals = np.float32(x)

                fig = plt.figure(figsize=(8, 8), dpi=100)
                axs = fig.subplots(self.n_cluster, 2)

                image_dict = {}

                for k in range(1, self.n_cluster+1):
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    compactness, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                    # convert data into 8-bit values
                    centers = np.uint8(centers)
                    segmented_data = centers[labels.flatten()]
                    # reshape data into the original image dimensions
                    segmented_image = segmented_data.reshape(image.shape)

                    image_dict[k] = segmented_image

                    i = k - 1
                    axs[i, 0].imshow(image)
                    axs[i, 0].axis('off')

                    axs[i, 1].imshow(segmented_image)
                    axs[i, 1].axis('off')
                    axs[i, 1].text(50, 200, str(k), c='r', fontsize=10)

                def select_mask(selection):
                    sel = int(selection)
                    segmented_image_dict[os.path.join(os.path.basename(folder), files[j])] = image_dict[sel-1]
                    if j == len(files) - 1:
                        save_images()
                    self.child_root.quit()

                self.image_selection(self.child_root, self.n_cluster, select_mask, self.pop_up(fig, self.child_root), start=1, end=1)
