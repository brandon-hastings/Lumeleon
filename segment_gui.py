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


def main(folder, N_cluster, toplevel, uv=False):
    folder = utils.correctPath(folder)
    if os.path.exists(Path(folder) / ".DS_Store"):
        os.remove(Path(folder) / ".DS_Store")

    # for every file in folder run a function that returns the modified image after input to a different function to
    # save
    def choose_clusters(image):
        subroot = tk.Toplevel(toplevel)
        subroot.title(str(file)+" ORIGINAL")

        image = image[:, :, :3]
        pixel_vals = image.reshape((-1, 3))

        fig = Figure(figsize=(8, 8), dpi=100)
        axs = fig.subplots(N_cluster - 1, 2)
        wcss = []
        for i in range(2, N_cluster+1):
            pixel_vals = np.float32(pixel_vals)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, labels, centers = cv2.kmeans(pixel_vals, i, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            wcss.append(compactness)
            # convert data into 8-bit values
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            # reshape data into the original image dimensions
            segmented_image = segmented_data.reshape(image.shape)
            i = i - 2
            axs[i, 0].imshow(image)
            axs[i, 0].axis('off')

            axs[i, 1].imshow(segmented_image)
            axs[i, 1].axis('off')
            axs[i, 1].text(50, 200, str(i+2), c='r', fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=subroot)
        canvas.draw()


        toolbar = NavigationToolbar2Tk(canvas, subroot, pack_toolbar=False)
        toolbar.update()

        canvas.mpl_connect(
            "key_press_event", lambda event: print(f"you pressed {event.key}"))
        canvas.mpl_connect("key_press_event", key_press_handler)

        button_quit = tk.Button(master=subroot, text="Quit", command=subroot.destroy)

        def handle_closing():
            subroot.destroy()
            sys.exit()

        def choose_segments(n_cluster):
            seg_root = tk.Toplevel(subroot)
            seg_root.title(str(file)+" CLUSTERED")
            # image = io.imread(Path(folder) / file).astype(np.uint32)
            n_cluster = int(n_cluster) + 2
            # image = image[:, :, :3]

            m = image.shape[0]
            n = image.shape[1]

            x = np.reshape(image, (m * n, 3))
            model = MiniBatchKMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, batch_size=2048, verbose=0,
                                    compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10,
                                    init_size=None,
                                    n_init=5, reassignment_ratio=0.01).fit(x)

            p = model.predict(x)

            levels = np.unique(p)
            fig = Figure(figsize=(8, 8), dpi=100)
            axs = fig.subplots(len(levels), 2)
            for i in levels:
                b = np.reshape((p == i) * 1, (m, n))

                axs[i, 0].imshow(b)
                axs[i, 0].axis('off')

                axs[i, 1].imshow(image * np.repeat(b[:, :, np.newaxis], 3, axis=2))
                axs[i, 1].axis('off')
                axs[i, 1].text(50, 200, str(i), c='r', fontsize=10)

            canvas = FigureCanvasTkAgg(fig, master=seg_root)  # A tk.DrawingArea.
            canvas.draw()

            # pack_toolbar=False will make it easier to use a layout manager later on.
            toolbar = NavigationToolbar2Tk(canvas, seg_root, pack_toolbar=False)
            toolbar.update()

            canvas.mpl_connect(
                "key_press_event", lambda event: print(f"you pressed {event.key}"))
            canvas.mpl_connect("key_press_event", key_press_handler)

            button_quit = tk.Button(master=seg_root, text="Quit", command=seg_root.destroy)

            def save_mask(selection):
                i = int(selection)
                b = np.reshape((p == i) * 1, (m, n))
                savefile = file[:-4] + '_' + str(selection) + '.png'
                io.imsave(Path(folder) / savefile, (image * np.repeat(b[:, :, np.newaxis], 3, axis=2)).astype(np.uint8))
                # canvas.draw()
                seg_root.quit()
                # seg_root.destroy()

            options = list(range(0, n_cluster))
            # convert to strings
            options = [str(x) for x in options]
            #
            variable = tk.StringVar(seg_root)
            variable.set(options[0])
            selector = tk.OptionMenu(seg_root, variable, *options, command=save_mask)
            button_quit.pack(side=tk.BOTTOM)
            selector.pack(side=tk.BOTTOM)
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # root.protocol("WM_DELETE_WINDOW", handle_closing)
            # root.destroy()
            tk.mainloop()
            seg_root.destroy()
            subroot.quit()

        options = list(range(2, N_cluster+1))
        # convert to strings
        options = [str(x) for x in options]
        #
        variable = tk.StringVar(subroot)
        variable.set(options[0])
        selector = tk.OptionMenu(subroot, variable, *options, command=choose_segments)
        button_quit.pack(side=tk.BOTTOM)
        selector.pack(side=tk.BOTTOM)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        tk.mainloop()
        subroot.destroy()
        # return selector

    for file in fnmatch.filter(sorted(os.listdir(folder)), '*e?.png'):
        I = io.imread(Path(folder) / file).astype(np.uint32)
        if uv:
            I[I == np.nan] = 0
        choose_clusters(I)
