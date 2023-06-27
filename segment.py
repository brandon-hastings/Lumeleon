from skimage import io
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import sys
import cv2


def main(file=None, N_cluster=None):
    trigger = False
    if None not in (file, N_cluster):
        file = file
        N_cluster = N_cluster
        trigger = True
    else:
        if len(sys.argv) != 3:
            print("usage:python match.py image N_clusters")
            return
    
        file = sys.argv[1]
        N_cluster = int(sys.argv[2])
    
    I = io.imread(file).astype(np.uint8)
    
    I = I[:,:,:3]
    
#    I = I - np.min(I)
#    I = I / np.max(I)
    
    m = I.shape[0]
    n = I.shape[1]
    
    
    x = np.reshape(I, (m*n, 3))
    model = MiniBatchKMeans(n_clusters= N_cluster, init='k-means++', max_iter=100, batch_size=2048, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=5, reassignment_ratio=0.01).fit(x)
    
    p = model.predict(x)
    
    levels = np.unique(p)
    fig = plt.figure(figsize=(8,8), dpi=100)
    axs = fig.subplots(len(levels),2)
    for i in levels:
        b = np.reshape((p==i)*1,(m,n))
        
        axs[i,0].imshow(b)
        axs[i,0].axis('off')
        
        axs[i,1].imshow(I * np.repeat(b[:, :, np.newaxis],3,axis=2))
        axs[i,1].axis('off')
        axs[i,1].text(50, 200, str(i), c='r', fontsize=10)

    if trigger is False:
        fig.show()
        
        key = ' '
        keys = [str(i) for i in range(N_cluster)] + ['q']
        while key not in keys:
            key = input('Which mask to save? or [q] to quit.  ')
            
        if key == 'q':
            return
        i = int(key)
        b = np.reshape((p==i)*1,(m,n))
        io.imsave(file[:-4]+'_'+str(N_cluster)+'.png', (I * np.repeat(b[:, :, np.newaxis],3,axis=2)).astype(np.uint8))
        # io.imshow(file[:-4]+'_'+str(N_cluster)+'.png')

    elif trigger is True:
        return [fig, p, m, n]


def cv_segment(image, N_cluster):

    # image = image[:, :, :3]
    pixel_vals = image.reshape((-1, 3))

    fig = Figure(figsize=(8, 8), dpi=100)
    axs = fig.subplots(N_cluster - 1, 2)
    # list for use in elbow graph
    # wcss = []
    for i in range(2, N_cluster+1):
        pixel_vals = np.float32(pixel_vals)

        # using cv2 kmeans clustering here as it gives better visual representation of image clustering to user than sklearn
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(pixel_vals, i, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # wcss.append(compactness)
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
    return fig
    
if __name__ == '__main__':
    main()