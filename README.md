# Lumeleon v0.1.0

Step 1: Use syntax "python patternAnalysis.py", to launch the user interface, where you can import folders of images to be analyzed in sequence.
## Intensity Matching:
Step 1: Use "Standardize Images", to scale the images to a color standard present in all images. The reference image will be displayed first.

Step 2: Define a reference rectangle on the image by clicking and holding the left mouse button. 

Step 3: If you are happy with the crop press "y", if not, press "n" and you can repeat the process.

![image](https://user-images.githubusercontent.com/69599932/179577716-e53f7de1-a30d-41db-be70-8d5ec646385c.png)

Step 4: This process will be repeated for all the images in the folder.

Result: A new subfolder called "modified" will be created containing all the modified images along with the reference image.



## Segmentation

![download](https://user-images.githubusercontent.com/69599932/179578176-55e46c1f-82c8-49fb-9576-e03a3eef46bc.png)

Step 1: Use "Segment Images" to begin color spot segmentation via k-means clustering. It shows you the resulting masks along with the original images for visual confirmation. At this point select the number of the mask that includes the correct part of the image. In this example with 4 clusters, I select number 0. It saves the segmented image with a "_4" at the end of the name so you will be able to reproduce your work later.

![download](https://user-images.githubusercontent.com/69599932/179578200-f5216b67-b2f6-47d0-9bc0-fa2e7c6f0904.png)

![thumbnail_test_4](https://user-images.githubusercontent.com/69599932/179578776-e1fcc879-f0de-4d41-8162-d351b7b83e4e.png)

Step 2: You can repeat step 1 as many times as you want to go down the hierarchical tree. For example I repeated step 2 for the above image with 3 clusters and the result was:

![thumbnail_image](https://user-images.githubusercontent.com/69599932/179578656-4dbcf7d2-8ccb-4ee3-981a-17909e4d489b.png)

![test_4_3](https://user-images.githubusercontent.com/69599932/179578628-010205bf-24a5-40c3-b144-30a430923a06.png)

## Extraction
After Segmenting, you can extract the luminance values for all images in the selected folder into a csv file.
