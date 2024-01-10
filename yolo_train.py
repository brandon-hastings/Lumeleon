import random
import pandas as pd
from utils import yaml_utils
from yolov5 import train
import shutil
import numpy as np
import yaml
import cv2
import os

'''function to take csv from bounding box labeling toolbox, melt the dataframe, and create lists from remaining col'''


def preprocess_csv(csv_file):
    # every column after index is a class's bounding box coordinates in the image for that row
    if not isinstance(csv_file, pd.DataFrame) and str(csv_file).endswith("csv"):
        csv_file = pd.read_csv(csv_file)
    # get column names for a later function's use
    classes = csv_file.columns.values.tolist()
    data = csv_file.to_dict('index')
    # begin some dictionary gymnastics
    for key, value in data.items():
        # overwrite original key with values as tuples
        data[key] = [(cls, coord) for cls, coord in value.items()]

    return data, classes


'''split images and labels into test and train sets based on test_split defined in train_network call.
Input is dictionary with image path as key and labels in yolov5 format as values.
'''


def format_yolo_training_data(data_dict):
    label_metadata = {}
    # images = []

    def midpoint(x1, x2, y1, y2):
        return (x1 + x2) / 2, (y1 + y2) / 2

    # key - image_paths - is image path to each image. values are a list holding tuples of the class and the bbox
    # coordinates in a tuple
    for image_path, values in data_dict.items():
        yolo_format = []
        # read in image that is dictionary key
        image = cv2.imread(image_path)
        # get height and width
        (h, w) = image.shape[:2]
        # for every class, coordinate list in values
        for value in values:
            # object class in the 1st position is the label
            obj_class = value
            # bbox coordinates are in the second position
            coordinates = values[value]
            # scale the coords relative to image height and width
            # TODO: Should image be resized before scaling?
            ul_x = coordinates[0] / w
            ul_y = coordinates[1] / h
            br_x = coordinates[2] / w
            br_y = coordinates[3] / h

            # find center of bbox
            center = midpoint(ul_x, br_x, ul_y, br_y)

            # scale the center to height and width
            scaled_center_x = center[0]
            scaled_center_y = center[1]

            # scale height and width of bbox
            # TODO: Rounding used to pass unit test case. Evaluate need for rounding
            scaled_object_width = round(abs(ul_x - br_x), 2)
            scaled_object_height = round(abs(ul_y - br_y), 2)

            # store obj class, scaled center points, and scaled bbox width height in tuple
            yolo_formatted = obj_class, scaled_center_x, scaled_center_y, scaled_object_width, scaled_object_height
            yolo_format.append(yolo_formatted)

        # add to dictionary for each image to later add each key as line in text file for each image
        label_metadata[image_path] = yolo_format

        return label_metadata


'''Create train and validation sets'''


def train_validation_split(label_metadata_dict, test_split):
    # split images and labels into test and train split
    correct_split = True
    while correct_split:
        # number of elements to pull from dict
        train_num = int(np.round((len(label_metadata_dict) * test_split)))
        random_keys = random.choices(list(label_metadata_dict), k=train_num)
        if len(random_keys) != len(set(random_keys)):
            correct_split = True
        elif len(random_keys) == len(set(random_keys)):
            correct_split = False

    # init new dict for training set
    validation_set = {}

    # remove random_keys from label_metadata_dict and move to new dict using pop
    # random_keys dict will always be defined after splitting process
    for key in random_keys:
        validation_set[key] = label_metadata_dict.pop(key, None)
    # rename original dict with values popped to train_set for clarity
    train_set = label_metadata_dict

    return train_set, validation_set


'''Create file structure needed for yolov5 training'''


def yolo_file_structuring(config_file, train_set, validation_set, classes):
    config = yaml_utils.read_config(config_file)

    # create folder structure for bounding box machine learning training
    bbox_project_folder = os.path.join(config["project_path"], "bounding_boxes")
    os.mkdir(bbox_project_folder)

    # create needed subdirectories from list. Could use makedirs but making individually felt safer
    subdirectories = ["test", "train", "valid"]
    for i in subdirectories:
        new_dir = os.path.join(bbox_project_folder, i)
        os.mkdir(new_dir)
        os.mkdir(os.path.join(new_dir, "images"))
        os.mkdir(os.path.join(new_dir, "labels"))

    # create config file for yolo training
    bbox_config = {'path': bbox_project_folder,
                   'train': os.path.join(bbox_project_folder, 'train'),
                   'val': os.path.join(bbox_project_folder, 'valid'),
                   'nc': len(classes),
                   'names': classes}

    # save config file
    data_yaml_file = os.path.join(bbox_project_folder, "data.yaml")
    with open(data_yaml_file, "w") as file:
        yaml.dump(bbox_config, file, default_flow_style=False)
        file.close()

    # organize images and label data into appropriate folders created above
    def write_metadata(dictionary, folder):
        for keys, values in dictionary.items():
            # save image to images directory via shutil copyfile
            images_output_file = os.path.join(bbox_config[folder], "images", os.path.basename(keys))
            shutil.copyfile(keys, images_output_file, follow_symlinks=False)
            # save dictionary values into metadata text file for yolo training
            labels_output_file = os.path.join(bbox_config[folder], "labels", str(os.path.basename(keys)).split('.')[0] + "_metadata.txt")
            with open(labels_output_file, "x") as s:
                for v in values:
                    s.write(f"{v}\n")
                s.close()

    write_metadata(train_set, "train")
    write_metadata(validation_set, "val")

    return data_yaml_file

'''Take in csv file of labeled data and flatten multiple classes into class column. Reformat labeled data to
yolov5 standards, split labeled data into test and train sets, then run the model with specified parameters'''


def train_network(config_file, csv_file, test_split=.2, model="yolov5s", size=640, batch_size=32, epochs=20):
    (data, classes) = preprocess_csv(csv_file=csv_file)
    label_metadata = format_yolo_training_data(data_dict=data)
    train_set, validation_set = train_validation_split(label_metadata_dict=label_metadata,
                                                       test_split=test_split)
    data_file = yolo_file_structuring(config_file=config_file, train_set=train_set,
                                      validation_set=validation_set, classes=classes)

    train.run(data=data_file, model=model, imagesz=size, batch=batch_size, epochs=epochs, workers=1,
              project="yolov5_bbox", name=f"{model}_size{size}_epochs{epochs}_batch{batch_size}")
