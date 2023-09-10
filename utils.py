import os.path
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import cv2
import tkinter as tk
import shutil
import ruamel.yaml.representer
import yaml
from ruamel.yaml import YAML
from pathlib import Path

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


'''make subdirectories'''


def make_folder(project_folder, folder_name):
    new_folder = folder_name
    if new_folder in os.listdir(project_folder):
        shutil.rmtree(new_folder)
    new_dir = os.path.join(project_folder, new_folder)
    os.makedirs(new_dir)
    return new_dir


'''find image directories that still need processed in selected step'''


def search_existing_directories(config, image_directories, new_folder_name, search_folder_name):
    working_folder = Path(config["project_path"]) / search_folder_name
    save_folder = Path(config["project_path"]) / new_folder_name

    if os.path.exists(working_folder):
        folders_to_process = []
        if os.path.exists(save_folder):
            processed_folders = [os.path.basename(i) for i in os.listdir(save_folder) if os.path.isdir(i)]
            for i in image_directories:
                if os.path.basename(i) not in processed_folders:
                    folders_to_process.append(working_folder / os.path.basename(i))
        else:
            os.makedirs(save_folder)
            for i in image_directories:
                folders_to_process.append(working_folder / os.path.basename(i))
        return folders_to_process, save_folder
    else:
        # TODO: Make more case specific error messages, as some steps (background segmentation) are optional
        print("Previous step not completed")
        raise FileNotFoundError(
            "Directory {} was not found, perhaps the previous step has not been completed?".format(working_folder)
        )


'''find subdirectories in given folder'''


def find_directories(folder):
    return [i for i in os.scandir(folder) if os.path.isdir(i)]


'''remove trailing slash in case of user input, not a problem with file selection'''


def correctPath(path):
    return os.path.normpath(path)


'''create figure canvas to be used with tkinter window'''


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


def image_selection(subroot, n_cluster, command, canvas, start=0, end=0):
    options = list(range(start, n_cluster+end))
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


'''functions for creating and saving yaml config files (Mostly taken from DeepLabCut)'''


def create_config():
    yaml_str = """\
    # Project definitions (do not edit)
        Task:
        scorer:
        date:
        \n
    # Project path (change when moving around)
        project_path:
        image_folders:
        \n
    # Image type of raw images
        image_type:
        """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile


def read_config(configname):
    """
    Reads structured config file defining a project.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = os.path.dirname(configname)
                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                        err.args[2]
                        == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg


def write_config(configname, cfg):
    """
    Write structured config file.
    """
    with open(configname, "w+") as cf:
        cfg_file, ruamelFile = create_config()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]
        ruamelFile.dump(cfg_file, cf)
