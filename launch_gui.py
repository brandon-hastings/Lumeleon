import os
import tkinter as tk
from tkinter import ttk
import match
import segment
import uv_extract
import extract
import main_window
from new_project import new_project
from tkinter import filedialog
from tkinter import simpledialog
from pathlib import Path


# window = tk.Tk()
# window.title("Lumeleon")

#
# standardframe = tk.Frame(window)
# standardframe.grid(row=0, column=0)
# cropframe = tk.Frame(window)
# cropframe.grid(row=1, column=0)
# maskframe = tk.Frame(window)
# maskframe.grid(row=2, column=0)
# extractframe = tk.Frame(window)
# extractframe.grid(row=3, column=0)


class NewProjectFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        options = {'padx': 5, 'pady': 0}

        self.project_name_label = tk.Label(self, text="Project name")
        self.project_name_label.grid(column=0, row=0, sticky='e', **options)
        self.project_name_variable = tk.StringVar()
        self.project_name_entry = tk.Entry(self, textvariable=self.project_name_variable, takefocus=True)
        self.project_name_entry.grid(column=1, row=0, sticky='w', **options)

        self.experimenter_name_label = tk.Label(self, text="Experimenter name")
        self.experimenter_name_label.grid(column=0, row=1, sticky='e', **options)
        self.experimenter_name_variable = tk.StringVar()
        self.experimenter_name_entry = tk.Entry(self, textvariable=self.experimenter_name_variable)
        self.experimenter_name_entry.grid(column=1, row=1, sticky='w', **options)

        self.select_images_label = tk.Label(self, text="select image folder(s)")
        self.select_images_label.grid(column=0, row=2, sticky='e', **options)
        self.select_images_button = tk.Button(self, text="Add image folders to project",
                                              command=self.select_image_paths)
        self.select_images_button.grid(column=1, row=2, sticky='w', **options)

        self.image_type_label = tk.Label(self, text="Select/input raw file type")
        self.image_type_label.grid(column=0, row=3, sticky='e', **options)
        self.image_type_box = ttk.Combobox(self, values=["cr2", "cr3", "dng", "nef", "nrw"])
        self.image_type_box.grid(column=1, row=3, sticky='w', **options)

        self.grid(column=0, row=2, **options)

    images_paths = None

    '''tkinter askdirectory can only handle one directory selection. It is crucial that if multiple image folders
    are to be input that they are stored in a single overarching folder. This function will process the sub-folders
    if present.
    Input:
    A SINGLE directory folder where images or sub-folders containing images to be used in the project exist.
    Output:
    A single directory path inside list or a list of directory paths, depending on if the input directory has sub-folders.'''

    def select_image_paths(self):
        main_dir = filedialog.askdirectory()
        sub_dirs = [i for i in os.listdir(main_dir) if os.path.isdir(i)]
        if len(sub_dirs) > 0:
            self.images_paths = [os.path.join(main_dir, i) for i in sub_dirs]
        elif len(sub_dirs) == 0:
            self.images_paths = [main_dir]

    def reset(self):
        self.project_name_entry.delete(0, "end")
        self.experimenter_name_entry.delete(0, 'end')
        self.images_paths = None


class LoadProjectFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        options = {'padx': 5, 'pady': 0}

        self.load_config_label = tk.Label(self, text="Load config file")
        self.load_config_label.grid(column=0, row=1, sticky='w', **options)

        # self.config_entry_var = tk.StringVar()
        # self.load_config_entry = tk.Entry(self, textvariable=self.config_entry_var, state='disabled', width=50)
        # self.load_config_entry.grid(column=1, row=1)
        self.load_config_button = tk.Button(self, text="Select config file", command=self.select_config_path)
        self.load_config_button.grid(column=1, row=1, sticky='e', **options)
        self.load_config_button.focus()

        self.grid(column=0, row=2, **options)

    config_path = None

    def select_config_path(self):
        filetypes = [('yaml files', '.yaml .yml')]
        config_string = filedialog.askopenfilename(filetypes=filetypes)
        # self.load_config_entry.configure(state='normal')
        # self.config_entry_var.set(config_string)
        self.config_path = Path(config_string)

    def reset(self):
        self.config_path = None


class ControlFrame(ttk.LabelFrame):
    def __init__(self, container):
        super().__init__(container)
        self.container = container

        self['text'] = 'Create or load project'

        self.selected_value = tk.IntVar()
        # new project button
        tk.Radiobutton(self, text="New Project", value=0, variable=self.selected_value, command=self.change_frame).grid(
            column=0, row=0, padx=5, pady=5)
        # load project button
        tk.Radiobutton(self, text="Load Project", value=1, variable=self.selected_value, command=self.change_frame
                       ).grid(column=1, row=0, padx=5, pady=5)

        self.grid(column=0, row=1, padx=5, pady=5, sticky='ew')

        self.frames = {0: NewProjectFrame(container), 1: LoadProjectFrame(container)}

        self.change_frame()

        self.button = tk.Button(self, text="Continue", command=self.proceed)
        self.button.grid(column=0, row=3, sticky='e')

    def proceed(self):
        # config_file = None
        if self.frames[0].images_paths is None and self.frames[1].config_path is not None:
            print(self.frames[0].images_paths)
            config_file = self.frames[1].config_path

        elif self.frames[0].images_paths is not None and self.frames[1].config_path is None:
            print(self.frames[0].images_paths)
            config_file = new_project(self.frames[0].project_name_variable.get(),
                                      self.frames[0].experimenter_name_variable.get(),
                                      self.frames[0].images_paths,
                                      image_type=self.frames[0].image_type_box.get())

        # destroy widget then pass config file to new function
        self.container.destroy()

        main_window.main(config_file)

    # destroy child widgets and replace frame (probably more complicated
    # for widget in self.container.winfo_children():
    #     widget.destroy()
    #
    # pull up new window with selected config path

    def change_frame(self):
        frame = self.frames[self.selected_value.get()]
        frame.grid(row=2, column=0, sticky="nsew")
        frame.reset()
        frame.tkraise()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Lumeleon')
        # self.geometry('500x500')
        self.resizable(True, True)

        # notebook = ttk.Notebook(self)
        # tab1 = ControlFrame(notebook)
        # notebook.add(tab1, text="Start")
        # notebook.pack()


if __name__ == "__main__":
    # app = App()
    ControlFrame(App()).mainloop()
    # app.mainloop()

# this works
# app = App()
# ControlFrame(app)
# app.mainloop()

# class ProjectStart(tk.Frame):
#     def __init__(self, master):
#         self.master = master
#         top_panel = tk.PanedWindow(self.master)
#         radial_new = tk.Radiobutton(self.master, text="New Project")
#         radial_load = tk.Radiobutton(self.master, text="Load Project")
#
#
#
# class FileBrowseAction(tk.Frame):
#     def __init__(self, master, step_num, step_name):
#         tk.Frame.__init__(self)
#         self.step_num = step_num
#         label = tk.Label(master, text=str(step_num)+". "+step_name)
#         label.grid(row=1, column=1, sticky=tk.W)
#         path_label = tk.Label(master, text="Folder path:")
#         path_label.grid(row=2, column=1, sticky=tk.W)
#         self.entry = tk.Entry(master, width=30)
#         self.entry.grid(row=2, column=2)
#         browse_button = tk.Button(master, text="Browse files", command=self.browse_files)
#         browse_button.grid(row=2, column=3)
#         self.action_button = tk.Button(master, text="Go", command=self.action_button_conditions)
#         self.action_button.grid(row=3, column=3, sticky=tk.E)
#         if master == maskframe or master == extractframe:
#             self.uv_value = tk.IntVar()
#             self.uv_button = tk.Checkbutton(master, text="Ultraviolet", variable=self.uv_value)
#             self.uv_button.grid(row=3, column=2)
#         if master == extractframe:
#             tk.Label(master, text="Body part:").grid(row=3, column=1, sticky=tk.W)
#             self.body_entry = tk.Entry(master, width=10)
#             self.body_entry.grid(row=3, column=2)
#             self.uv_button.grid(row=3, column=3)
#             self.action_button.grid(row=3, column=4)
#
#     def browse_files(self):
#         folder_name = filedialog.askdirectory(initialdir='/', mustexist=True, title="Select folder holding images to"
#                                                                                     "be standardized")
#         self.entry.insert(0, string=folder_name)
#
#     def standardize_images(self):
#         folder_path = self.entry.get()
#         match.main(folder=folder_path)
#         self.entry.delete(0, tk.END)
#
#     def clusters_dialog(self):
#         folder_path = self.entry.get()
#         N_cluster = simpledialog.askinteger("Input", "Number of clusters to use:",
#                                             parent=window,
#                                             minvalue=0, maxvalue=10)
#         if self.uv_value.get() == 1:
#             segment.Segmentation(folder=folder_path, n_cluster=N_cluster, toplevel=window).manual_pattern_segmentation()
#         else:
#             segment.Segmentation(folder=folder_path, n_cluster=N_cluster, toplevel=window).background_segmentation()
#         self.entry.delete(0, tk.END)
#
#     def extract_values(self):
#         folder_path = self.entry.get()
#         body_part = self.body_entry.get()
#         print(body_part)
#         if self.uv_value.get() == 1 and body_part == "":
#             uv_extract.main(folder=folder_path)
#         elif self.uv_value.get() == 1 and body_part != "":
#             uv_extract.main(folder=folder_path, body_part=body_part)
#         elif self.uv_value.get() == 0 and body_part == "":
#             extract.main(folder=folder_path)
#         elif self.uv_value.get() == 0 and body_part != "":
#             extract.main(folder=folder_path, body_part=body_part)
#         self.entry.delete(0, tk.END)
#
#     def action_button_conditions(self):
#         if self.step_num == 1:
#             self.standardize_images()
#         elif self.step_num == 2:
#             self.clusters_dialog()
#         elif self.step_num == 3:
#             self.extract_values()
#
#
# FileBrowseAction(master=standardframe, step_num=1, step_name="Standardize images")
# # FileBrowseAction(master=cropframe, step_num=2, step_name="Crop images")
# FileBrowseAction(master=maskframe, step_num=2, step_name="Mask images")
# FileBrowseAction(master=extractframe, step_num=3, step_name="Extract luminance")
#
#
# def on_closing():
#     window.destroy()
#
#
# window.protocol("WM_DELETE_WINDOW", on_closing)
#
# window.mainloop()
