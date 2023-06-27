import tkinter as tk
import match
import segment
import uv_extract
import extract
from tkinter import filedialog
from tkinter import simpledialog
window = tk.Tk()
window.title("Lumeleon")
print(type(window))


standardframe = tk.Frame(window)
standardframe.grid(row=0, column=0)
cropframe = tk.Frame(window)
cropframe.grid(row=1, column=0)
maskframe = tk.Frame(window)
maskframe.grid(row=2, column=0)
extractframe = tk.Frame(window)
extractframe.grid(row=3, column=0)


class FileBrowseAction(tk.Frame):
    def __init__(self, master, step_num, step_name):
        tk.Frame.__init__(self)
        self.step_num = step_num
        label = tk.Label(master, text=str(step_num)+". "+step_name)
        label.grid(row=1, column=1, sticky=tk.W)
        path_label = tk.Label(master, text="Folder path:")
        path_label.grid(row=2, column=1, sticky=tk.W)
        self.entry = tk.Entry(master, width=30)
        self.entry.grid(row=2, column=2)
        browse_button = tk.Button(master, text="Browse files", command=self.browse_files)
        browse_button.grid(row=2, column=3)
        self.action_button = tk.Button(master, text="Go", command=self.action_button_conditions)
        self.action_button.grid(row=3, column=3, sticky=tk.E)
        if master == maskframe or master == extractframe:
            self.uv_value = tk.IntVar()
            self.uv_button = tk.Checkbutton(master, text="Ultraviolet", variable=self.uv_value)
            self.uv_button.grid(row=3, column=2)
        if master == extractframe:
            tk.Label(master, text="Body part:").grid(row=3, column=1, sticky=tk.W)
            self.body_entry = tk.Entry(master, width=10)
            self.body_entry.grid(row=3, column=2)
            self.uv_button.grid(row=3, column=3)
            self.action_button.grid(row=3, column=4)

    def browse_files(self):
        folder_name = filedialog.askdirectory(initialdir='/', mustexist=True, title="Select folder holding images to"
                                                                                    "be standardized")
        self.entry.insert(0, string=folder_name)

    def standardize_images(self):
        folder_path = self.entry.get()
        match.main(folder=folder_path)
        self.entry.delete(0, tk.END)

    def clusters_dialog(self):
        folder_path = self.entry.get()
        N_cluster = simpledialog.askinteger("Input", "Number of clusters to use:",
                                            parent=window,
                                            minvalue=0, maxvalue=10)
        if self.uv_value.get() == 1:
            segment.segment_gui(folder=folder_path, N_cluster=N_cluster, uv=True, toplevel=window)
        else:
            segment.segment_gui(folder=folder_path, N_cluster=N_cluster, uv=False, toplevel=window)
        self.entry.delete(0, tk.END)

    def extract_values(self):
        folder_path = self.entry.get()
        body_part = self.body_entry.get()
        print(body_part)
        if self.uv_value.get() == 1 and body_part == "":
            uv_extract.main(folder=folder_path)
        elif self.uv_value.get() == 1 and body_part != "":
            uv_extract.main(folder=folder_path, body_part=body_part)
        elif self.uv_value.get() == 0 and body_part == "":
            extract.main(folder=folder_path)
        elif self.uv_value.get() == 0 and body_part != "":
            extract.main(folder=folder_path, body_part=body_part)
        self.entry.delete(0, tk.END)

    def action_button_conditions(self):
        if self.step_num == 1:
            self.standardize_images()
        elif self.step_num == 2:
            self.clusters_dialog()
        elif self.step_num == 3:
            self.extract_values()


FileBrowseAction(master=standardframe, step_num=1, step_name="Standardize images")
# FileBrowseAction(master=cropframe, step_num=2, step_name="Crop images")
FileBrowseAction(master=maskframe, step_num=2, step_name="Mask images")
FileBrowseAction(master=extractframe, step_num=3, step_name="Extract luminance")


def on_closing():
    window.destroy()


window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
