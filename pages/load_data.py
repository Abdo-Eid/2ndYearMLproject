import tkinter as tk   
from tkinter import filedialog
import pandas as pd
import tksheet
from .shared import DataModel, display_df
        
class LoadData(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data : DataModel = parent.data
        self.create_widgets()
        

    def create_widgets(self):
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(padx=5,pady=5)

        # First Row
        tk.Button(self.main_frame, text="Load Data", command=self.load_data).grid(row=1, column=0, padx=5, pady=5)

        self.path_entry = tk.Entry(self.main_frame)
        self.path_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        # Second Row
        tk.Button(self.main_frame, text="Show DataFrame", command=self.show_dataframe).grid(row=2, column=0, padx=5, pady=5)

        tk.Button(self.main_frame, text="Number of Duplicates", command=self.num_duplicates).grid(row=2, column=1, padx=5, pady=5)

        tk.Button(self.main_frame, text="Number of NaN", command=self.num_nan).grid(row=2, column=2, padx=5, pady=5)

        # Third Row
        self.show_frame = tk.Frame(self)
        self.show_frame.pack(expand=True, fill='both')
        self.result_label = tk.Label(self.show_frame, text="")
        self.result_label.pack()

    def load_data(self):
        """
        get csv file from user and insert the path into the entry
        """
        self.data.file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if self.data.file_path:
            try:
                self.data.df = pd.read_csv(self.data.file_path)
                self.path_entry.delete(0, tk.END)
                self.path_entry.insert(0, self.data.file_path)
            except Exception as e:
                self.result_label.config(text="Error loading DataFrame: " + str(e))



    def show_dataframe(self):
        if self.data.file_path:
            try:
                self.make_sheet(self.data.df)

            except Exception as e:
                self.result_label.config(text="Error: " + str(e))
        else:
            self.result_label.config(text="Please load a data file first.")

    def num_duplicates(self):
        if self.data.file_path:
            try:
                num_duplicates = self.data.df.duplicated().sum()
                self.result_label.config(text=f"Number of Duplicates rows: \n{num_duplicates}")
            except Exception as e:
                self.result_label.config(text="Error loading DataFrame: " + str(e))
        else:
            self.result_label.config(text="Please load a data file first.")

    def num_nan(self):
        if self.data.file_path:
            try:
                self.make_sheet(self.data.df.isna().sum())
            except Exception as e:
                self.result_label.config(text="Error loading DataFrame: " + str(e))
        else:
            self.result_label.config(text="Please load a data file first.")

    def make_sheet(self,df:pd.DataFrame):

        if hasattr(self,'sheet'):
            self.sheet.destroy()

        self.sheet = display_df(self,df)
        self.sheet.pack(expand=True,fill='both')
        