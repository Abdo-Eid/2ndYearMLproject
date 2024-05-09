import tkinter as tk   
from tkinter import filedialog
import pandas as pd
        
class LoadData(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).place(x = 5, y = 5)
        self.data = parent.data
        self.create_widgets()
        
    def create_widgets(self):
        # First Row
        self.load_data_button = tk.Button(self, text="Load Data", command=self.load_data)
        self.load_data_button.grid(row=0, column=0, padx=5, pady=5)

        self.path_entry = tk.Entry(self)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Second Row
        self.show_dataframe_button = tk.Button(self, text="Show DataFrame", command=self.show_dataframe)
        self.show_dataframe_button.grid(row=1, column=0, padx=5, pady=5)

        self.num_duplicates_button = tk.Button(self, text="Number of Duplicates", command=self.num_duplicates)
        self.num_duplicates_button.grid(row=1, column=1, padx=5, pady=5)

        self.num_nan_button = tk.Button(self, text="Number of NaN", command=self.num_nan)
        self.num_nan_button.grid(row=1, column=2, padx=5, pady=5)

        # Third Row
        self.result_label = tk.Label(self, text="")
        self.result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

    def load_data(self):
        self.data.file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if self.data.file_path:
            try:
                self.data.data_frame = pd.read_csv(self.data.file_path)
                self.path_entry.delete(0, tk.END)
                self.path_entry.insert(0, self.data.file_path)
            except Exception as e:
                self.result_label.config(text="Error loading DataFrame: " + str(e))



    def show_dataframe(self):
        if self.data.file_path:
            try:
                self.data.data_frame = pd.read_csv(self.data.file_path)  # Assuming data is in CSV format
                self.result_label.config(text=self.data.data_frame.values.tolist())
            except Exception as e:
                self.result_label.config(text="Error loading DataFrame: " + str(e))
        else:
            self.result_label.config(text="Please load a data file first.")

    def num_duplicates(self):
        if self.data.file_path:
            try:
                if self.data.data_frame is None:
                    self.data.data_frame = pd.read_csv(self.data.file_path)  # Assuming data is in CSV format
                num_duplicates = self.data.data_frame.duplicated().sum()
                self.result_label.config(text=f"Number of Duplicates: {num_duplicates}")
            except Exception as e:
                self.result_label.config(text="Error loading DataFrame: " + str(e))
        else:
            self.result_label.config(text="Please load a data file first.")

    def num_nan(self):
        if self.data.file_path:
            try:
                if self.data.data_frame is None:
                    self.data.data_frame = pd.read_csv(self.data.file_path)  # Assuming data is in CSV format
                num_nan = self.data.data_frame.isna().sum().sum()
                self.result_label.config(text=f"Number of NaN: {num_nan}")
            except Exception as e:
                self.result_label.config(text="Error loading DataFrame: " + str(e))
        else:
            self.result_label.config(text="Please load a data file first.")

