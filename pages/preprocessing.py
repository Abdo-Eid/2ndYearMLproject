import tkinter as tk   
from .utilits import simpel_imputer, min_max, apply_smote, encoding_option
from .shared import DataModel

class PreProcessing(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data : DataModel = parent.data
        self.create_widgets()
        

    def create_widgets(self):
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(padx=5,pady=5)

        # First Row
        tk.Button(self.main_frame, text="Simple Imputer", command=simpel_imputer).pack(padx=5,pady=5, side='left')
        tk.Button(self.main_frame, text="Minimax Scaler", command=min_max).pack(padx=5,pady=5, side='left')
        tk.Button(self.main_frame, text="Encoding", command=encoding_option).pack(padx=5,pady=5, side='left')
        tk.Button(self.main_frame, text="Smote", command=apply_smote).pack(padx=5,pady=5, side='left')

        # Second Row
        self.show_frame = tk.Frame(self)
        self.show_frame.pack(expand=True, fill='both')
        self.result_label = tk.Label(self.show_frame, text="")
        self.result_label.pack()

          