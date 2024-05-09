import tkinter as tk   
from .utilits import simpel_imputer, min_max, apply_smote, encoding_option

class PreProcessing(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).place(x = 5, y = 5)


        frame = tk.Frame(self)
        tk.Button(frame, text="Simple Imputer", command=simpel_imputer).pack(side=tk.TOP, pady=5)
        tk.Button(frame, text="Minimax Scaler", command=min_max).pack(side=tk.TOP, pady=5)
        tk.Button(frame, text="Encoding", command=encoding_option).pack(side=tk.TOP, pady=5)
        tk.Button(frame, text="Smote", command=apply_smote).pack(side=tk.TOP, pady=5)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

          