import pandas as pd
import tkinter as tk
from tkinter import ttk
# ------------------------------- Styles ----------------------------------

class AppStyle:
    def __init__(self):

        self.label_MB = {
            "fg": "#FFFFFF",
            "bg": "#2C3E50",
            "cursor": "arrow",
            "font": ("Arial", 14),
            "width": 18,
            "height": 3
        }
# ------------------------------- Data ----------------------------------
class DataModel:
    def __init__(self):
        self.data = None
        self.file_path = None
        self.df = pd.DataFrame()
