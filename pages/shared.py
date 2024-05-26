import pandas as pd
import customtkinter as ctk

# ------------------------------- Styles ----------------------------------

class AppStyle:
    def __init__(self):
        self.label_MB = {
            "width": 300,
            "height": 100,
            "font": ("Arial", 30)
        }
        self.font_g = ctk.CTkFont(family="Arial", size=30, weight="bold")

# ------------------------------- Data ----------------------------------
class DataModel:
    def __init__(self):
        self.file_path = None
        self.df = pd.DataFrame()
        self.selected_col = []
    
    # @property
    # def selected_col(self):
    #     return self._selected_col

    # @selected_col.setter
    # def selected_col(self, value):
    #     self._selected_col = value
    #     print(value)