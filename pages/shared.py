import pandas as pd

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
        self.selected_col = []
    
    # @property
    # def selected_col(self):
    #     return self._selected_col

    # @selected_col.setter
    # def selected_col(self, value):
    #     self._selected_col = value
    #     print(value)