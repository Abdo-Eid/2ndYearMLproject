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

# ------------------------------- sheet ----------------------------------


def display_df(root, df):
    """Make Dataframe as sheet using ttk TreeView
    
    Keyword arguments:
    df: pd.DataFrame
    df: 
    argument -- description
    Return: return_description
    """

    frame = tk.Frame(root)

    tree = ttk.Treeview(frame, show='headings')

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df, columns=['Value']).T


    # Insert columns
    tree["columns"] = list(df.columns)
    for col in df.columns:
        tree.column(col,width=5)
        tree.heading(col, text=col)

    # Insert data
    for index, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))

    tree.pack(expand=True,fill='both', side='left')

    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right',fill="y")

    return frame

