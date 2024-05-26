import tkinter as tk
from tkinter import ttk
from .shared import DataModel

class Clustring(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data : DataModel = parent.data
        self.create_widgets()
    def create_widgets(self):
        pass
