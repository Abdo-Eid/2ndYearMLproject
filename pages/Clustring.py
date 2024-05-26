import tkinter as tk
from tkinter import ttk
from .shared import DataModel
from .utilits import KM


class Clustring(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data : DataModel = parent.data
        self.create_widgets()

    def create_widgets(self):
        self.header = ttk.Notebook(self)
        self.header.pack(expand=True, fill='both')
        self.create_frame_km()

    def create_frame_km(self):
        # frame 1
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)
        # strategy option menu
        ttk.Label(frame, text="Enter Number of cluster:").pack()
        entry = ttk.Entry(frame, width=30)
        entry.pack()
        tk.Button(frame, text="implement",command=lambda: KM(self.data, frame, int((entry.get())))).pack()

        # adding to header
        self.header.add(frame, text='clustring')


