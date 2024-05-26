import customtkinter as ctk
from .shared import DataModel
from .utilits import KM


class Clustring(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        

        ctk.CTkButton(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data : DataModel = parent.data
        self.create_widgets()

    def create_widgets(self):
        self.header = ctk.CTkTabview(self)
        self.header.pack(expand=True, fill='both')
        self.create_frame_km()

    def create_frame_km(self):
        # adding to header
        self.header.add('clustring')
        frame = ctk.CTkFrame(self.header.tab('clustring'))
        frame.pack(fill='both', expand=True)
        # strategy option menu
        ctk.CTkLabel(frame, text="Enter Number of cluster:").pack()
        entry = ctk.CTkEntry(frame)
        entry.pack()
        ctk.CTkButton(frame, text="implement",command=lambda: KM(self.data, frame, int((entry.get())))).pack()




