import tkinter as tk   
from tkinter import ttk


class Regression(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        ttk.Button(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data = parent.data

        self.create_widgets()

    def create_widgets(self):

        # making tab view using notebook
        self.header = ttk.Notebook(self)
        self.header.pack(expand=True, fill='both')
        # header
        self.create_Linear_frame()
        self.create_Logistic_frame()

    def create_Linear_frame(self):

        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        # welcome_label = Label(frame2, text="Regression Page", fg='#2C3E50', bg="#F2F3F4",
        #                     font=("Arial", 14))  # Text color changed to dark blue
        # welcome_label.pack()
        # q = Button(frame2, text="Linear Regression", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
        #         command=call_linear_regression)  # Text color changed to dark blue, background color changed to white
        # q.pack()
        # q = Button(frame2, text="Logistic Regression", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
        #         command=call_logistic_regression)  # Text color changed to dark blue, background color changed to white
        # q.pack()

        # adding to header
        self.header.add(frame,text = 'simple imputer')

        
    def create_Logistic_frame(self):
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)



        # adding to header
        self.header.add(frame,text = 'simple imputer')




