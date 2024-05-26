import tkinter as tk   
from tkinter import ttk
from .utilits import linear_regression, SVM_r

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
        self.create_SVR_frame()

    def create_Linear_frame(self):

        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        # ratio entry
        ttk.Label(frame, text="Enter split test size:").pack()
        entry = ttk.Entry(frame, width=30)
        entry.pack()

        tk.Button(frame, text="implement",
                  command=lambda: linear_regression(self.data, int(entry.get()))).pack()

        # adding to header
        self.header.add(frame,text = 'Linear')

        
    def create_SVR_frame(self):
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        # kernal option menu
        ttk.Label(frame, text="chose kernal:").pack()
        options = ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed']
        selected_option = tk.StringVar(frame)
        selected_option.set(options[1])
        combobox = ttk.Combobox(frame, textvariable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        # ratio entry
        ttk.Label(frame, text="Enter split test size:").pack()
        entry = ttk.Entry(frame, width=30)
        entry.pack()

        tk.Button(frame, text="implement",
                  command=lambda: SVM_r(self.data, selected_option.get(), int(entry.get()))).pack()

        # adding to header
        self.header.add(frame,text = 'SVR')




