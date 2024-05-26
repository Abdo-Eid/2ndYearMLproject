import tkinter as tk
from tkinter import ttk
from .utilits import simple_imputer, min_max, label_encode, one_hot_encode, delete_selected
from .shared import DataModel
from .utilits import smote

class PreProcessing(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data : DataModel = parent.data
        self.create_widgets()
        

    def create_widgets(self):

        # making tab view using notebook
        self.header = ttk.Notebook(self)
        self.header.pack(expand=True, fill='both')
        # header
        self.create_imputer_frame()
        self.create_Encoder_frame()
        self.create_scaler_frame()
        self.create_more_frame()

    def create_imputer_frame(self):

        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        # option menu
        tk.Label(frame, text="Enter Strategy:").pack()
        options = ["mean", "median", "most_frequent"]
        # make variable inside the frame called selected_option
        selected_option = tk.StringVar(frame)
        # set it to the second item
        selected_option.set(options[1])
        combobox = ttk.Combobox(frame, textvariable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        tk.Button(frame, text="Deal With NaN", command=lambda : simple_imputer(self.data,selected_option.get())).pack()
        
        # adding to header
        self.header.add(frame,text = 'simple imputer')

        
    def create_Encoder_frame(self):

        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        tk.Button(frame, text="Label Encoder", command= lambda : label_encode(self.data)).pack()
        tk.Button(frame, text="One Hot Encoder", command=lambda : one_hot_encode(self.data)).pack()
        
        # adding to header
        self.header.add(frame,text = 'Encoding')

    def create_scaler_frame(self):
        
        button2 = tk.Button(self, text="MinMax scaler", command=lambda : min_max(self.data))
        button2.pack()
        
        # adding to header
        self.header.add(button2,text = 'MinMax scaler')
        
    def create_frame4(self):

        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text="Enter split test size:").pack()
        entry_test_size = ttk.Entry(frame, width=30)
        entry_test_size.pack()

        button3 = tk.Button(frame, text="Implement SMOTE", command=lambda: smote(self.data, int(entry_test_size.get()),frame))
        button3.pack()

        self.header.add(frame, text='SMOTE')

    def create_more_frame(self):
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        tk.Button(frame, text= "delete selected", command= lambda : delete_selected(self.data)).pack()
        tk.Button(frame, text= "delete dublcate", command= lambda : self.data.df.drop_duplicates(inplace=True)).pack()
        
        self.header.add(frame,text = 'more')



