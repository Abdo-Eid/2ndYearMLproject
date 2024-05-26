import tkinter as tk
from tkinter import ttk
from .utilits import SVM_C,KNN,DTC, ANN


class Classification(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.data = parent.data

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.create_widgets()

    def create_widgets(self):
        self.header = ttk.Notebook(self)
        self.header.pack(expand=True, fill='both')
        # header
        self.create_frame_SVC()
        self.create_frame_knn()
        self.create_frame_ANN()
        self.create_frame_DTC()


    def create_frame_SVC(self):
        # frame 1
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        # strategy option menu
        ttk.Label(frame, text="Enter kernal:").pack()
        options = ["linear", "rbf"]
        # make variable inside the frame called selected_option
        selected_option = tk.StringVar(frame)
        # set it to the second item
        selected_option.set(options[1])
        combobox = ttk.Combobox(frame, textvariable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        # ratio entry
        ttk.Label(frame, text="Enter split test size:").pack()
        entry = ttk.Entry(frame, width=30)
        entry.pack()

        tk.Button(frame, text="implement",
                  command=lambda: SVM_C(self.data, selected_option.get(), int(entry.get()))).pack()

        # adding to header
        self.header.add(frame, text='SVM')

    def create_frame_knn(self):
        # frame 1
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)
        # strategy option menu
        ttk.Label(frame, text="Enter Number of Neighbors:").pack()
        n = ttk.Entry(frame, width=30)
        n.pack()
        ttk.Label(frame, text="chose matrix:").pack()
        options = ["hamming","chebyshev"]
        # make variable inside the frame called selected_option
        selected_option = tk.StringVar(frame)
        # set it to the second item
        selected_option.set(options[1])
        combobox = ttk.Combobox(frame, textvariable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        # ratio entry
        ttk.Label(frame, text="Enter split test size:").pack()
        entry = ttk.Entry(frame, width=30)
        entry.pack()

        tk.Button(frame, text="implement",
                  command=lambda: KNN(self.data, int(n.get()), selected_option.get(), int(entry.get()))).pack()

        # adding to header
        self.header.add(frame, text='K.N.N')
        
    def create_frame_logistic(self):
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        # ratio entry
        ttk.Label(frame, text="Enter split test size:").pack()
        entry = ttk.Entry(frame, width=30)
        entry.pack()

        tk.Button(frame, text="implement",
                  command=lambda: logistic_regression(self.data, int(entry.get()))).pack()

        # adding to header
        self.header.add(frame,text = 'Logistic')

    def create_frame_ANN(self):

        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text="Enter Number of layers:").pack()
        entry_layers = ttk.Entry(frame, width=30)
        entry_layers.pack()

        ttk.Label(frame, text="chose learning_rate:").pack()
        options = ["constant", "invscaling", "adaptive"]
        selected_option = tk.StringVar(frame)
        selected_option.set(options[0])
        combobox = ttk.Combobox(frame, textvariable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        ttk.Label(frame, text="Enter split test size:").pack()
        entry_test_size = ttk.Entry(frame, width=30)
        entry_test_size.pack()

        tk.Button(frame, text="implement",
                  command=lambda: ANN(self.data, selected_option.get(), int(entry_test_size.get()),int(entry_layers.get()))).pack()

        self.header.add(frame, text='ANN')

    def create_frame_DTC(self):
        # frame 1
        frame = ttk.Frame(self.header)
        frame.pack(fill='both', expand=True)
        # strategy option menu
        ttk.Label(frame, text="Enter Max Depth:").pack()
        depth = ttk.Entry(frame, width=30)
        depth.pack()
        ttk.Label(frame, text="chose Criterion:").pack()
        options = ["entropy","gini"]
        # make variable inside the frame called selected_option
        selected_option = tk.StringVar(frame)
        # set it to the second item
        selected_option.set(options[1])
        combobox = ttk.Combobox(frame, textvariable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        # ratio entry
        ttk.Label(frame, text="Enter split test size:").pack()
        entry = ttk.Entry(frame, width=30)
        entry.pack()

        tk.Button(frame, text="implement",
                  command=lambda: DTC(self.data, int(depth.get()), selected_option.get(), int(entry.get()))).pack()

        # adding to header
        self.header.add(frame, text='D.T.C')