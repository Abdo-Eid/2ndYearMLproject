import customtkinter as ctk

from .utilits import SVM_C,KNN, DTC, logistic_regression, ANN

class Classification(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.data = parent.data

        ctk.CTkButton(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.create_widgets()

    def create_widgets(self):
        self.header = ctk.CTkTabview(self)
        self.header.pack(expand=True, fill='both')
        # header
        self.create_frame_SVC()
        self.create_frame_knn()
        self.create_frame_logistic()
        self.create_frame_DTC()
        self.create_frame_ANN()


    def create_frame_SVC(self):
        # adding to header
        self.header.add('SVM')

        frame = ctk.CTkFrame(self.header.tab('SVM'))
        frame.pack(fill='both', expand=True)

        # strategy option menu
        ctk.CTkLabel(frame, text="Enter kernal:").pack()
        options = ["linear", "rbf"]
        # make variable inside the frame called selected_option
        selected_option = ctk.StringVar(frame)
        # set it to the second item
        selected_option.set(options[1])
        combobox = ctk.CTkComboBox(frame, variable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        # ratio entry
        ctk.CTkLabel(frame, text="Enter split test size:").pack()
        entry = ctk.CTkEntry(frame)
        entry.pack()

        ctk.CTkButton(frame, text="implement",
                  command=lambda: SVM_C(self.data, selected_option.get(), int(entry.get()))).pack()


    def create_frame_knn(self):
        # adding to header
        self.header.add('K.N.N')
        frame = ctk.CTkFrame(self.header.tab('K.N.N'))
        frame.pack(fill='both', expand=True)

        ctk.CTkLabel(frame, text="Enter Number of Neighbors:").pack()
        n = ctk.CTkEntry(frame)
        n.pack()

        # matrix option menu
        ctk.CTkLabel(frame, text="chose matrix:").pack()
        options = ["hamming","chebyshev"]
        selected_option = ctk.StringVar(frame)
        selected_option.set(options[1])
        combobox = ctk.CTkComboBox(frame, variable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        # ratio entry
        ctk.CTkLabel(frame, text="Enter split test size:").pack()
        entry = ctk.CTkEntry(frame)
        entry.pack()

        ctk.CTkButton(frame, text="implement",
                  command=lambda: KNN(self.data, int(n.get()), selected_option.get(), int(entry.get()))).pack()
        
    def create_frame_logistic(self):
        # adding to header
        self.header.add('Logistic')
        frame = ctk.CTkFrame(self.header.tab('Logistic'))
        frame.pack(fill='both', expand=True)

        # ratio entry
        ctk.CTkLabel(frame, text="Enter split test size:").pack()
        entry = ctk.CTkEntry(frame)
        entry.pack()

        ctk.CTkButton(frame, text="implement",command=lambda: logistic_regression(self.data, int(entry.get()))).pack()




    def create_frame_ANN(self):

        self.header.add('ANN')
        frame = ctk.CTkFrame(self.header.tab('ANN'))
        frame.pack(fill='both', expand=True)

        ctk.CTkLabel(frame, text="Enter Number of layers:").pack()
        entry_layers = ctk.CTkEntry(frame)
        entry_layers.pack()

        ctk.CTkLabel(frame, text="chose learning_rate:").pack()
        options = ["constant", "invscaling", "adaptive"]
        selected_option = ctk.StringVar(frame)
        selected_option.set(options[0])
        combobox = ctk.CTkComboBox(frame, variable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        ctk.CTkLabel(frame, text="Enter split test size:").pack()
        entry_test_size = ctk.CTkEntry(frame)
        entry_test_size.pack()

        ctk.CTkButton(frame, text="implement",
                  command=lambda: ANN(self.data, selected_option.get(), int(entry_test_size.get()),int(entry_layers.get()))).pack()

    def create_frame_DTC(self):
            self.header.add('D.T.C')
            frame = ctk.CTkFrame(self.header.tab('D.T.C'))
            frame.pack(fill='both', expand=True)
            # strategy option menu
            ctk.CTkLabel(frame, text="Enter Max Depth:").pack()
            depth = ctk.CTkEntry(frame)
            depth.pack()
            ctk.CTkLabel(frame, text="chose Criterion:").pack()
            options = ["entropy","gini"]
            # make variable inside the frame called selected_option
            selected_option = ctk.StringVar(frame)
            # set it to the second item
            selected_option.set(options[1])
            combobox = ctk.CTkComboBox(frame, variable=selected_option, values=options)
            combobox.pack(padx=20, pady=20)

            # ratio entry
            ctk.CTkLabel(frame, text="Enter split test size:").pack()
            entry = ctk.CTkEntry(frame)
            entry.pack()

            ctk.CTkButton(frame, text="implement",
                    command=lambda: DTC(self.data, int(depth.get()), selected_option.get(), int(entry.get()))).pack()

