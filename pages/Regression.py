import customtkinter as ctk
from .utilits import linear_regression, SVM_r

class Regression(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        ctk.CTkButton(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data = parent.data

        self.create_widgets()

    def create_widgets(self):

        # making tab view using notebook
        self.header = ctk.CTkTabview(self)
        self.header.pack(expand=True, fill='both')
        # header
        self.create_Linear_frame()
        self.create_SVR_frame()

    def create_Linear_frame(self):

        # adding to header
        self.header.add('Linear')
        frame = ctk.CTkFrame(self.header.tab('Linear'))
        frame.pack(fill='both', expand=True)

        # ratio entry
        ctk.CTkLabel(frame, text="Enter split test size:").pack()
        entry = ctk.CTkEntry(frame)
        entry.pack()

        ctk.CTkButton(frame, text="implement",
                  command=lambda: linear_regression(self.data, int(entry.get()))).pack()

        
    def create_SVR_frame(self):
        # adding to header
        self.header.add('SVR')
        frame = ctk.CTkFrame(self.header.tab('SVR'))
        frame.pack(fill='both', expand=True)

        # kernal option menu
        ctk.CTkLabel(frame, text="chose kernal:").pack()
        options = ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed']
        selected_option = ctk.StringVar(frame)
        selected_option.set(options[1])
        combobox = ctk.CTkComboBox(frame, variable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        # ratio entry
        ctk.CTkLabel(frame, text="Enter split test size:").pack()
        entry = ctk.CTkEntry(frame)
        entry.pack()

        ctk.CTkButton(frame, text="implement",
                  command=lambda: SVM_r(self.data, selected_option.get(), int(entry.get()))).pack()




