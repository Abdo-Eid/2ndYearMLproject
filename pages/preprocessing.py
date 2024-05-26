import customtkinter as ctk
from .utilits import simple_imputer, min_max, label_encode, one_hot_encode, delete_selected
from .shared import DataModel
from .utilits import smote

class PreProcessing(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.style = parent.style
        

        ctk.CTkButton(self, text="Main", command=lambda: parent.show_page("Main")).pack(padx=5,pady=5)
        self.data : DataModel = parent.data
        self.create_widgets()
        

    def create_widgets(self):

        # making tab view using notebook
        self.header = ctk.CTkTabview(self)
        self.header.pack(expand=True, fill='both')
        # header
        self.create_imputer_frame()
        self.create_Encoder_frame()
        self.create_scaler_frame()
        self.create_smote_frame()
        self.create_more_frame()

    def create_imputer_frame(self):

        # adding to header
        self.header.add('simple imputer')
        frame = ctk.CTkFrame(self.header.tab('simple imputer'))
        frame.pack(fill='both', expand=True)

        # option menu
        ctk.CTkLabel(frame, text="Enter Strategy:").pack()
        options = ["mean", "median", "most_frequent"]
        # make variable inside the frame called selected_option
        selected_option = ctk.StringVar(frame)
        # set it to the second item
        selected_option.set(options[1])
        combobox = ctk.CTkComboBox(frame, variable=selected_option, values=options)
        combobox.pack(padx=20, pady=20)

        ctk.CTkButton(frame, text="Deal With NaN", command=lambda : simple_imputer(self.data,selected_option.get())).pack(pady=5)
        


        
    def create_Encoder_frame(self):

        # adding to header
        self.header.add('Encoding')
        frame = ctk.CTkFrame(self.header.tab('Encoding'))
        frame.pack(fill='both', expand=True)

        ctk.CTkButton(frame, text="Label Encoder", command= lambda : label_encode(self.data)).pack(pady=5)
        ctk.CTkButton(frame, text="One Hot Encoder", command=lambda : one_hot_encode(self.data)).pack(pady=5)
        

    def create_scaler_frame(self):
        
        # adding to header
        self.header.add('MinMax scaler')
        ctk.CTkButton(self.header.tab('MinMax scaler'), text="MinMax scaler", command=lambda : min_max(self.data)).pack()
        
        
    def create_smote_frame(self):

        self.header.add('SMOTE')
        frame = ctk.CTkFrame(self.header.tab('SMOTE'))
        frame.pack(fill='both', expand=True)


        ctk.CTkButton(frame, text="Implement SMOTE", command=lambda: smote(self.data, frame)).pack(pady=5)


    def create_more_frame(self):
        self.header.add('more')
        frame = ctk.CTkFrame(self.header.tab('more'))
        frame.pack(fill='both', expand=True)

        ctk.CTkButton(frame, text= "delete selected", command= lambda : delete_selected(self.data)).pack(pady=5)
        ctk.CTkButton(frame, text= "delete dublcate", command= lambda : self.data.df.drop_duplicates(inplace=True)).pack(pady=5)
        



