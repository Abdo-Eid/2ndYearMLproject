import tkinter as tk   

        
class LoadData(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).place(x = 5, y = 5)

        
     