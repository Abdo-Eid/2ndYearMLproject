import tkinter as tk   
# ------------------------------- Styles ----------------------------------

class AppStyle:
    def __init__(self):

        self.label_MB = {
            "fg": "#FFFFFF",
            "bg": "#2C3E50",
            "cursor": "arrow",
            "font": ("Arial", 14),
            "width": 18,
            "height": 3
        }

# ------------------------------- pages ----------------------------------

class Main(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.style = parent.style

        frame = tk.Frame(self)
        tk.Button(frame, text="Load Data", **self.style.label_MB, command=lambda: parent.show_page("LoadData")).pack(side=tk.TOP, pady=5)
        tk.Button(frame, text="Pre Processing", **self.style.label_MB, command=lambda: parent.show_page("PreProcessing")).pack(side=tk.TOP, pady=5)
        tk.Button(frame, text="Classification", **self.style.label_MB, command=lambda: parent.show_page("Classification")).pack(side=tk.TOP, pady=5)
        tk.Button(frame, text="Regression", **self.style.label_MB, command=lambda: parent.show_page("Regression")).pack(side=tk.TOP, pady=5)

        # Center the frame within the window
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# -----------------------------------------------------------------
        
class LoadData(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).place(x = 5, y = 5)

                
# -----------------------------------------------------------------
        
class PreProcessing(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).place(x = 5, y = 5)

                
# -----------------------------------------------------------------
        
class Classification(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).place(x = 5, y = 5)
                
                
# -----------------------------------------------------------------
        

class Regression(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        tk.Button(self, text="Main", command=lambda: parent.show_page("Main")).place(x = 5, y = 5)

