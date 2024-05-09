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


# ------------------------------- ROOT WINDOW ----------------------------------


class TkinterApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.style = AppStyle()

        self.title("LM App")
        self.w,self.h = 720,480
        self.geometry(f'{self.w}x{self.h}')


        self.main_win_buttons()
    
    def main_win_buttons(self):

        frame = tk.Frame(self)
        button1 = tk.Button(frame, text="Load Data", **self.style.label_MB).pack(side=tk.TOP, pady=5)
        button2 = tk.Button(frame, text="Pre Processing", **self.style.label_MB).pack(side=tk.TOP, pady=5)
        button3 = tk.Button(frame, text="Classification", **self.style.label_MB).pack(side=tk.TOP, pady=5)
        button4 = tk.Button(frame, text="Regression", **self.style.label_MB).pack(side=tk.TOP, pady=5)

        # Center the frame within the window
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

if __name__ == "__main__":
    root = TkinterApp()
    root.mainloop()