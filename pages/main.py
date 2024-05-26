import customtkinter as ctk


class Main(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.style = parent.style

        frame = ctk.CTkFrame(self)
        ctk.CTkButton(frame, text="Pre Processing", **self.style.label_MB, command=lambda: parent.show_page("PreProcessing")).pack(side=ctk.TOP, pady=5)
        ctk.CTkButton(frame, text="Classification", **self.style.label_MB, command=lambda: parent.show_page("Classification")).pack(side=ctk.TOP, pady=5)
        ctk.CTkButton(frame, text="Regression", **self.style.label_MB, command=lambda: parent.show_page("Regression")).pack(side=ctk.TOP, pady=5)
        ctk.CTkButton(frame, text="Clustring", **self.style.label_MB, command=lambda: parent.show_page("Clustring")).pack(side=ctk.TOP, pady=5)

        # Center the frame within the window
        frame.place(relx=0.5, rely=0.5, anchor=ctk.CENTER,)
