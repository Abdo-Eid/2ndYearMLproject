from pages import *
import customtkinter as ctk

# ------------------------------- ROOT WINDOW ----------------------------------

class TkinterApp(ctk.CTk):

    def __init__(self, size : tuple[int, int]):
        super().__init__()
        self.style = AppStyle()

        self.title("LM App")
        self.w,self.h = size
        self.geometry(f'{self.w}x{self.h}')
        self.minsize(self.w,self.h)
        # make instanse of the model to be shared
        self.data = DataModel()

        self.store_frames((Main, PreProcessing, Classification, Regression, Clustring))
        self.show_page("Main")
        self.LoadData_page = LoadData(self)
        self.columnconfigure(1,  weight=4)
        self.LoadData_page.grid(row=0, column=1, sticky="nsew")
    
    def store_frames(self, frames:tuple):
        """
        takes tuple of frame objects to store them 
        so we can access them using name
        """

        # define grid in the root 
        self.columnconfigure(0,  weight=1)
        self.rowconfigure(0, weight=1)

        # Dictionary to hold all pages
        self.pages = {}

        # Create and add pages to the dictionary
        for Page in frames:
            page_name = Page.__name__
            # initialize the frame obj into the root window
            page = Page(self)
            # store obj by name
            self.pages[page_name] = page
            # stack all pages in the same cell on top of each other
            page.grid(row=0, column=0, sticky="nsew")


    def show_page(self, page_name):
        """Show a page by name."""
        page = self.pages[page_name]
        # raise to frame to the top to be visible
        page.tkraise()

if __name__ == "__main__":
    root = TkinterApp((1180,520))
    root.mainloop()