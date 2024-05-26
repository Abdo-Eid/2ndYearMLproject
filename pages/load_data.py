import customtkinter as ctk
from tkinter import ttk, filedialog
import pandas as pd
from .shared import DataModel
        
class LoadData(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.data : DataModel = parent.data
        self.create_widgets()
    def create_widgets(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(padx=15,pady=5)

        # First Row
        ctk.CTkButton(self.main_frame, text="Load Data", command=self.load_data).grid(row=0, column=0, padx=5, pady=5)

        self.path_entry = ctk.CTkEntry(self.main_frame)
        self.path_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(self.main_frame, text="save", command=self.save_dataframe).grid(row=0, column=4, padx=5, pady=5)

        # Second Row
        ctk.CTkButton(self.main_frame, text="Show DataFrame", command=self.show_dataframe).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkButton(self.main_frame, text="hide table", command=self.clear_dataframe).grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(self.main_frame, text="clear select", command=self.clear_select).grid(row=1, column=2, padx=5, pady=5)
        ctk.CTkButton(self.main_frame, text="# of Duplicates", command=self.num_duplicates).grid(row=1, column=3, padx=5, pady=5)
        ctk.CTkButton(self.main_frame, text="# of NaN", command=self.num_nan).grid(row=1, column=4, padx=5, pady=5)

        # Third Row
        self.result_label = ctk.CTkLabel(self, text="")
        self.result_label.pack()

    def load_data(self):
        """
        get csv file from user and insert the path into the entry
        """
        self.data.file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if self.data.file_path:
            try:
                self.data.df = pd.read_csv(self.data.file_path)
                self.path_entry.delete(0, ctk.END)
                self.path_entry.insert(0, self.data.file_path)
            except Exception as e:
                self.result_label.configure(text="Error loading DataFrame: " + str(e))

    def show_dataframe(self):
        if self.data.file_path or not self.data.df.empty:
            try:
                self.show_sheet(self.data.df)

            except Exception as e:
                self.result_label.configure(text="Error: " + str(e))
        else:
            self.result_label.configure(text="Please load a data file first.")

    def clear_dataframe(self):
        if hasattr(self,'sheet'):
            self.sheet.destroy()

    def num_duplicates(self):
        if self.data.file_path:
            try:
                num_duplicates = self.data.df.duplicated().sum()
                self.result_label.configure(text=f"Number of Duplicates rows: \n{num_duplicates}")
            except Exception as e:
                self.result_label.configure(text="Error loading DataFrame: " + str(e))
        else:
            self.result_label.configure(text="Please load a data file first.")

    def num_nan(self):
        if self.data.file_path:
            try:
                self.show_sheet(self.data.df.isna().sum())
            except Exception as e:
                self.result_label.configure(text="Error loading DataFrame: " + str(e))
        else:
            self.result_label.configure(text="Please load a data file first.")

    def save_dataframe(self):
        """
        Save a DataFrame to a file with "new" added to the end of the filename.
        """
        # Split the path into directory and filename
        directory, filename = self.data.file_path.rsplit('/', 1) if '/' in self.data.file_path else ('', self.data.file_path)

        # Add "new" to the filename
        new_filename = filename.replace('.csv', '_new.csv')

        # Concatenate directory and new filename
        new_path = f"{directory}/{new_filename}" if directory else new_filename

        # Save the DataFrame to the new path
        self.data.df.to_csv(new_path, index=False)

    def show_sheet(self,df:pd.DataFrame | pd.Series):
        """Make Dataframe as sheet using ttk TreeView.
        
        if the sheet exist it will be removed then packed again (updating)
        """
        
        # check if the root (self) has attribute 'sheet'
        # if so delete it
        if hasattr(self,'sheet'):
            self.sheet.destroy()

        # making the tree view in a frame
        self.sheet = ctk.CTkFrame(self)

        tree = ttk.Treeview(self.sheet, show='headings')

        # check if the dataframe is a series if so transpose it
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df, columns=['Value']).T


        # Insert columns
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.column(col,width=5)
            tree.heading(col, text=col)

        # Insert data
        for index, row in df.iterrows():
            tree.insert("", ctk.END, values=list(row))

        tree.pack(expand=True,fill='both', side='left')

        # When selecting a column it stored 
        def on_column_select(event):
            selected_column = tree.identify_column(event.x)
            column_id = int(selected_column.replace("#", "")) - 1
            column_name = df.columns[column_id]

            if column_name in self.data.selected_col:
                self.data.selected_col.remove(column_name)
            else:
                self.data.selected_col.append(column_name)

            self.path_entry.delete(0, ctk.END)
            self.path_entry.insert(0, str(self.data.selected_col))


        # Bind the selection event
        tree.bind("<ButtonRelease-1>", on_column_select)


        # make a scrollbar
        scrollbar = ctk.CTkScrollbar(self.sheet, orientation=ctk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right',fill="y")

        self.sheet.pack(expand=True,fill='both')

    def clear_select(self):
        self.data.selected_col.clear()
        self.path_entry.delete(0, ctk.END)