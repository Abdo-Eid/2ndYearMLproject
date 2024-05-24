from tkinter import  font
from tkinter import *
from tkinter import filedialog
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from tksheet import Sheet
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE

global data, y_train, y_test, x_test, x_train
data = pd.DataFrame
a = Tk()
a.geometry('800x500')
a.title("Machine Learning")


def display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat):
    # Create a Tkinter window
    window = Tk()
    window.geometry('600x400')  # Set window size
    window.title("Evaluation Metrics Display")

    # Create a frame to contain labels
    frame = Frame(window)
    frame.pack(padx=20, pady=10)

    # Define font styles
    title_font = font.Font(family='Helvetica', size=16, weight='bold')
    label_font = font.Font(family='Helvetica', size=12)

    # Accuracy label
    accuracy_label = Label(frame, text="Accuracy: {:.2f}%".format(accuracy * 100), font=title_font)
    accuracy_label.pack()

    # Precision label
    precision_label = Label(frame, text="Precision: {:.2f}".format(precision), font=label_font)
    precision_label.pack()

    # Recall label
    recall_label = Label(frame, text="Recall: {:.2f}".format(recall), font=label_font)
    recall_label.pack()

    # F1-score label
    f1_label = Label(frame, text="F1-score: {:.2f}".format(f1), font=label_font)
    f1_label.pack()

    # Confusion matrix label
    confusion_label = Label(frame, text="Confusion Matrix:\n{}".format(confusion_mat), font=label_font)
    confusion_label.pack()

    # Run the Tkinter event loop
    window.mainloop()
def display():
    global data
    def enable_encoding_button(h=None):
        h.config(state="normal")
    frame2 = Toplevel()
    frame2.title("CSV Screen")
    frame2.geometry("600x400")
    frame2.configure(bg="#F2F3F4")  # Background color changed to light gray
    welcome_label = Label(frame2, text="Pre Processing Page", fg='#2C3E50', bg="#F2F3F4",
                          font=("Arial", 14))  # Text color changed to dark blue
    welcome_label.pack()
    q = Button(frame2, text="Simple Imputer", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,command=simpel_imputer)  # Text color changed to dark blue, background color changed to white
    q.pack()
    e = Button(frame2, text="Minimax Scaler", fg="#2C3E50", bg="#FFFFFF", width=20,
               height=3,
               command= min_max)  # Text color changed to dark blue, background color changed to white
    e.pack()
    h = Button(frame2, text="Encoding", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
               command=encoding_option)  # Text color changed to dark blue, background color changed to white
    h.pack()
    t = Button(frame2, text="Smote", fg="#2C3E50", bg="#FFFFFF", width=20,
               height=3, command=apply_smote)  
    t.pack()


def read_csv():
    global data
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        print(data.shape)
        return data
    else:
        return None


def min_max():
    global data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data))
    print(data)


def simp_nan(strateg):
    # Initialize SimpleImputer
    global data
    imputer = SimpleImputer(strategy=strateg)
    # Impute missing values in the 'age' column
    data = imputer.fit_transform(data)
def apply_smote():
    global data
    X = data.iloc[:, :-1]  # Take all columns except the last one
    y = data.iloc[:, -1]  # Take only the last column as the target variable

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)],
                               axis=1)
def simpel_imputer():
    global data
    frame2 = Toplevel()
    frame2.title("Simple Imputer Option")
    frame2.geometry("400x300")

    def get_input():
        strategy = selected_option.get()
        simp_nan(strategy)

    label = Label(frame2, text="Enter Strategy:")
    label.pack()
    options = ["mean", "most_frequent", "median"]
    selected_option = StringVar(frame2)
    selected_option.set(options[0])
    option_menu = OptionMenu(frame2, selected_option, *options)
    option_menu.pack(padx=20, pady=20)
    # entry = Entry(frame2, width=30)
    # entry.pack()
    # Create a button to submit the input
    button = Button(frame2, text="Deal With NaN", command=get_input)
    button.pack()
def print_data():
    global data
    # Example DataFrame# Create Tkinter window
    root = Tk()
    root.title("DataFrame Viewer")
    # Create a tksheet instance
    sheet = Sheet(root)
    # Convert DataFrame to a list of lists
    data_list = data.values.tolist()
    # Set the data to the tksheet
    sheet.set_sheet_data(data_list)
    # Pack the tksheet into the Tkinter window
    sheet.pack(padx=10, pady=10, fill="both", expand=True)
    # Run the Tkinter event loop
    root.mainloop()
def one_hot_encode():
    # Identify categorical columns
    global data
    categorical_columns = data.select_dtypes(include=['object']).columns
    # Extract categorical data
    categorical_data = data[categorical_columns]
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the encoder on the categorical data
    encoded_data = encoder.fit_transform(categorical_data)
    # Create column names for the encoded features
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    # Create a DataFrame for the encoded features
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    # Concatenate the encoded DataFrame with the original DataFrame
    data = pd.concat([data.drop(columns=categorical_columns), encoded_df], axis=1)
def KNN():
    global data
    frame2 = Toplevel()
    frame2.title("KNN Options")
    frame2.geometry("600x400")

    def get_input(entry):
        strategy = entry.get()
        return strategy

    label = Label(frame2, text="Enter Number of Neighbors:")
    label.pack()
    entry = Entry(frame2, width=30)
    entry.pack()
    button = Button(frame2, text="Submit Number of Neighbors", command=lambda: print(get_input(entry)))
    button.pack()

    label = Label(frame2, text="Enter Distance Metric (metric eg :: (hamming,mahalanobis,chebyshev,....):")
    label.pack()
    entry2 = Entry(frame2, width=30)
    entry2.pack()
    button = Button(frame2, text="Submit Distance Metric", command=lambda: print(get_input(entry2)))
    button.pack()

    label3 = Label(frame2, text="Enter Train Ratio:")
    label3.pack()
    entry3 = Entry(frame2, width=30)
    entry3.pack()
    button4 = Button(frame2, text="Submit Ratio", command=lambda: print(get_input(entry3)))
    button4.pack()

    button = Button(frame2, text="implement", command=lambda :calll())
    button.pack()

    def calll():
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],test_size=1 - float(entry3.get()) / 100,random_state=42)
        # Initialize SVM classifier
        KNN_classifier = KNeighborsClassifier(n_neighbors=int(entry.get()),metric=entry2.get())
        # Train the classifier
        KNN_classifier.fit(X_train, y_train)
        # Predict on the test set
        y_pred = KNN_classifier.predict(X_test)
        # Calculate accuracy
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        # Create a Tkinter window
        display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)
def DTC():
    global data
    frame2 = Toplevel()
    frame2.title("DTC Options")
    frame2.geometry("600x400")

    def get_input(entry):
        strategy = entry.get()
        return strategy

    label = Label(frame2, text="Enter Max Depth:")
    label.pack()
    entry = Entry(frame2, width=30)
    entry.pack()
    button = Button(frame2, text="Submit Depth", command=lambda: print(get_input(entry)))
    button.pack()

    label = Label(frame2, text="Enter Criterion (e.g., gini or entropy):")
    label.pack()
    entry2 = Entry(frame2, width=30)
    entry2.pack()
    button = Button(frame2, text="Submit Criterion", command=lambda: print(get_input(entry2)))
    button.pack()
    label3 = Label(frame2, text="Enter Train Ratio:")
    label3.pack()
    entry3 = Entry(frame2, width=30)
    entry3.pack()
    button4 = Button(frame2, text="Submit Criterion", command=lambda: print(get_input(entry3)))
    button4.pack()
    button = Button(frame2, text="implement", command=lambda:call())
    button.pack()

    def call():
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                            test_size=1 - float(entry3.get()) / 100,
                                                            random_state=42)

        # Initialize SVM classifier
        DTC_classifier = DecisionTreeClassifier(max_depth=int(entry.get()),criterion=entry2.get())
        # Train the classifier
        DTC_classifier.fit(X_train, y_train)
        # Predict on the test set
        y_pred = DTC_classifier.predict(X_test)
        # Calculate accuracy
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)
def label_encode(label_column):
    # Initialize LabelEncoder
    global data
    encoder = LabelEncoder()
    # Encode the label column
    om = pd.DataFrame(data.iloc[:, :-1])
    data = pd.concat([om, pd.DataFrame(encoder.fit_transform(label_column))], axis=1)
def encoding_option():
    global data
    frame2 = Toplevel()
    frame2.title("Encoding Option")
    frame2.geometry("600x800")
    frame2.configure(bg="#F2F3F4")  # Background color changed to light gray
    w = Button(frame2, text="Label Encoder", fg="#2C3E50", bg="#FFFFFF",
               width=15, command=lambda: label_encode(
            data.iloc[:, -1]))  # Text color changed to dark blue, background color changed to white
    w.pack(pady=120)
    d = Button(frame2, text="One Hot Encoder", fg="#2C3E50", bg="#FFFFFF",
               width=20,
               command=lambda: one_hot_encode())  # Text color changed to dark blue, background color changed to white
    d.pack(pady=210)
# smoteeeeeeee
# def smote():
#     global data
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#
#     # Instantiate SMOTE
#     smote = SMOTE(random_state=42)
#
#     # Apply SMOTE to generate synthetic samples
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#
#     # Concatenate resampled features and target variable into a DataFrame
#     data= pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['target'])], axis=1)
#
#     # Now 'resampled_data' contains the resampled dataset with synthetic samples
# ##
def SVM_C():
    global data
    frame2 = Toplevel()
    frame2.title("SVM Options")
    frame2.geometry("600x200")

    def get_input(entry):
        strategy = entry.get()
        return strategy

    label = Label(frame2, text="Enter Kernel:")
    label.pack()

    entry = Entry(frame2, width=30)
    entry.pack()

    button = Button(frame2, text="Submit Strategy", command=lambda: print(get_input(entry)))
    button.pack()

    entry2 = Entry(frame2, width=30)
    entry2.pack()

    button = Button(frame2, text="Submit Strategy", command=lambda: print(get_input(entry2)))
    button.pack()
    button = Button(frame2, text="implement", command=lambda:call())
    button.pack()

    def call():
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                            test_size=1 - float(entry2.get()) / 100,
                                                            random_state=42)
        # Initialize SVM classifier
        svm_classifier = SVC(kernel=entry.get())
        # Train the classifier
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)
def call_linear_regression():
    global data
    # Convert feature names to strings
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                        test_size=0.2, random_state=42)
    X_train.columns = X_train.columns.astype(str)
    linear_reg = LinearRegression()
    # Train the model
    linear_reg.fit(X_train, y_train)
    # Predict on the test set
    y_pred = linear_reg.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    # Display evaluation metrics
    display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)
def call_logistic_regression():
    global data
    # Convert feature names to strings
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                        test_size=0.2, random_state=42)
    X_train.columns = X_train.columns.astype(str)
    logistic_reg = LogisticRegression()
    # Train the model
    logistic_reg.fit(X_train, y_train)
    # Predict on the test set
    y_pred = logistic_reg.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    # Display evaluation metrics
    display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)
def classification():
    global data
    def enable_encoding_button(h=None):
        h.config(state="normal")
    frame2 = Toplevel()
    frame2.title("CSV Screen")
    frame2.geometry("600x400")
    frame2.configure(bg="#F2F3F4")  # Background color changed to light gray
    welcome_label = Label(frame2, text="classification Page", fg='#2C3E50', bg="#F2F3F4",
                          font=("Arial", 14))  # Text color changed to dark blue
    welcome_label.pack()
    q = Button(frame2, text="SVM", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
               command=SVM_C)  # Text color changed to dark blue, background color changed to white
    q.pack()
    q = Button(frame2, text="decision tree classifier", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
               command=DTC)  # Text color changed to dark blue, background color changed to white
    q.pack()
    q = Button(frame2, text="K Neighbors Classifier", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
               command=KNN)  # Text color changed to dark blue, background color changed to white
    q.pack()
def Regression():
    global data

    def enable_encoding_button(h=None):
        h.config(state="normal")

    frame2 = Toplevel()
    frame2.title("Regression Screen")
    frame2.geometry("400x400")
    frame2.configure(bg="#F2F3F4")  # Background color changed to light gray
    welcome_label = Label(frame2, text="Regression Page", fg='#2C3E50', bg="#F2F3F4",
                          font=("Arial", 14))  # Text color changed to dark blue
    welcome_label.pack()
    q = Button(frame2, text="Linear Regression", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
               command=call_linear_regression)  # Text color changed to dark blue, background color changed to white
    q.pack()
    q = Button(frame2, text="Logistic Regression", fg="#2C3E50", bg="#FFFFFF", width=20, height=3,
               command=call_logistic_regression)  # Text color changed to dark blue, background color changed to white
    q.pack()

def after_data():
    read_csv()
    bt2.config(state="normal")
    bt3.config(state="normal")
    bt4.config(state="normal")
    prin.config(state="normal")

bt1 = Button(a, text="select data file", fg='#FFFFFF', bg='#2C3E50', cursor='arrow', font=("Arial", 14), width='18',height='3', command=after_data)  # Text color changed to white, background color changed to dark blue
bt1.pack()

bt2 = Button(a, state= "disabled" ,text="Pre Processing", fg='#FFFFFF', bg='#2C3E50', cursor='arrow', font=("Arial", 14), width='18',height='3', command=display)  # Text color changed to white, background color changed to dark blue
bt2.pack()

bt3 = Button(a, state= "disabled" ,text="Classification", fg='#FFFFFF', bg='#2C3E50', cursor='arrow', font=("Arial", 14), width='18',height='3',command=classification)  # Text color changed to white, background color changed to dark blue
bt3.pack()
bt4 = Button(a, state= "disabled" ,text="Regression", fg='#FFFFFF', bg='#2C3E50', cursor='arrow', font=("Arial", 14), width='18',height='3',command=Regression)  # Text color changed to white, background color changed to dark blue
bt4.pack()
prin = Button(a, state= "disabled" ,text="show data head", fg="#2C3E50", bg="#FFFFFF", width=20,
                  height=3,
                  command=print_data)  # Text color changed to dark blue, background color changed to white
prin.pack()
a.mainloop()
