from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import tkinter as tk
from .shared import DataModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ------------------- Pre Processing -------------------

def simple_imputer(data_model: DataModel, strategy):
    data_model.df.columns = data_model.df.columns.astype(str)

    imputer = SimpleImputer(strategy=strategy)

    data_model.df = pd.DataFrame(imputer.fit_transform(data_model.df))


def label_encode(data_model: DataModel):
    X = data_model.df.iloc[:, :-1]
    y = data_model.df.iloc[:, -1]

    y = LabelEncoder().fit_transform(y)

    data_model.df = pd.concat([X, pd.DataFrame(y, columns=['lable'])], axis=1)


def one_hot_encode(data_model: DataModel):
    categorical_columns = data_model.df.select_dtypes(include=['object']).columns

    ct = ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
    data_model.df = pd.DataFrame(ct.fit_transform(data_model.df))


def min_max(data_model: DataModel):
    scaler = MinMaxScaler()
    data_model.df = pd.DataFrame(scaler.fit_transform(data_model.df))


def apply_smote():
    global data
    X = data.iloc[:, :-1]  # Take all columns except the last one
    y = data.iloc[:, -1]  # Take only the last column as the target variable

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)],
                     axis=1)


# ------------------- Clasification ----------------
def SVM_C(data_model: DataModel, kernel, size):
    X = data_model.df.iloc[:, :-1]
    y = data_model.df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    # Initialize SVM classifier
    svm_classifier = SVC(kernel=kernel)
    # Train the classifier
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)


# def KNN(data_model: DataModel,strategy):
#     label = Label(frame2, text="Enter Number of Neighbors:")
#     label.pack()
#     entry = Entry(frame2, width=30)
#     entry.pack()
#     button = Button(frame2, text="Submit Number of Neighbors", command=lambda: print(get_input(entry)))
#     button.pack()

#     label = Label(frame2, text="Enter Distance Metric (metric eg :: (hamming,mahalanobis,chebyshev,....):")
#     label.pack()
#     entry2 = Entry(frame2, width=30)
#     entry2.pack()
#     button = Button(frame2, text="Submit Distance Metric", command=lambda: print(get_input(entry2)))
#     button.pack()

#     label3 = Label(frame2, text="Enter Train Ratio:")
#     label3.pack()
#     entry3 = Entry(frame2, width=30)
#     entry3.pack()
#     button4 = Button(frame2, text="Submit Ratio", command=lambda: print(get_input(entry3)))
#     button4.pack()

#     button = Button(frame2, text="implement", command=lambda :calll())
#     button.pack()

#     def calll():
#         X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],test_size=1 - float(entry3.get()) / 100,random_state=42)
#         # Initialize SVM classifier
#         KNN_classifier = KNeighborsClassifier(n_neighbors=int(entry.get()),metric=entry2.get())
#         # Train the classifier
#         KNN_classifier.fit(X_train, y_train)
#         # Predict on the test set
#         y_pred = KNN_classifier.predict(X_test)
#         # Calculate accuracy
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         confusion_mat = confusion_matrix(y_test, y_pred)
#         accuracy = accuracy_score(y_test, y_pred)
#         # Create a Tkinter window
#         display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)
# def DTC():
#     global data
#     frame2 = Toplevel()
#     frame2.title("DTC Options")
#     frame2.geometry("600x400")

#     def get_input(entry):
#         strategy = entry.get()
#         return strategy

#     label = Label(frame2, text="Enter Max Depth:")
#     label.pack()
#     entry = Entry(frame2, width=30)
#     entry.pack()
#     button = Button(frame2, text="Submit Depth", command=lambda: print(get_input(entry)))
#     button.pack()

#     label = Label(frame2, text="Enter Criterion (e.g., gini or entropy):")
#     label.pack()
#     entry2 = Entry(frame2, width=30)
#     entry2.pack()
#     button = Button(frame2, text="Submit Criterion", command=lambda: print(get_input(entry2)))
#     button.pack()
#     label3 = Label(frame2, text="Enter Train Ratio:")
#     label3.pack()
#     entry3 = Entry(frame2, width=30)
#     entry3.pack()
#     button4 = Button(frame2, text="Submit Criterion", command=lambda: print(get_input(entry3)))
#     button4.pack()
#     button = Button(frame2, text="implement", command=lambda:call())
#     button.pack()

#     def call():
#         X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
#                                                             test_size=1 - float(entry3.get()) / 100,
#                                                             random_state=42)

#         # Initialize SVM classifier
#         DTC_classifier = DecisionTreeClassifier(max_depth=int(entry.get()),criterion=entry2.get())
#         # Train the classifier
#         DTC_classifier.fit(X_train, y_train)
#         # Predict on the test set
#         y_pred = DTC_classifier.predict(X_test)
#         # Calculate accuracy
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         confusion_mat = confusion_matrix(y_test, y_pred)
#         accuracy = accuracy_score(y_test, y_pred)
#         display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat)


def display_evaluation_metrics(accuracy, precision, recall, f1, confusion_mat):
    # Create a Tkinter window
    window = tk.Tk()
    window.geometry('600x400')  # Set window size
    window.title("Evaluation Metrics Display")

    # Create a frame to contain labels
    frame = tk.Frame(window)
    frame.pack(padx=20, pady=10)

    # Accuracy label
    accuracy_label = tk.Label(frame, text="Accuracy: {:.2f}%".format(accuracy * 100))
    accuracy_label.pack()

    # Precision label
    precision_label = tk.Label(frame, text="Precision: {:.2f}".format(precision))
    precision_label.pack()

    # Recall label
    recall_label = tk.Label(frame, text="Recall: {:.2f}".format(recall))
    recall_label.pack()

    # F1-score label
    f1_label = tk.Label(frame, text="F1-score: {:.2f}".format(f1))
    f1_label.pack()

    # Confusion matrix label
    confusion_label = tk.Label(frame, text="Confusion Matrix:\n{}".format(confusion_mat))
    confusion_label.pack()

    # Run the Tkinter event loop
    window.mainloop()