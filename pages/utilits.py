from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import tkinter as tk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from .shared import DataModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report  # Removed the backslash
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# ------------------- Pre Processing -------------------

def simple_imputer(data_model: DataModel, strategy):
    data_model.df.columns = data_model.df.columns.astype(str)

    imputer = SimpleImputer(strategy=strategy)

    data_model.df = pd.DataFrame(imputer.fit_transform(data_model.df))


def label_encode(data_model: DataModel):
    X = data_model.df.iloc[:, :-1]
    y = data_model.df.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    y = y.astype(int)
    data_model.df = pd.concat([X, pd.DataFrame(y, columns=['label'])], axis=1)


def one_hot_encode(data_model: DataModel):
    categorical_columns = data_model.df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col not in ["label"]]

    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(), categorical_columns)],
        remainder='passthrough'
    )

    transformed_features = ct.fit_transform(data_model.df)

    feature_names = ct.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed_features, columns=feature_names)

    # Add the label column back to the transformed DataFrame
    transformed_df['label'] = data_model.df['label'].astype(int)
    data_model.df = transformed_df


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
    display_evaluation_metrics(y_test, y_pred, "c")


def KNN(data_model: DataModel, n, metric, size):
    X = data_model.df.iloc[:, :-1]
    y = data_model.df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    # Initialize SVM classifier
    KNN_classifier = KNeighborsClassifier(n_neighbors=n, metric=metric)
    # Train the classifier
    KNN_classifier.fit(X_train, y_train)
    # Predict on the test set
    y_pred = KNN_classifier.predict(X_test)
    # Calculate accuracy
    display_evaluation_metrics(y_test, y_pred, "c")


def display_evaluation_metrics(y_test, y_pred, type):
    # Create a Tkinter window
    window = tk.Tk()
    window.geometry('600x400')  # Set window size
    window.title("Evaluation Metrics Display")

    # Create a frame to contain labels
    frame = tk.Frame(window)
    frame.pack(padx=20, pady=10)

    accuracy = accuracy_score(y_test, y_pred)
    tk.Label(frame, text="Accuracy: {:.2f}%".format(accuracy * 100)).pack()

    if type == "r":
        precision = precision_score(y_test, y_pred)
        tk.Label(frame, text="Precision: {:.2f}".format(precision)).pack()

        recall = recall_score(y_test, y_pred)
        tk.Label(frame, text="Recall: {:.2f}".format(recall)).pack()

        f1 = f1_score(y_test, y_pred)
        tk.Label(frame, text="F1-score: {:.2f}".format(f1)).pack()

    confusion_mat = confusion_matrix(y_test, y_pred)
    tk.Label(frame, text="Confusion Matrix:\n{}".format(confusion_mat)).pack()

    if type == "c":
        clas_matric = classification_report(y_test, y_pred)
        tk.Label(frame, text="classification report:\n{}".format(clas_matric)).pack()
    # Run the Tkinter event loop
    window.mainloop()


def clustring(data_model: DataModel, frame, entry):
    # Create a frame to contain labels
    data = data_model.df.values  # Convert DataFrame to NumPy array
    km = KMeans(n_clusters=entry)
    km.fit(data)
    labels = km.labels_
    centers = km.cluster_centers_

    print("Cluster centers:", centers)
    tk.Label(frame, text="Cluster centers:\n{}".format(centers)).pack()
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title(f'KMeans Clustering with {entry} Clusters')
    plt.show()


def ANN(data_model: DataModel, selected_option,entry_test_size,entry_layers):
    X = data_model.df.iloc[:, :-1]
    y = data_model.df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= entry_test_size / 100)
    NN = MLPClassifier(hidden_layer_sizes=entry_layers, activation='relu', learning_rate=selected_option)
    NN.fit(X_train, y_train)
    y_pred = NN.predict(X_test)

    # Print the weights of the trained model
    #print(NN.coefs_)

    # Display the evaluation metrics
    display_evaluation_metrics(y_test, y_pred, "c")
def smote(data_model: DataModel, entry_test_size,frame):

    X = data_model.df.iloc[:, :-1]
    y = data_model.df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=entry_test_size / 100)

    tk.Label(frame, text="Before OverSampling # 1 =\n{}".format(sum(y_train == 1))).pack()
    print("Before OverSampling # 0 =", sum(y_train == 0))
    tk.Label(frame, text="Before OverSampling # 0  =\n{}".format(sum(y_train == 0))).pack()

    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    print("-----------------------------------------")
    tk.Label(frame, text="After OverSampling # 1 =\n{}".format(sum(y_resampled == 1))).pack()
    tk.Label(frame, text="After OverSampling # 0 =\n{}".format(sum(y_resampled == 0))).pack()

