from typing import Literal
import tkinter as tk
import pandas as pd
from numpy import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .shared import DataModel



# ------------------- Pre Processing -------------------

def simple_imputer(data_model: DataModel, strategy):

    # Separate numeric and non-numeric columns
    numeric_cols = data_model.df.select_dtypes(include='number').columns
    non_numeric_cols = data_model.df.select_dtypes(exclude='number').columns

    # Define the imputers for different column types
    numeric_imputer = SimpleImputer(strategy=strategy)
    
    non_numeric_imputer = SimpleImputer(strategy="most_frequent")
    
    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_imputer, numeric_cols),
            ('cat', non_numeric_imputer, non_numeric_cols)
        ],
        remainder='passthrough'
    )
    
    # Apply the transformations
    transformed_data = preprocessor.fit_transform(data_model.df)
    
    # Reconstruct the DataFrame with the original column order
    all_columns = list(numeric_cols) + list(non_numeric_cols)
    data_model.df = pd.DataFrame(transformed_data, columns=all_columns)




def label_encode(data_model: DataModel):
    
    # Ensure columns are selected
    if not data_model.selected_col:
        print("No columns selected for label encoding")
    
    # Extract selected columns
    X = data_model.df[data_model.selected_col]
    
    # Apply label encoding to each selected column
    for col in X.columns:
        data_model.df[col] = LabelEncoder().fit_transform(X[col]).astype(int)

def one_hot_encode(data_model: DataModel):
    # Ensure columns are selected
    if not data_model.selected_col:
        raise ValueError("No columns selected for one-hot encoding")
    
    # Extract selected columns
    X = data_model.df[data_model.selected_col]
    
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create ColumnTransformer only for selected categorical columns
    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(), categorical_columns)],
        remainder='passthrough'
    )

    # Transform selected features
    transformed_features = ct.fit_transform(data_model.df)

    # Get feature names
    feature_names = ct.get_feature_names_out()

    # Create DataFrame with transformed features
    data_model.df = pd.DataFrame(transformed_features, columns=feature_names)


def min_max(data_model: DataModel):
    scaler = MinMaxScaler()
    data_model.df = pd.DataFrame(scaler.fit_transform(data_model.df))

def delete_selected(data_model: DataModel):
    data_model.df = data_model.df.drop(columns=data_model.selected_col, axis=1)


def smote(data_model: DataModel,frame):

    if len(data_model.selected_col) == 1:    
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(frame, text="please select only one column")
        return

    # tk.Label(frame, text="Before OverSampling # 1 =\n{}".format(sum(X == 1))).pack()
    # tk.Label(frame, text="Before OverSampling # 0  =\n{}".format(sum(y == 0))).pack()

    X_resampled, y_resampled = SMOTE().fit_resample(X, y)

    data_model.df = pd.concat([X_resampled,y_resampled],axis=1)

    # tk.Label(frame, text="After OverSampling # 1 =\n{}".format(sum(y_resampled == 1))).pack()
    # tk.Label(frame, text="After OverSampling # 0 =\n{}".format(sum(y_resampled == 0))).pack()

# ------------------- Clasification ------------------------

def SVM_C(data_model: DataModel, kernel, size):
    if len(data_model.selected_col) == 1:
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(text="please select only one column")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    # Initialize SVM classifier
    svm_classifier = SVC(kernel=kernel)
    # Train the classifier
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    display_evaluation_metrics(y_test, y_pred, "c")


def KNN(data_model: DataModel, n, metric, size):
    if len(data_model.selected_col) == 1:
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(text="please select only one column")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    # Initialize SVM classifier
    KNN_classifier = KNeighborsClassifier(n_neighbors=n, metric=metric)
    # Train the classifier
    KNN_classifier.fit(X_train, y_train)
    # Predict on the test set
    y_pred = KNN_classifier.predict(X_test)
    # Calculate accuracy
    display_evaluation_metrics(y_test, y_pred, "c")

def logistic_regression(data_model: DataModel, size):

    if len(data_model.selected_col) == 1:    
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(text="please select only one column")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    logistic_reg = LogisticRegression()
    # Train the model
    logistic_reg.fit(X_train, y_train)
    # Predict on the test set
    y_pred = logistic_reg.predict(X_test)

    display_evaluation_metrics(y_test, y_pred, "c")
    
def DTC(data_model: DataModel,depth, metric, size):
    if len(data_model.selected_col) == 1:
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(text="please select only one column")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    # Initialize SVM classifier
    DTC_classifier = DecisionTreeClassifier(max_depth=depth, criterion=metric)
    # Train the classifier
    DTC_classifier.fit(X_train, y_train)
    # Predict on the test set
    y_pred = DTC_classifier.predict(X_test)
    # Calculate accuracy
    display_evaluation_metrics(y_test, y_pred, "c")

def ANN(data_model: DataModel, selected_option,entry_test_size,entry_layers):
    if len(data_model.selected_col) == 1:
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(text="please select only one column")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= entry_test_size / 100)
    NN = MLPClassifier(hidden_layer_sizes=entry_layers, activation='relu', learning_rate=selected_option)
    NN.fit(X_train, y_train)
    y_pred = NN.predict(X_test)

    # Print the weights of the trained model
    #print(NN.coefs_)

    # Display the evaluation metrics
    display_evaluation_metrics(y_test, y_pred, "c")

# ------------------- Regression ------------------------

def linear_regression(data_model: DataModel, size):


    if len(data_model.selected_col) == 1:
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(text="please select only one column")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    linear_reg = LinearRegression()
    # Train the model
    linear_reg.fit(X_train, y_train)
    # Predict on the test set
    y_pred = linear_reg.predict(X_test)

    display_evaluation_metrics(y_test, y_pred, "r")

def SVM_r(data_model: DataModel, kernel, size):
    

    if len(data_model.selected_col) == 1:    
        X = data_model.df.drop(data_model.selected_col.pop(),axis=1)
        y = data_model.df[data_model.selected_col]
    else:
        tk.Message(text="please select only one column")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100)
    svr = SVR(kernel=kernel)

    svr.fit(X_train, y_train)

    y_pred = svr.predict(X_test)

    display_evaluation_metrics(y_test, y_pred, "r")

# ------------------- Clustring ------------------------

def KM(data_model: DataModel, frame, entry):
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


# ------------------- more ------------------------
def display_evaluation_metrics(y_test, y_pred, type: Literal['r', 'c']):
    # Create a Tkinter window
    window = tk.Tk()
    window.geometry('600x400')  # Set window size
    window.title("Evaluation Metrics Display")

    # Create a frame to contain labels
    frame = tk.Frame(window)
    frame.pack(padx=20, pady=10)


    if type == "r":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        t = """Mean Absolute Error (MAE): {mae:.2f}
        Mean Squared Error (MSE): {mse:.2f}
        Root Mean Squared Error (RMSE): {rmse:.2f}
        R-squared (R²): {r2:.2f}""".format(mae=mae, mse=mse, rmse=rmse, r2=r2)

        tk.Label(frame, text = t).pack()

    if type == "c":

        accuracy = accuracy_score(y_test, y_pred)
        tk.Label(frame, text="Accuracy: {:.2f}%".format(accuracy)).pack()

        confusion_mat = confusion_matrix(y_test, y_pred)
        tk.Label(frame, text ="Confusion Matrix:\n{}".format(confusion_mat)).pack()

        clas_matric=classification_report(y_test, y_pred, zero_division=0)
        clas_matric = classification_report(y_test, y_pred)
        tk.Label(frame, text="classification report:\n{}".format(clas_matric)).pack()
    # Run the Tkinter event loop
    window.mainloop()


