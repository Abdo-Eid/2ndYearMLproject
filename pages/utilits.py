
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd

import tkinter as tk
# ------------------- Pre Processing -------------------

def simpel_imputer():
    global data
    frame2 = tk.Toplevel()
    frame2.title("Simple Imputer Option")
    frame2.geometry("400x300")

    def get_input():
        strategy = selected_option.get()
        simp_nan(strategy)

    tk.Label(frame2, text="Enter Strategy:").pack()
    options = ["mean", "most_frequent", "median"]
    selected_option = tk.StringVar(frame2)
    selected_option.set(options[0])
    option_menu = tk.OptionMenu(frame2, selected_option, *options)
    option_menu.pack(padx=20, pady=20)

    tk.Button(frame2, text="Deal With NaN", command=get_input).pack()

def simp_nan(strateg):
    # Initialize SimpleImputer
    global data
    imputer = SimpleImputer(strategy=strateg)
    # Impute missing values in the 'age' column
    data = imputer.fit_transform(data)

def min_max():
    global data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data))
    print(data)


def apply_smote():
    global data
    X = data.iloc[:, :-1]  # Take all columns except the last one
    y = data.iloc[:, -1]  # Take only the last column as the target variable

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)],
                               axis=1)
    

def encoding_option():
    global data
    frame2 = tk.Toplevel()
    frame2.title("Encoding Option")
    frame2.geometry("600x800")
    frame2.configure(bg="#F2F3F4")  # Background color changed to light gray
    w = tk.Button(frame2, text="Label Encoder", fg="#2C3E50", bg="#FFFFFF",
               width=15, command=lambda: label_encode(
            data.iloc[:, -1]))  # Text color changed to dark blue, background color changed to white
    w.pack(pady=120)
    d = tk.Button(frame2, text="One Hot Encoder", fg="#2C3E50", bg="#FFFFFF",
               width=20,
               command=lambda: one_hot_encode())  # Text color changed to dark blue, background color changed to white
    d.pack(pady=210)

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

def label_encode(label_column):
    # Initialize LabelEncoder
    global data
    encoder = LabelEncoder()
    # Encode the label column
    om = pd.DataFrame(data.iloc[:, :-1])
    data = pd.concat([om, pd.DataFrame(encoder.fit_transform(label_column))], axis=1)

