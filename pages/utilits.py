from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd
import tkinter as tk
from .shared import DataModel
# ------------------- Pre Processing -------------------

def simple_imputer(data_model: DataModel, strategy):

    data_model.df.columns = data_model.df.columns.astype(str)

    imputer = SimpleImputer(strategy=strategy)

    data_model.df = pd.DataFrame(imputer.fit_transform(data_model.df))


def label_encode(data_model: DataModel):

    X = data_model.df.iloc[:, :-1]
    y = data_model.df.iloc[:, -1]

    y = LabelEncoder().fit_transform(y)

    data_model.df = pd.concat([X,pd.DataFrame(y,columns=['lable'])], axis=1)

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
    

