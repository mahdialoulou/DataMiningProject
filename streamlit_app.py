import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import io  # Importing the io module

st.title('Data Mining Project - Analyze, Clean, and Visualize Data')

# Description for the overall project
st.write("""
## Project Overview
This application helps users analyze, clean, and visualize data from a dataset. Each feature in the application includes a description of how to use it, the outcomes, and the interpretation of possible results.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Option to specify if the file contains a header
    header_option = st.selectbox("Does the file contain a header?", ("Yes", "No"))
    header = 0 if header_option == "Yes" else None

    # Option to specify the file delimiter
    delimiter = st.text_input("Enter the file delimiter (default is ',')", ',')

    # Read the file
    df = pd.read_csv(uploaded_file, delimiter=delimiter, header=header)

    # Display the first and last few lines of the dataframe
    st.write("### First few lines of the data")
    st.write("""
    #### Description
    This section allows users to preview the first and last few lines of the dataset to ensure correct loading.
    """)
    st.write(df.head())

    st.write("### Last few lines of the data")
    st.write(df.tail())

    # Provide a basic statistical summary
    st.write("### Statistical Summary")
    st.write("""
    #### Description
    This section provides a basic statistical summary of the dataset, including measures such as mean, median, standard deviation, and quartiles for numeric columns.
    """)
    st.write(df.describe())

    # Basic information about the data
    st.write("### Data Information")
    st.write("""
    #### Description
    This section provides basic information about the dataset, including the number of rows and columns, data types of each column, and memory usage.
    """)
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Display the number of missing values per column
    st.write("### Missing Values per Column")
    st.write("""
    #### Description
    This section displays the number of missing values in each column of the dataset. Identifying columns with missing values is a crucial step before handling them.
    """)
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Values']
    st.write(missing_values)

    # Outlier Detection
    st.write("### Outlier Detection")
    st.write("""
    #### Description
    Outliers are data points that differ significantly from other observations. Detecting and handling outliers can improve the quality of data analysis. This application provides two methods for outlier detection:
    - **Z-score Method**: Calculates the Z-score for each data point. Data points with a Z-score greater than 3 are considered outliers.
    - **IQR Method**: Calculates the Interquartile Range (IQR). Data points outside 1.5 times the IQR from the first and third quartiles are considered outliers.

    ##### How to Use
    Select the outlier detection method from the dropdown. The application will display the number of outliers detected in each numeric column.

    ##### Interpretation
    - **Z-score Method**: A high Z-score indicates a data point that is far from the mean.
    - **IQR Method**: Data points outside the 1.5*IQR range are considered outliers.
    """)
    method = st.selectbox("Select the outlier detection method", ["Z-score", "IQR"])

    # Filter numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    if method == "Z-score":
        z_scores = np.abs((df_numeric - df_numeric.mean()) / df_numeric.std())
        outliers = (z_scores > 3).sum().reset_index()
        outliers.columns = ['Column', 'Outliers']
        st.write(outliers)
    elif method == "IQR":
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum().reset_index()
        outliers.columns = ['Column', 'Outliers']
        st.write(outliers)

    # Handling Missing Values
    st.write("### Handling Missing Values")
    st.write("""
    #### Description
    Handling missing values is essential to ensure the quality and completeness of the dataset. This application provides several methods to handle missing values:
    - **Delete rows**: Removes rows containing 70% or more missing values.
    - **Delete columns**: Allows users to select and remove columns with missing values.
    - **Fill with mean**: Replaces missing values in numeric columns with the mean of the column.
    - **Fill with median**: Replaces missing values in numeric columns with the median of the column.
    - **Fill with mode**: Replaces missing values with the mode of the column.
    - **KNN Imputation**: Uses K-Nearest Neighbors to impute missing values.

    ##### How to Use
    Select the method for handling missing values from the dropdown. The application will display the dataset after applying the selected method.

    ##### Interpretation
    - **Delete rows/columns**: Simplifies the dataset but may result in loss of data.
    - **Fill with mean/median/mode**: Maintains dataset size but may introduce bias.
    - **KNN Imputation**: Provides a more sophisticated way to handle missing values by using the similarity of data points.
    """)
    missing_option = st.selectbox("Select missing values handling method",
                                  ["Delete rows with 70% missing values", "Delete columns", "Fill with mean",
                                   "Fill with median", "Fill with mode", "KNN Imputation"])

    if missing_option == "Delete rows with 70% missing values":
        threshold = 0.7 * df.shape[1]
        df_cleaned = df.dropna(thresh=threshold)
    elif missing_option == "Delete columns":
        columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
        df_cleaned = df.drop(columns=columns_to_drop)
    elif missing_option == "Fill with mean":
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    elif missing_option == "Fill with median":
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    elif missing_option == "Fill with mode":
        df_cleaned = df.fillna(df.mode().iloc[0])
    elif missing_option == "KNN Imputation":
        df_numeric = df.select_dtypes(include=[np.number])
        imputer = KNNImputer(n_neighbors=3)
        df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns, index=df_numeric.index)
        df_cleaned = df.copy()
        df_cleaned[df_numeric.columns] = df_numeric_imputed

    st.write("### Data after handling missing values")
    st.write(df_cleaned)

    # Data Normalization
    st.write("### Data Normalization")
    st.write("""
    #### Description
    Normalization is the process of scaling data to a standard range. This application provides two methods for normalization:
    - **Min-Max Normalization**: Scales data to a range of [0, 1].
    - **Z-score Standardization**: Scales data to have a mean of 0 and a standard deviation of 1.

    ##### How to Use
    Select the normalization method from the dropdown. The application will display the dataset after applying the selected normalization method.

    ##### Interpretation
    - **Min-Max Normalization**: Useful for algorithms that require data within a specific range.
    - **Z-score Standardization**: Useful for algorithms that assume data is normally distributed.
    """)
    normalization_option = st.selectbox("Select normalization method",
                                        ["Min-Max Normalization", "Z-score Standardization"])

    if normalization_option == "Min-Max Normalization":
        scaler = MinMaxScaler()
    elif normalization_option == "Z-score Standardization":
        scaler = StandardScaler()

    df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned.select_dtypes(include=[np.number])),
                                 columns=df_cleaned.select_dtypes(include=[np.number]).columns)
    df_normalized[df_cleaned.select_dtypes(exclude=[np.number]).columns] = df_cleaned.select_dtypes(exclude=[np.number])

    st.write("### Data after normalization")
    st.write(df_normalized)
