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
