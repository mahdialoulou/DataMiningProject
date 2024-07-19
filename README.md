# Data Mining Project

## Overview

The goal of this project is to develop an interactive web application using Streamlit to analyze, clean, encode, normalize, visualize, cluster, and predict data from a dataset. Each feature in the application includes a description of how to use it, the outcomes, and the interpretation of possible results.

## Features

- **Data Loading**: Load your own dataset in CSV format.
- **Data Description**: Preview the first and last few lines of the dataset to ensure correct loading.
- **Statistical Summary**: Basic statistical summary of the dataset, including measures such as mean, median, standard deviation, and quartiles for numeric columns.
- **Missing Values Handling**: Several methods to handle missing values such as deletion, mean/median/mode imputation, and KNN imputation.
- **Outlier Detection and Removal**: Detect and optionally remove outliers using Z-score or IQR methods.
- **Data Encoding**: Various encoding methods including label encoding, one-hot encoding, and multi-label encoding.
- **Data Normalization**: Several normalization methods such as Min-Max Scaling, Z-score Standardization, Robust Scaling, and more.
- **Data Visualization**: Create various types of plots to visualize the data.
- **Clustering**: Implement clustering algorithms such as K-Means, DBSCAN, Agglomerative Clustering, and Self-Organizing Maps (SOM).
- **Prediction**: Implement regression and classification models.
- **PCA Analysis**: Perform PCA and visualize the results.

## Installation

To run this project, you need to have Python installed on your machine. Follow the steps below to set up and run the Streamlit application:

1. **Clone the repository:**

    `sh
    git clone https://github.com/mahdialoulou/DataMiningProject.git
    cd DataMiningProject
    `

2. **Create a virtual environment:**

    `sh
    python -m venv venv
    `

3. **Activate the virtual environment:**

    - For Windows:

        `sh
        .\\venv\\Scripts\\activate
        `

    - For macOS and Linux:

        `sh
        source venv/bin/activate
        `

4. **Install the dependencies:**

    `sh
    pip install -r requirements.txt
    `

5. **Run the Streamlit application:**

    `sh
    streamlit run streamlit_app.py
    `

6. **Open your browser**: The Streamlit application will usually run on http://localhost:8501.

## Usage

1. **Upload your CSV file**: Use the file uploader to upload your dataset in CSV format.
2. **Data Exploration**: Preview the first and last few lines of your dataset and view a statistical summary.
3. **Data Cleaning**: Handle missing values and detect/remove outliers.
4. **Data Encoding**: Apply different encoding methods to categorical data.
5. **Data Normalization**: Normalize your data using various scaling methods.
6. **Data Visualization**: Create plots to visualize your data.
7. **Clustering**: Apply clustering algorithms to group your data.
8. **Prediction**: Use regression or classification models to make predictions based on your data.
9. **PCA Analysis**: Perform PCA to reduce the dimensionality of your data and visualize the results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

