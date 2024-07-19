import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer, \
    QuantileTransformer, LabelEncoder, MultiLabelBinarizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom  # Assuming you have the MiniSom library installed
import io

st.title('Data Mining Project - Analyze, Clean, Encode, Normalize, Visualize, Cluster, and Predict')

# Description for the overall project
st.write("""
## Project Overview
This application helps users analyze, clean, encode, normalize, visualize, cluster, and predict data from a dataset. Each feature in the application includes a description of how to use it, the outcomes, and the interpretation of possible results.
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

    # Option to delete specific columns
    st.write("### Delete Columns")
    st.write("""
    #### Description
    This section allows users to delete specific columns from the dataset.
    """)
    columns_to_delete = st.multiselect("Select columns to delete", df.columns.tolist())
    if columns_to_delete:
        df = df.drop(columns=columns_to_delete)
        st.write("### Data after deleting selected columns")
        st.write(df)

    # Display the number of missing values per column
    st.write("### Missing Values per Column")
    st.write("""
    #### Description
    This section displays the number of missing values in each column of the dataset. Identifying columns with missing values is a crucial step before handling them.
    """)
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Values']
    st.write(missing_values)

    # Outlier Detection and Removal
    st.write("### Outlier Detection and Removal")
    st.write("""
    #### Description
    Outliers are data points that differ significantly from other observations. Detecting and handling outliers can improve the quality of data analysis. This application provides two methods for outlier detection:
    - **Z-score Method**: Calculates the Z-score for each data point. Data points with a Z-score greater than 3 are considered outliers.
    - **IQR Method**: Calculates the Interquartile Range (IQR). Data points outside 1.5 times the IQR from the first and third quartiles are considered outliers.

    ##### How to Use
    Select the outlier detection method from the dropdown. The application will display the number of outliers detected in each numeric column. You can also choose to remove the detected outliers.

    ##### Interpretation
    - **Z-score Method**: A high Z-score indicates a data point that is far from the mean.
    - **IQR Method**: Data points outside the 1.5*IQR range are considered outliers.
    """)
    method = st.selectbox("Select the outlier detection method", ["Z-score", "IQR"])

    # Filter numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    if method == "Z-score":
        z_scores = np.abs((df_numeric - df_numeric.mean()) / df_numeric.std())
        outliers = (z_scores > 3)
    elif method == "IQR":
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR)))

    outlier_count = outliers.sum().reset_index()
    outlier_count.columns = ['Column', 'Outliers']
    st.write(outlier_count)

    remove_outliers = st.checkbox("Remove detected outliers?")
    if remove_outliers:
        df = df[~outliers.any(axis=1)]
        st.write("### Data after outlier removal")
        st.write(df)

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
        numeric_cols_to_impute = st.multiselect("Select numeric columns to apply KNN Imputation",
                                                df_numeric.columns.tolist())
        if numeric_cols_to_impute:
            df_numeric_to_impute = df[numeric_cols_to_impute]
            imputer = KNNImputer(n_neighbors=3)
            df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric_to_impute),
                                              columns=df_numeric_to_impute.columns, index=df_numeric_to_impute.index)
            df_cleaned = df.copy()
            df_cleaned[df_numeric_imputed.columns] = df_numeric_imputed
        else:
            st.write("No columns selected for KNN Imputation. Please select at least one column.")

    st.write("### Data after handling missing values")
    st.write(df_cleaned)

    # Encoding
    st.write("### Data Encoding")
    st.write("""
    #### Description
    Encoding is the process of converting categorical data into numerical format. This application provides several methods for encoding data:
    - **No Encoding**: Keeps the data as is.
    - **Label Encoding**: Converts each unique value in a column to a unique integer.
    - **One-Hot Encoding**: Converts each unique value in a column to a separate binary column.
    - **Multi-Label Encoding**: Converts a column of lists of labels into multiple binary columns.

    ##### How to Use
    Select the encoding method from the dropdown. Then select the columns to apply the encoding method on. The application will display the dataset after applying the selected encoding method.
    """)
    encoding_option = st.selectbox("Select encoding method",
                                   ["No Encoding", "Label Encoding", "One-Hot Encoding", "Multi-Label Encoding"])

    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    cols_to_encode = st.multiselect("Select columns to apply encoding", categorical_cols)

    if encoding_option == "Label Encoding":
        encoder = LabelEncoder()
        for col in cols_to_encode:
            df_cleaned[col] = encoder.fit_transform(df_cleaned[col])
    elif encoding_option == "One-Hot Encoding":
        df_cleaned = pd.get_dummies(df_cleaned, columns=cols_to_encode)
    elif encoding_option == "Multi-Label Encoding":
        multi_label_col = st.selectbox("Select column for Multi-Label Encoding", cols_to_encode)
        if multi_label_col:
            mlb = MultiLabelBinarizer()
            df_cleaned = df_cleaned.join(pd.DataFrame(mlb.fit_transform(df_cleaned.pop(multi_label_col)),
                                                      columns=mlb.classes_,
                                                      index=df_cleaned.index))

    st.write("### Data after encoding")
    st.write(df_cleaned)

    # Data Normalization
    st.write("### Data Normalization")
    st.write("""
    #### Description
    Normalization is the process of scaling data to a standard range. This application provides several methods for normalization:
    - **No Normalization**: Keeps data in its original scale.
    - **Min-Max Normalization**: Scales data to a range of [0, 1].
    - **Z-score Standardization**: Scales data to have a mean of 0 and a standard deviation of 1.
    - **Robust Scaling**: Scales data using statistics that are robust to outliers.
    - **Normalization**: Scales data to have unit norm.
    - **Power Transformation**: Applies a power transformation to make data more Gaussian-like.
    - **Quantile Transformation**: Transforms data to follow a uniform or normal distribution.

    ##### How to Use
    Select the normalization method from the dropdown. The application will display the dataset after applying the selected normalization method.

    ##### Interpretation
    - **No Normalization**: Keeps data in its original scale.
    - **Min-Max Normalization**: Useful for algorithms that require data within a specific range.
    - **Z-score Standardization**: Useful for algorithms that assume data is normally distributed.
    - **Robust Scaling**: Useful for data with outliers.
    - **Normalization**: Useful for making data have unit norm.
    - **Power Transformation**: Useful for making data more Gaussian-like.
    - **Quantile Transformation**: Useful for transforming data to follow a uniform or normal distribution.
    """)
    normalization_option = st.selectbox("Select normalization method",
                                        ["No Normalization", "Min-Max Normalization", "Z-score Standardization",
                                         "Robust Scaling", "Normalization", "Power Transformation",
                                         "Quantile Transformation"])

    numeric_cols_to_normalize = st.multiselect("Select numeric columns to apply normalization",
                                               df_cleaned.select_dtypes(include=[np.number]).columns.tolist())

    df_normalized = df_cleaned.copy()

    if normalization_option != "No Normalization" and numeric_cols_to_normalize:
        if normalization_option == "Min-Max Normalization":
            scaler = MinMaxScaler()
        elif normalization_option == "Z-score Standardization":
            scaler = StandardScaler()
        elif normalization_option == "Robust Scaling":
            scaler = RobustScaler()
        elif normalization_option == "Normalization":
            scaler = Normalizer()
        elif normalization_option == "Power Transformation":
            scaler = PowerTransformer()
        elif normalization_option == "Quantile Transformation":
            scaler = QuantileTransformer()

        df_normalized[numeric_cols_to_normalize] = scaler.fit_transform(df_cleaned[numeric_cols_to_normalize])

    st.write("### Data after normalization")
    st.write(df_normalized)

    # Visualization of the cleaned data
    st.write("### Data Visualization")
    st.write("""
    #### Description
    Visualizing data helps in understanding the distribution and relationships between features. This section provides options for creating various types of plots.
    """)

    # Number of graphs
    num_graphs = st.number_input("Select number of graphs to create", min_value=1, max_value=10, step=1, value=1)

    for i in range(num_graphs):
        st.write(f"### Graph {i + 1}")
        plot_type = st.selectbox(f"Select plot type for graph {i + 1}",
                                 ["Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Bar Plot", "Heatmap",
                                  "Pair Plot", "Violin Plot", "Pie Chart", "Area Plot", "Density Plot"],
                                 key=f"plot_type_{i}")
        x_column = st.selectbox(f"Select x-axis column for graph {i + 1}", df_cleaned.columns.tolist(),
                                key=f"x_column_{i}")
        y_column = st.selectbox(f"Select y-axis column for graph {i + 1} (if applicable)", df_cleaned.columns.tolist(),
                                key=f"y_column_{i}")
        color = st.color_picker(f"Select color for graph {i + 1}", "#69b3a2", key=f"color_{i}")
        if plot_type == "Scatter Plot":
            marker = st.selectbox(f"Select marker shape for graph {i + 1}", ["o", "s", "^", "D", "v"],
                                  key=f"marker_{i}")
        else:
            marker = "o"
        line_width = st.slider(f"Select line width for graph {i + 1}", min_value=0.5, max_value=5.0, step=0.1,
                               value=1.5, key=f"line_width_{i}")
        title = st.text_input(f"Enter title for graph {i + 1}", key=f"title_{i}")

        if plot_type == "Histogram":
            plt.figure(figsize=(10, 4))
            sns.histplot(df_cleaned[x_column], kde=True, color=color)
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Box Plot":
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df_cleaned[x_column], color=color)
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Scatter Plot":
            plt.figure(figsize=(10, 4))
            plt.scatter(df_cleaned[x_column], df_cleaned[y_column], color=color, marker=marker)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Line Plot":
            plt.figure(figsize=(10, 4))
            plt.plot(df_cleaned[x_column], df_cleaned[y_column], color=color, linewidth=line_width)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Bar Plot":
            plt.figure(figsize=(10, 4))
            sns.barplot(x=df_cleaned[x_column], y=df_cleaned[y_column], color=color)
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Heatmap":
            plt.figure(figsize=(10, 4))
            sns.heatmap(df_cleaned[[x_column, y_column]].corr(), annot=True, cmap="coolwarm")
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Pair Plot":
            sns.pairplot(df_cleaned[[x_column, y_column]])
            plt.suptitle(title, y=1.02)
            st.pyplot(plt)
        elif plot_type == "Violin Plot":
            plt.figure(figsize=(10, 4))
            sns.violinplot(x=df_cleaned[x_column], palette="Set2")
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Pie Chart":
            plt.figure(figsize=(8, 8))
            df_cleaned[x_column].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set2", len(
                df_cleaned[x_column].unique())))
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Area Plot":
            plt.figure(figsize=(10, 4))
            df_cleaned[[x_column, y_column]].plot.area(stacked=False, color=[color])
            plt.title(title)
            st.pyplot(plt)
        elif plot_type == "Density Plot":
            plt.figure(figsize=(10, 4))
            sns.kdeplot(df_cleaned[x_column], shade=True, color=color)
            plt.title(title)
            st.pyplot(plt)
        else:
            st.write("Please select the appropriate columns for the selected plot type.")

    # Choice to use clustering, prediction, or both
    task_option = st.selectbox("Select task to perform", ["Clustering", "Prediction", "Both"])

    if task_option in ["Clustering", "Both"]:
        # Clustering
        st.write("### Clustering")
        st.write("""
        #### Description
        This section provides various clustering algorithms to group the data into clusters. Available algorithms:
        - **K-Means**: Partitions data into K clusters.
        - **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise.
        - **Agglomerative Clustering**: Hierarchical clustering method.
        - **Self-Organizing Maps (SOM)**: Neural network-based clustering.

        ##### How to Use
        Select the clustering algorithm and configure the parameters. The application will display the clustering results and statistics.
        """)
        clustering_algorithm = st.selectbox("Select clustering algorithm",
                                            ["K-Means", "DBSCAN", "Agglomerative Clustering",
                                             "Self-Organizing Maps (SOM)"])

        if clustering_algorithm == "K-Means":
            n_clusters = st.slider("Select number of clusters (K)", 2, 10)
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(df_normalized.select_dtypes(include=[np.number]))
            df_normalized['Cluster'] = clusters
            st.write("### Clustering Results")
            st.write(df_normalized)
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(df_normalized.select_dtypes(include=[np.number]))
            plt.figure(figsize=(10, 4))
            sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=clusters, palette="viridis")
            plt.title("K-Means Clustering")
            plt.legend()
            st.pyplot(plt)
            st.write("Cluster Centers:")
            st.write(kmeans.cluster_centers_)
        elif clustering_algorithm == "DBSCAN":
            eps = st.slider("Select epsilon (eps)", 0.1, 10.0)
            min_samples = st.slider("Select minimum samples", 1, 10)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(df_normalized.select_dtypes(include=[np.number]))
            df_normalized['Cluster'] = clusters
            st.write("### Clustering Results")
            st.write(df_normalized)
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(df_normalized.select_dtypes(include=[np.number]))
            plt.figure(figsize=(10, 4))
            sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=clusters, palette="viridis")
            plt.title("DBSCAN Clustering")
            st.pyplot(plt)
            st.write("Cluster Density (number of points per cluster):")
            st.write(pd.Series(clusters).value_counts())
        elif clustering_algorithm == "Agglomerative Clustering":
            n_clusters = st.slider("Select number of clusters", 2, 10)
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = agglomerative.fit_predict(df_normalized.select_dtypes(include=[np.number]))
            df_normalized['Cluster'] = clusters
            st.write("### Clustering Results")
            st.write(df_normalized)
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(df_normalized.select_dtypes(include=[np.number]))
            plt.figure(figsize=(10, 4))
            sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=clusters, palette="viridis")
            plt.title("Agglomerative Clustering")
            st.pyplot(plt)
            st.write("Cluster Counts:")
            st.write(pd.Series(clusters).value_counts())
        elif clustering_algorithm == "Self-Organizing Maps (SOM)":
            som_size = st.slider("Select SOM grid size", 1, 10)
            som = MiniSom(som_size, som_size, df_normalized.select_dtypes(include=[np.number]).shape[1], sigma=0.3,
                          learning_rate=0.5)
            som.train_random(df_normalized.select_dtypes(include=[np.number]).to_numpy(), 100)
            win_map = som.win_map(df_normalized.select_dtypes(include=[np.number]).to_numpy())
            clusters = np.zeros(df_normalized.shape[0], dtype=int)
            for i, x in enumerate(df_normalized.select_dtypes(include=[np.number]).to_numpy()):
                winner = som.winner(x)
                if winner in win_map:
                    clusters[i] = list(win_map.keys()).index(winner)
            df_normalized['Cluster'] = clusters
            st.write("### Clustering Results")
            st.write(df_normalized)
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(df_normalized.select_dtypes(include=[np.number]))
            plt.figure(figsize=(10, 4))
            sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=clusters, palette="viridis")
            plt.title("Self-Organizing Maps (SOM) Clustering")
            st.pyplot(plt)
            st.write("Cluster Counts:")
            st.write(pd.Series(clusters).value_counts())

        # Clustering Evaluation
        st.write("#### Clustering Evaluation")
        st.write("Cluster statistics:")
        cluster_stats = df_normalized.groupby('Cluster').size().reset_index(name='Counts')
        st.write(cluster_stats)

        if clustering_algorithm == "K-Means":
            st.write("Cluster Centers:")
            st.write(kmeans.cluster_centers_)
        elif clustering_algorithm == "DBSCAN":
            st.write("Cluster Density (number of points per cluster):")
            st.write(pd.Series(clusters).value_counts())

    if task_option in ["Prediction", "Both"]:
        # Prediction
        st.write("### Prediction")
        st.write("""
        #### Description
        This section provides options for regression and classification prediction models. Available models:
        - **Linear Regression**: A linear approach to modeling the relationship between a dependent variable and one or more independent variables.
        - **Random Forest Classifier**: An ensemble learning method for classification that operates by constructing a multitude of decision trees.

        ##### How to Use
        Select the prediction model and configure the parameters. The application will display the prediction results and model evaluation metrics.
        """)
        prediction_task = st.selectbox("Select prediction task", ["Regression", "Classification"])

        if prediction_task == "Regression":
            target_col = st.selectbox("Select target column for regression", df_normalized.columns)
            if target_col:
                X = df_normalized.drop(columns=[target_col])
                y = df_normalized[target_col]
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)
                st.write("### Regression Results")
                st.write("Mean Squared Error:", mean_squared_error(y, predictions))
                plt.figure(figsize=(10, 4))
                plt.scatter(y, predictions, color='blue')
                plt.plot(y, y, color='red')
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("Linear Regression")
                st.pyplot(plt)
        elif prediction_task == "Classification":
            target_col = st.selectbox("Select target column for classification", df_normalized.columns)
            if target_col:
                X = df_normalized.drop(columns=[target_col])
                y = df_normalized[target_col]
                model = RandomForestClassifier()
                model.fit(X, y)
                predictions = model.predict(X)
                st.write("### Classification Results")
                st.write("Accuracy Score:", accuracy_score(y, predictions))
                st.write("Precision Score:", precision_score(y, predictions, average='weighted'))
                st.write("Recall Score:", recall_score(y, predictions, average='weighted'))
                st.write("F1 Score:", f1_score(y, predictions, average='weighted'))
                plt.figure(figsize=(10, 4))
                sns.heatmap(pd.crosstab(y, predictions, rownames=['Actual'], colnames=['Predicted']), annot=True,
                            fmt="d", cmap="YlGnBu")
                plt.title("Random Forest Classifier")
                st.pyplot(plt)

        # Prediction Evaluation
        st.write("#### Prediction Evaluation")
        if prediction_task == "Regression":
            st.write("Mean Squared Error:", mean_squared_error(y, predictions))
        elif prediction_task == "Classification":
            st.write("Accuracy Score:", accuracy_score(y, predictions))
            st.write("Precision Score:", precision_score(y, predictions, average='weighted'))
            st.write("Recall Score:", recall_score(y, predictions, average='weighted'))
            st.write("F1 Score:", f1_score(y, predictions, average='weighted'))

    # PCA Analysis
    st.write("### PCA Analysis")
    st.write("""
    #### Description
    PCA (Principal Component Analysis) is used to reduce the dimensionality of data while preserving as much variability as possible. This section provides an option to perform PCA and visualize the results.
    """)
    pca_analysis = st.checkbox("Perform PCA Analysis")
    if pca_analysis:
        n_components = st.slider("Select number of PCA components", 1,
                                 min(len(df_normalized.select_dtypes(include=[np.number]).columns), 10), 2)
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(df_normalized.select_dtypes(include=[np.number]))
        pca_df = pd.DataFrame(pca_components, columns=[f"PC{i + 1}" for i in range(n_components)])
        st.write("### PCA Components")
        st.write(pca_df)

        # Correlation matrix
        st.write("### Correlation Matrix")
        corr_matrix = df_normalized.corr()
        plt.figure(figsize=(20, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)
        st.write(corr_matrix)

        # Most important features
        st.write("### Most Important Features")
        most_important_features = pd.DataFrame(pca.components_.T,
                                               index=df_normalized.select_dtypes(include=[np.number]).columns,
                                               columns=[f"PC{i + 1}" for i in range(n_components)])
        st.write(most_important_features)

