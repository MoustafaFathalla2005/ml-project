from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
#  load data
# data EDA
# visualization (Before preprocess)
# preprocessing
# visualization (After preprocess)
# model training
# model prediction
# model evaluation

st.set_page_config(page_title="Heart Disease ML Project", page_icon='❤️‍🩹', layout="wide")
def eda_data(df, cat=True):  
    ""
    "## Data"
    df
    "## Data Info"
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.code(s)

    "## Data Description"
    "*Numeric Description*"
    df.describe().T

    if not cat: return
    "*Categorical Description*"
    df.describe(include='object').T
def visualize_data(data, numeric_cols, categorical_cols):
    # box
    # heatmap
    # countplot
    # barplot
    # scatterplot
    # pairplot
    box_tab, heatmap_tab, countplot_tab, scatterplot_tab, lineplot_tab = st.tabs(
        ['Box plot', 'Heatmap', 'Count plot', 'Scatter plot', 'Line plot']) 
    with box_tab:
        for col in numeric_cols:
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(data=data, x=col)
            st.pyplot(fig)
    with heatmap_tab:
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot(fig)
    with countplot_tab:
        for col in categorical_cols:
            fig = plt.figure(figsize=(10, 6))
            sns.countplot(data=data, x=col)
            st.pyplot(fig)
    with scatterplot_tab:
        for col in numeric_cols:
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=data.index, y=col)
            st.pyplot(fig)
    with lineplot_tab:
        for col in numeric_cols:
            fig = plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x=data.index, y=col)
            st.pyplot(fig)

with st.sidebar:
    "# Heart Disease ML Project"
    page = st.radio("Select a page:", ["Home", "Data EDA (Before Preprocessing)", "Visualization (Before preprocess)", "Preprocessing", "Data EDA (After preprocess)", "Visualization (After preprocess)", "Model Training", "Model Evaluation"])

df = None
df_preprocessed = None
numeric_cols = None
if 'df_preprocessed' in st.session_state:
    df_preprocessed = st.session_state.df_preprocessed
if 'df' in st.session_state:
    df = st.session_state.df
    numeric_cols = st.session_state.numeric_cols
if page == "Home":
    "# Welcome!"
    file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", 'xls'])
    if file is None:
        st.stop()

    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file)

    st.session_state.df = df
    numeric_cols = df.select_dtypes(include='number').drop(columns=['HeartDisease', 'FastingBS']).columns
    st.session_state.numeric_cols = numeric_cols
    st.success("Data loaded successfully!")
elif page == "Data EDA (Before Preprocessing)":
    eda_data(df)
elif page == "Visualization (Before preprocess)":
    "# Visualization"
    visualize_data(df, numeric_cols, df.select_dtypes(include=['object']).columns)
elif page == "Preprocessing":
    df_preprocessed = df.copy()

    "# Preprocessing"
    missingval_tab, duplicated_tab, outliers_tab, normalization_tab, encoding_tab, pca_tab = st.tabs(
        ['Missing Values', 'Duplicated Rows', 'Outliers', 'Normalization', 'Encoding', 'PCA'])
    with missingval_tab:
        "### Number of nan in each column"
        number_of_nan = df.isnull().sum()
        flipped = pd.DataFrame(number_of_nan).T
        flipped
        "### Handling missing values using KNN Imputer"
        imputer = KNNImputer(n_neighbors=2)
        df_preprocessed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        st.success(f"Succesfully imputed **{number_of_nan.sum()}** values using **KNN Imputer**")
        df_preprocessed

    with duplicated_tab:
        "## Handling duplicated rows"
        duplicate_count = df_preprocessed.duplicated().sum()
        df_preprocessed = df_preprocessed.drop_duplicates()
        st.success(f"Succesfully dropped **{duplicate_count}** duplicate rows")
        df_preprocessed

    with outliers_tab:
        "## Handling outliers"
        Q1 = df_preprocessed[numeric_cols].quantile(0.25)
        Q3 = df_preprocessed[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        df_preprocessed = df_preprocessed[~((df_preprocessed[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_preprocessed[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)] 
        st.success(f"Succesfully removed **{len(df.index) - len(df_preprocessed.index)}** outliers using IQR method.")
        df_preprocessed

    with normalization_tab:
        "## Data normalization"
        st.success('Succesfully normalized numeric columns using **Standard Scaler**')
        df_preprocessed = df_preprocessed.copy()
        scaler = StandardScaler()

        numeric_features = list(numeric_cols)
        df_preprocessed[numeric_features] = scaler.fit_transform(df_preprocessed[numeric_features])
        df_preprocessed

    with encoding_tab:
        # encoder 
        "## Encoding categorical data"
        st.success("Succesfully encoded data using **Label Encoder**")
        cat_cols = df_preprocessed.select_dtypes(include=["object"]).columns
        encoder = LabelEncoder()
        for col in cat_cols:
            df_preprocessed[col] = encoder.fit_transform(df_preprocessed[col])

        df_preprocessed

    with pca_tab:
        "## Performing PCA"
        x = df_preprocessed.drop('HeartDisease' , axis = 1)
        y = df_preprocessed['HeartDisease']

        pca = PCA(n_components = 2)
        x = pca.fit_transform(x)
        df_preprocessed=pd.DataFrame(x, columns = ['PC1' , 'PC2'])
        df_preprocessed['HeartDisease'] = y.values
        st.success("Succesfully performed PCA")
        df_preprocessed

    st.session_state.df_preprocessed = df_preprocessed
elif page == "Data EDA (After preprocess)":
    eda_data(df_preprocessed, cat=False)
elif page == "Visualization (After preprocess)":
    "# Visualization"
    visualize_data(df_preprocessed, ['PC1', 'PC2'], ['HeartDisease'])
elif page == "Model Training":
    "# Model Training"
    model_type= st.selectbox('Select model', [ 'Neural Network', 'Clustering'], index=None)
    st.session_state.model_type = model_type
    if model_type == 'Neural Network':
        "## Neural Network"
        # split data into train and test sets
        X = df_preprocessed.drop('HeartDisease', axis=1)
        y = df_preprocessed['HeartDisease']
        train_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test
        st.session_state.train_test = train_test
        # X.shape, y.shape
        # st.write(f"X shape: {X.shape}")
        # st.write(f"y shape: {y.shape}")

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        
        with st.spinner("Training the model..."):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=1)
            st.dataframe(history.history)

        st.session_state.model = model
        st.success("Model trained successfully!")

    elif model_type == 'Clustering':
        "## Clustering"
        with st.spinner("Training the model..."):
            kmeans=KMeans(n_clusters=2,random_state=0, n_init='auto').fit(df_preprocessed.drop('HeartDisease', axis=1))
            df_labeled = df_preprocessed.copy() 
            df_labeled["labels"] = kmeans.labels_
            sns.scatterplot(df_labeled,x="PC1", y="PC2", hue="labels")
            st.pyplot(plt)
            st.session_state.kmeans = kmeans
        st.success("Model trained successfully!")

elif page == "Model Evaluation":
    "# Model evaluation"
    if st.session_state.model_type == 'Neural Network':
        model = st.session_state.model
        X_train, X_test, y_train, y_test = st.session_state.train_test
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df
        st.success("Neural Network Model evaluation completed successfully!")

    elif st.session_state.model_type == 'Clustering':
        kmeans = st.session_state.kmeans
        X_test = df_preprocessed.drop('HeartDisease', axis=1)
        y_pred = kmeans.predict(X_test)
        silhouette_avg = silhouette_score(X_test, y_pred)
        st.write(f"Silhouette Score: {silhouette_avg}")
        st.success("Kmeans model evaluation completed successfully!")
