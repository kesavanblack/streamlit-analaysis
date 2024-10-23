import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())
    
    # Get columns list
    columns = list(data.columns)
    
    # Select target column for color encoding
    target = st.selectbox('Choose a target:', columns)
    
    # Determine if the target is categorical or continuous
    is_classification = False
    if data[target].dtype == 'object' or len(data[target].unique()) < 20:
        is_classification = True  # Assume it's classification
    
    # Remove target column from available X options
    col2 = columns.copy()
    col2.remove(target)
    
    # Select variables for X and Y axes
    x_var = st.selectbox('Choose an X variable:', col2)
    y_var = st.selectbox('Choose a Y variable:', col2)
    
    # Select plot type
    plot_type = st.selectbox('Choose plot type:', ['Scatter', 'Line', 'Bar', 'Histogram', 'Pie', 'Heatmap', 'Box'])
    
    # Generate plot based on selected plot type
    if plot_type == 'Scatter':
        fig = px.scatter(data, x=x_var, y=y_var, color=target)
    elif plot_type == 'Line':
        fig = px.line(data, x=x_var, y=y_var, color=target)
    elif plot_type == 'Bar':
        fig = px.bar(data, x=x_var, y=y_var, color=target)
    elif plot_type == 'Histogram':
        fig = px.histogram(data, x=x_var, y=y_var, color=target)
    elif plot_type == 'Pie':
        pie_data = data[target].value_counts().reset_index()
        pie_data.columns = [target, 'count']  # Rename columns for clarity
        fig = px.pie(pie_data, values='count', names=target, title='Pie Chart of ' + target)
    elif plot_type == 'Heatmap':
        # Only use numeric columns for correlation
        numeric_data = data.select_dtypes(include=[np.number])
        heatmap_data = numeric_data.corr()  # Calculate correlation matrix
        fig = px.imshow(heatmap_data, title='Heatmap of Correlation Matrix')
    elif plot_type == 'Box':
        fig = px.box(data, x=x_var, y=y_var, color=target, title='Box Plot')
    
    # Display the plot
    st.plotly_chart(fig)

    # Machine Learning Model Deployment
    st.header("Machine Learning Model Deployment")

    # Preprocess data: encode categorical variables
    data_encoded = data.copy()
    label_encoders = {}
    
    for col in data_encoded.columns:
        if data_encoded[col].dtype == 'object':
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col])
            label_encoders[col] = le  # Save the encoder for future use

    # Split the data
    X = data_encoded[col2]
    y = data_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model based on classification or regression
    if is_classification:
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"### Model Accuracy: {accuracy:.2f}")
    else:
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        st.write(f"### Model R² Score: {r2:.2f}")

    # User Input for Prediction
    st.write("### Input Features for Prediction")
    input_data = {}
    for col in col2:
        if col in label_encoders:  # Check if we need to encode this input
            input_data[col] = st.selectbox(f"Select {col}:", label_encoders[col].classes_)
        else:
            input_data[col] = st.number_input(f"Enter {col}")

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        # Encode input for prediction
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        if is_classification:
            prediction = model.predict(input_df)
            st.write(f'### Predicted Class: {label_encoders[target].inverse_transform(prediction)[0]}')
        else:
            prediction = model.predict(input_df)
            st.write(f'### Predicted Value: {prediction[0]:.2f}')
