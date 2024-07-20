import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import yfinance as yf

# Load your dataset
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    return df

# Add technical indicators
def add_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df = df.dropna()
    return df

# Main function to run the app
def main():
    st.title('Stock Price Prediction with Technical Indicators')

    # Sidebar for ticker input
    ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')

    df = load_data(ticker)
    st.write(df.head())

    df = add_indicators(df)
    
    # Sidebar for selecting the model
    model_name = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Random Forest', 'Support Vector Machine'])

    # Splitting data into features and target
    features = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50']
    X = df[features]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model parameters
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Random Forest':
        n_estimators = st.sidebar.slider('Number of trees', 10, 100, 10)
        max_depth = st.sidebar.slider('Max depth of tree', 1, 20, 5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == 'Support Vector Machine':
        C = st.sidebar.slider('Regularization parameter', 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        model = SVR(C=C, kernel=kernel)

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display results
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse}')

    # Optionally, you can show actual vs predicted prices
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write(results_df.head())

if __name__ == '__main__':
    main()
