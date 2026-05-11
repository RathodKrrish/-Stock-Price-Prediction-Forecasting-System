
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import joblib # Added joblib for the scaler

from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
from datetime import date, timedelta


print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)

# ------------------------------
# CONFIG
# ------------------------------
start = "2010-01-01"
today = date.today()
yesterday = today - timedelta(days=1)
end = yesterday.strftime('%Y-%m-%d')

st.title("Stock Trend Prediction")

# ------------------------------
# USER INPUT
# ------------------------------
user_input = st.text_input("Enter Stock Ticker:", "AAPL")

# ------------------------------
# DATA LOAD
# ------------------------------
data = yf.download(user_input, start=start, end=end)

if data.empty:
    st.error("Invalid ticker or no data found.")
    st.stop()

df = pd.DataFrame(data)

st.subheader("Data Summary")
st.write(df.describe())

# START FOR OPEN

st.subheader("Opening Price vs Time")
fig1 = plt.figure(figsize=(12,6))
plt.plot(df['Open'])
plt.xlabel("Time")
plt.ylabel("Price")
st.pyplot(fig1)

# Moving averages
st.subheader("Moving Averages (100 & 200)")
ma100 = df['Open'].rolling(100).mean()
ma200 = df['Open'].rolling(200).mean()

fig2 = plt.figure(figsize=(12,6))
plt.plot(df['Open'], label="Open")
plt.plot(ma100, label="MA100")
plt.plot(ma200, label="MA200")
plt.legend()
st.pyplot(fig2)

# ------------------------------
# DATA SPLIT & SCALING (FIXED)
# ------------------------------
data_training = df[['Open']][:int(len(df)*0.70)]
data_testing = df[['Open']][int(len(df)*0.70):]

# 1. Load the pre-trained scaler from your Jupyter Notebook
scaler = joblib.load('scaler.pkl')

# 2. Transform the data (Notice we DO NOT use fit() or recreate the MinMaxScaler)
data_training_scaled = scaler.transform(data_training)

# ------------------------------
# TRAIN DATA PREP
# ------------------------------
X_train = []
y_train = []

for i in range(100, len(data_training_scaled)):
    X_train.append(data_training_scaled[i-100:i])
    y_train.append(data_training_scaled[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# ------------------------------
# LOAD MODEL
# ------------------------------
from tensorflow.keras.layers import Dense

def custom_dense(**kwargs):
    kwargs.pop("quantization_config", None)
    return Dense(**kwargs)

model = keras.models.load_model(
    "Opnkeras_model.h5",
    custom_objects={"Dense": custom_dense},
    compile=False
)
# ------------------------------
# TEST DATA PREP
# ------------------------------
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Transform the test data using the SAME loaded scaler
input_data = scaler.transform(final_df)

X_test = []
y_test = []

for i in range(100, len(input_data)):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0]) # Ensure we just grab the scalar value

X_test = np.array(X_test)
y_test = np.array(y_test)

# ------------------------------
# PREDICTION
# ------------------------------
y_predicted = model.predict(X_test)

# Inverse scaling (Reshaping ensures no dimensional errors)
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)) 

# accuracy metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, y_predicted)

rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

mae = mean_absolute_error(y_test, y_predicted)


st.subheader("Prediction vs Original For Open")

# Extract the actual dates that correspond to our test data
test_dates = data_testing.index

fig3 = plt.figure(figsize=(12,6))
# Add 'test_dates' as the first argument to plot them on the X-axis
plt.plot(test_dates, y_test, 'g', label="Original Price") 
plt.plot(test_dates, y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time (Years)")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig3)




# ------------------------------
# VISUALIZATION FOR CLOSE
# ------------------------------
st.subheader("Closing Price vs Time")
fig1 = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel("Time")
plt.ylabel("Price")
st.pyplot(fig1)

# Moving averages
st.subheader("Moving Averages (100 & 200)")
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

fig2 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label="Close")
plt.plot(ma100, label="MA100")
plt.plot(ma200, label="MA200")
plt.legend()
st.pyplot(fig2)

# ------------------------------
# DATA SPLIT & SCALING (FIXED)
# ------------------------------
data_training = df[['Close']][:int(len(df)*0.70)]
data_testing = df[['Close']][int(len(df)*0.70):]

# 1. Load the pre-trained scaler from your Jupyter Notebook
scaler = joblib.load('scaler.pkl')

# 2. Transform the data (Notice we DO NOT use fit() or recreate the MinMaxScaler)
data_training_scaled = scaler.transform(data_training)

# ------------------------------
# TRAIN DATA PREP
# ------------------------------
X_train = []
y_train = []

for i in range(100, len(data_training_scaled)):
    X_train.append(data_training_scaled[i-100:i])
    y_train.append(data_training_scaled[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# ------------------------------
# LOAD MODEL
# ------------------------------
from tensorflow.keras.layers import Dense

def custom_dense(**kwargs):
    kwargs.pop("quantization_config", None)
    return Dense(**kwargs)

model = keras.models.load_model(
    "keras_modell.h5",
    custom_objects={"Dense": custom_dense},
    compile=False
)

# ------------------------------
# TEST DATA PREP
# ------------------------------
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Transform the test data using the SAME loaded scaler
input_data = scaler.transform(final_df)

X_test = []
y_test = []

for i in range(100, len(input_data)):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0]) # Ensure we just grab the scalar value

X_test = np.array(X_test)
y_test = np.array(y_test)

# ------------------------------
# PREDICTION
# ------------------------------
y_predicted = model.predict(X_test)

# Inverse scaling (Reshaping ensures no dimensional errors)
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)) 

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, y_predicted)

rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

mae = mean_absolute_error(y_test, y_predicted)


st.subheader("Prediction vs Original For Closing Price")

# Extract the actual dates that correspond to our test data
test_dates = data_testing.index

fig3 = plt.figure(figsize=(12,6))
# Add 'test_dates' as the first argument to plot them on the X-axis
plt.plot(test_dates, y_test, 'g', label="Original Price") 
plt.plot(test_dates, y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time (Years)")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig3)




# =========================================================
# FUTURE 30 DAYS FORECASTING
# =========================================================

st.subheader("Next 30 Days Predicted Price Forecast")

# =========================================================
# LAST 100 DAYS HISTORICAL DATA
# =========================================================

historical_data = df.tail(100)

# =========================================================
# CLOSE PRICE FORECAST
# =========================================================

# Reload CLOSE model separately
close_model = keras.models.load_model(
    "keras_modell.h5",
    custom_objects={"Dense": custom_dense},
    compile=False
)

close_data = df[['Close']]

# Scale close data
scaled_close = scaler.transform(close_data)

# Take last 100 days
last_100_close = scaled_close[-100:]

future_close = []

x_input = last_100_close.reshape(1,100,1)

# Predict next 30 business days
for i in range(30):

    pred = close_model.predict(
        x_input,
        verbose=0
    )

    future_close.append(pred[0][0])

    # Update input sequence
    x_input = x_input.reshape(100,1)

    x_input = np.vstack((x_input[1:], pred))

    x_input = x_input.reshape(1,100,1)

# Convert back to original prices
future_close = scaler.inverse_transform(
    np.array(future_close).reshape(-1,1)
)

# =========================================================
# OPEN PRICE FORECAST
# =========================================================

# Reload OPEN model separately
open_model = keras.models.load_model(
    "Opnkeras_model.h5",
    custom_objects={"Dense": custom_dense},
    compile=False
)

open_data = df[['Open']]

# Scale open data
scaled_open = scaler.transform(open_data)

# Take last 100 days
last_100_open = scaled_open[-100:]

future_open = []

x_input = last_100_open.reshape(1,100,1)

# Predict next 30 business days
for i in range(30):

    pred = open_model.predict(
        x_input,
        verbose=0
    )

    future_open.append(pred[0][0])

    # Update input sequence
    x_input = x_input.reshape(100,1)

    x_input = np.vstack((x_input[1:], pred))

    x_input = x_input.reshape(1,100,1)

# Convert back to original prices
future_open = scaler.inverse_transform(
    np.array(future_open).reshape(-1,1)
)

# =========================================================
# FUTURE BUSINESS DATES ONLY
# =========================================================

future_dates = pd.bdate_range(
    start=df.index[-1] + timedelta(days=1),
    periods=30
)

# =========================================================
# FINAL FORECAST GRAPH
# =========================================================

fig4 = plt.figure(figsize=(18,8))

# ------------------------------
# HISTORICAL CLOSE
# ------------------------------

plt.plot(
    historical_data.index,
    historical_data['Close'],
    label='Historical Close',
    linewidth=2
)

# ------------------------------
# HISTORICAL OPEN
# ------------------------------

plt.plot(
    historical_data.index,
    historical_data['Open'],
    label='Historical Open',
    linewidth=2
)

# ------------------------------
# PREDICTED CLOSE
# ------------------------------

plt.plot(
    future_dates,
    future_close,
    label='Predicted Close',
    linestyle='dashed',
    linewidth=2
)

# ------------------------------
# PREDICTED OPEN
# ------------------------------

plt.plot(
    future_dates,
    future_open,
    label='Predicted Open',
    linestyle='dashed',
    linewidth=2
)

# =========================================================
# CURRENT DATE DIVIDER
# =========================================================

plt.axvline(
    x=df.index[-1],
    linestyle='--',
    linewidth=2
)

# =========================================================
# LABELS & STYLE
# =========================================================

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("AI Based 30 Business Days Forecast")
plt.legend()
plt.grid(True)
st.pyplot(fig4)

st.subheader("Model Evaluation Metrics")

st.write(f"R² Score: {r2:.4f}")

st.write(f"RMSE: {rmse:.4f}")

st.write(f"MAE: {mae:.4f}")