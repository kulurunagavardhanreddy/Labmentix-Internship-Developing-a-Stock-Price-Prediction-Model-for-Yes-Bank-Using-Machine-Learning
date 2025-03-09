import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load models and scaler 
def load_model(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        st.error(f"Model file {file_path} not found. Please train and save the model first.")
        st.stop()

rf_model = load_model('C:/Users/nag15/OneDrive/Desktop/Labmentix/Yes_Bank_Stock_Price/random_forest_model.pkl')
ridge_model = load_model('C:/Users/nag15/OneDrive/Desktop/Labmentix/Yes_Bank_Stock_Price/ridge_model.pkl')
lr_model = load_model('C:/Users/nag15/OneDrive/Desktop/Labmentix/Yes_Bank_Stock_Price/linear_regression_model.pkl')
gb_model = load_model('C:/Users/nag15/OneDrive/Desktop/Labmentix/Yes_Bank_Stock_Price/gradient_boosting_model.pkl')
scaler = load_model('C:/Users/nag15/OneDrive/Desktop/Labmentix/Yes_Bank_Stock_Price/scaler.pkl')

st.set_page_config(page_title="Yes Bank Stock Prediction", layout="wide") 

# Function to load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css") # Load the CSS

# Load historical data
data_path = "C:/Users/nag15/OneDrive/Desktop/Labmentix/Yes_Bank_Stock_Price/data_YesBank_StockPrices.csv"
df = pd.read_csv(data_path)
df['Year'] = 2025
df['Date'] = pd.to_datetime(df['Date'] + '-' + df['Year'].astype(str), format="%d-%b-%Y")

st.title("Yes Bank Stock Price Prediction")
st.subheader("Sample Dataset:")
st.dataframe(df.head())

# User input
open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
date_input = st.date_input("Select Date", value=df['Date'].iloc[0])

if st.button("Predict"):
    prev_close = df['Close'].iloc[-2]
    input_data = pd.DataFrame({
        'Open': [open_price], 'High': [high_price], 'Low': [low_price],
        'Prev_Close': [prev_close], 'Price_Range': [high_price - low_price],
        'Daily_Return': [(open_price - prev_close) / prev_close if prev_close != 0 else 0],
        'Rolling_Mean': [df['Close'].rolling(window=12).mean().iloc[-1]],
        'Year': [date_input.year], 'Month': [date_input.month],
        'DayOfWeek': [date_input.weekday()], 'Quarter': [(date_input.month - 1) // 3 + 1]
    })
    input_scaled = scaler.transform(input_data)
    predictions = {
        "RF": rf_model.predict(input_scaled)[0],
        "Ridge": ridge_model.predict(input_scaled)[0],
        "LR": lr_model.predict(input_scaled)[0],
        "Gradient": gb_model.predict(input_scaled)[0]
    }
    
    formatted_date = date_input.strftime("%d-%b-%Y")
    if formatted_date in df['Date'].dt.strftime("%d-%b-%Y").values:
        actual_close = df[df['Date'].dt.strftime("%d-%b-%Y") == formatted_date]['Close'].values[0]

        st.subheader("Predicted Closing Prices:")
        cols = st.columns(len(predictions))
        for i, (model, pred) in enumerate(predictions.items()):
            cols[i].metric(model, f"{pred:.2f}")

        st.subheader("Actual Close")
        st.metric("Actual Close", f"{actual_close:.2f}")

        diffs = {model: abs(actual_close - pred) for model, pred in predictions.items()}
        sorted_diffs = sorted(diffs.items(), key=lambda x: x[1])

        color_map = {}
        if len(sorted_diffs) > 0:
            color_map[sorted_diffs[0][0]] = 'green'
        if len(sorted_diffs) > 1:
            color_map[sorted_diffs[1][0]] = 'orange'
        if len(sorted_diffs) > 2:
            color_map[sorted_diffs[2][0]] = 'red'
        for i in range(3, len(sorted_diffs)):
            color_map[sorted_diffs[i][0]] = 'black'

        st.subheader("Differences (Actual - Predicted):")
        cols_diff = st.columns(len(diffs))
        for i, (model, diff_val) in enumerate(diffs.items()):
            color = color_map[model]

            if color == 'green':
                color_code = 'green'
            elif color == 'orange':
                color_code = 'orange'
            elif color == 'red':
                color_code = 'red'
            else:
                color_code = 'black'

            cols_diff[i].markdown(f'<p style="color:{color_code}; font-size: 18px;">{model} Difference: {diff_val:.2f}</p>', unsafe_allow_html=True)

        st.markdown(
            '<div style="font-weight: bold; font-size: 18px; margin-top: 20px; border-bottom: 2px solid #ccc; padding-bottom: 5px;">'
            "Note: "
            '<span style="color:green;">Green</span> indicates the closest prediction, '
            '<span style="color:orange;">orange</span> the second closest, '
            '<span style="color:red;">red</span> the third closest, and '
            '<span style="color:black;">black</span> the very last.',
            unsafe_allow_html=True,
        )
        
        # 3D Visual for Color Ranking
        fig_color_rank = plt.figure(figsize=(6, 4))
        ax_color_rank = fig_color_rank.add_subplot(111, projection='3d')

        model_names = list(predictions.keys())
        ranks = range(len(model_names))
        colors = [color_map[model] for model in model_names]

        ax_color_rank.bar3d(ranks, [0] * len(ranks), [0] * len(ranks), [0.4] * len(ranks), [0.4] * len(ranks), [1] * len(ranks), color=colors)
        ax_color_rank.set_xticks(ranks)
        ax_color_rank.set_xticklabels(model_names)
        ax_color_rank.set_yticks([])
        ax_color_rank.set_zticks([])
        ax_color_rank.set_title("Prediction Accuracy Ranking")

        st.pyplot(fig_color_rank)

        # 3D Bar Chart with Difference and Line Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        model_names = list(predictions.keys())
        predicted_values = list(predictions.values())
        actual_values = [actual_close] * len(model_names)

        x = np.arange(len(model_names))
        y_actual = np.zeros(len(model_names))
        y_predicted = np.ones(len(model_names))

        # Color the predicted bars based on difference
        predicted_colors = [color_map[model] for model in model_names]

        ax.bar3d(x, y_actual, np.zeros(len(model_names)), 0.4, 0.4, actual_values, color='blue', alpha=0.8, label='Actual')
        for i in range(len(model_names)):
            ax.bar3d(x[i] + 0.5, y_predicted[i], 0, 0.4, 0.4, predicted_values[i], color=predicted_colors[i], alpha=0.8, label=f'Predicted ({model_names[i]})')

        # Line plot connecting predicted bar tops
        ax.plot(x + 0.7, y_predicted + 0.2, predicted_values, color='blue', linestyle='-', linewidth=2)

        for i in range(len(model_names)):
            # Display difference value on top of predicted bar
            text_color = 'blue' if predicted_colors[i] == 'red' else 'black'
            ax.text(x[i] + 0.7, 1.2, predicted_values[i], f'{diffs[model_names[i]]:.2f}', ha='center', va='bottom', color=text_color)

        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(model_names)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Actual', 'Predicted'])
        ax.set_zlabel('Close Price')
        ax.set_title('Actual vs Predicted Close Price Difference (3D)')
        ax.legend()

        st.pyplot(fig)
    
        # Second Visualization (Grouped Bar Chart)
        fig2, ax2 = plt.subplots()
        ax2.plot(['Actual'], [actual_close], 'go-', label='Actual')
        model_names = list(predictions.keys())
        predicted_values = list(predictions.values())

        # Create custom legend entries with colors
        legend_elements = [plt.Line2D([0], [0], marker='o', color=color_map[model], label=model, linestyle='None') for model in model_names]

        ax2.plot(model_names, predicted_values, 'b-', label='Predicted')

        for name, pred in zip(model_names, predicted_values):
            ax2.text(name, pred, f'{pred:.2f}', ha='center', va='bottom')

        for i, name in enumerate(model_names):
            ax2.plot(name, predicted_values[i], marker='o', color=color_map[list(predictions.keys())[i]])

        ax2.set_ylabel('Close Price')
        ax2.set_title("Actual vs Predicted")

        # Add custom legend
        ax2.legend(handles=[plt.Line2D([0], [0], marker='o', color='green', label='Actual', linestyle='-')] + legend_elements)

        st.pyplot(fig2)

    else:
        st.write(f"Prediction based on entered data for {formatted_date}-2025.")