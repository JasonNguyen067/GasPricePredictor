from flask import Flask, render_template
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Disable GUI mode before importing pyplot

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


app = Flask(__name__)

@app.route('/')
def index():
    # Load and clean the CSV
    df = pd.read_csv("gas_prices.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Price']).sort_values('Date')


    # Convert dates to numbers
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    X = df[['Date_Ordinal']]
    y = df['Price']

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    df['Predicted'] = model.predict(X)

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Price'], label='Actual Prices')
    plt.plot(df['Date'], df['Predicted'], label='Predicted Trend', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title('Gas Price Trend')
    plt.legend()
    plt.tight_layout()

    # Save chart to static folder
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/trend.png")
    plt.close()

    return render_template("index.html")
