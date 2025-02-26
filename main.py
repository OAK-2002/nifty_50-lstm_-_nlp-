from datetime import datetime
import requests
import pandas as pd
from transformers import pipeline
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense
import gradio as gr


def create_lstm_data(data, time_steps=1):
  x, y = [], []
  for i in range(len(data) - time_steps):
    x.append(data[i:(i + time_steps), 0])
    y.append(data[i + time_steps, 0])
  return np.array(x), np.array(y)


# Using a pretrained model for headline sentiment analysis
def predict(stock, date, weeks):
  pipe = pipeline("text-classification", model="ProsusAI/finbert")

  # Using the News Api to fetch headlines about a certain stock
  url = "https://newsapi.org/v2/everything"
  params = {
    "q": stock,
    "from": "2024-04-07",
    "sortBy": "popularity",
    "apiKey": "eed35c50fff840f2bf19d01fef61ed27"
  }
  response = requests.get(url, params=params)
  df = pd.DataFrame(response.json()["articles"])
  headlines = list()

  i = 0
  for name, item in df["title"].items():
    headlines.insert(i, str(item))
    i += 1
  sentiments = pipe(headlines)

  # Fetching stock data from yahoo finance api
  stock_symbol = "^NSEI"
  start_date = "2020-01-01"
  data = yf.download(stock_symbol, start=start_date, end=None)

  # Preprocessing
  close_prices = data["Close"].values.reshape(-1, 1)
  scaler = MinMaxScaler(feature_range=(0, 1))
  close_prices_scaled = scaler.fit_transform(close_prices)

  # Preparing the training data
  time_steps = 10
  x, y = create_lstm_data(close_prices_scaled, time_steps)
  x = np.reshape(x, (x.shape[0], x.shape[1], 1))

  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
  model.add(LSTM(units=50))
  model.add(Dense(units=1))
  model.compile(optimizer="adam", loss="mean_squared_error")

  model.fit(x, y, epochs=50, batch_size=32)

  last_prices = close_prices[-time_steps:]
  last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1))
  x_pred = np.array([last_prices_scaled[-time_steps:, 0]])
  x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
  predicted_prices_scaled = model.predict(x_pred)
  predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
  info = ""
  j = 0
  for headline in headlines:
    info = info + headline + "\nlabel: " + sentiments[j]["label"] + " confidence: " + str(sentiments[j]["score"]) + "\n"
    j+=1
  current_stock_price = data["Close"][-1]
  change = ((predicted_prices - current_stock_price) / current_stock_price) * 100
  max_change = 1
  for sentiment in sentiments:
    label = sentiment["label"]
    if label == "positive":
      change += 0.025
    elif label == "negative":
      change -= 0.025
    elif label == "neutral":
      pass
    else:
      print("Invalid")
  response = "The " + stock + " stock is expected changes by " + str(change) + "%"
  return info, predicted_prices, current_stock_price, response

app = gr.Interface(
  predict,
  inputs=[
    gr.Text(
      label="Stock",
      info="Companies from the National Stock Market"
    ),
    gr.Text(
      label="Date",
      value=datetime.today().strftime('%Y-%m-%d'),
      info="Date from which the prediction is made, use format yyyy-mm-dd"
    ),
    gr.Slider(
      minimum=1,
      maximum=4,
      value=3,
      step=1,
      label="n_weeks",
      info="Information of the past n weeks will be utilized, choose between 1 and 4"
    )
  ],
  outputs=[
    gr.Textbox(
      label="Information"
    ),
    gr.Textbox(
      label="Predicted Price"
    ),
    gr.Textbox(
      label="Current Price"
    ),
    gr.Textbox(
      label="Response"
    )
  ],
  examples = [
    ["NIFTY", datetime.today().strftime('%Y-%m-%d'), 3],
    ["TCS", datetime.today().strftime('%Y-%m-%d'), 3],
    ["ADANIPORTS", datetime.today().strftime('%Y-%m-%d'), 3],
    ["ASIANPAINT", datetime.today().strftime('%Y-%m-%d'), 3],
    ["AXISBANK", datetime.today().strftime('%Y-%m-%d'), 3],
    ["BAJAJ-AUTO", datetime.today().strftime('%Y-%m-%d'), 3],
    ["BAJFINANCE", datetime.today().strftime('%Y-%m-%d'), 3],
    ["BAJAJFINSV", datetime.today().strftime('%Y-%m-%d'), 3],
    ["BPCL", datetime.today().strftime('%Y-%m-%d'), 3],
    ["BHARTIARTL", datetime.today().strftime('%Y-%m-%d'), 3],
    ["BRITANNIA", datetime.today().strftime('%Y-%m-%d'), 3],
    ["CIPLA", datetime.today().strftime('%Y-%m-%d'), 3],
    ["COALINDIA", datetime.today().strftime('%Y-%m-%d'), 3],
    ["DIVISLAB", datetime.today().strftime('%Y-%m-%d'), 3],
    ["DRREDDY", datetime.today().strftime('%Y-%m-%d'), 3],
    ["EICHERMOT", datetime.today().strftime('%Y-%m-%d'), 3],
    ["GRASIM", datetime.today().strftime('%Y-%m-%d'), 3],
    ["HCLTECH", datetime.today().strftime('%Y-%m-%d'), 3],
    ["HDFCBANK", datetime.today().strftime('%Y-%m-%d'), 3],
    ["HDFCLIFE", datetime.today().strftime('%Y-%m-%d'), 3],
    ["HEROMOTOCO", datetime.today().strftime('%Y-%m-%d'), 3],
    ["HINDALCO", datetime.today().strftime('%Y-%m-%d'), 3],
    ["HINDUNILVR", datetime.today().strftime('%Y-%m-%d'), 3],
    ["ICICIBANK", datetime.today().strftime('%Y-%m-%d'), 3],
    ["IOC", datetime.today().strftime('%Y-%m-%d'), 3],
    ["INDUSINDBK", datetime.today().strftime('%Y-%m-%d'), 3],
    ["INFY", datetime.today().strftime('%Y-%m-%d'), 3],
    ["ITC", datetime.today().strftime('%Y-%m-%d'), 3],
    ["JSWSTEEL", datetime.today().strftime('%Y-%m-%d'), 3],
    ["KOTAKBANK", datetime.today().strftime('%Y-%m-%d'), 3],
    ["LT", datetime.today().strftime('%Y-%m-%d'), 3],
    ["M&M", datetime.today().strftime('%Y-%m-%d'), 3],
    ["MARICO", datetime.today().strftime('%Y-%m-%d'), 3],
    ["MARUTI", datetime.today().strftime('%Y-%m-%d'), 3],
    ["NTPC", datetime.today().strftime('%Y-%m-%d'), 3],
    ["NESTLEIND", datetime.today().strftime('%Y-%m-%d'), 3],
    ["ONGC", datetime.today().strftime('%Y-%m-%d'), 3],
    ["POWERGRID", datetime.today().strftime('%Y-%m-%d'), 3],
    ["RELIANCE", datetime.today().strftime('%Y-%m-%d'), 3],
    ["SHREECEM", datetime.today().strftime('%Y-%m-%d'), 3],
    ["SBIN", datetime.today().strftime('%Y-%m-%d'), 3],
    ["SBILIFE", datetime.today().strftime('%Y-%m-%d'), 3],
    ["SUNPHARMA", datetime.today().strftime('%Y-%m-%d'), 3],
    ["TATACONSUM", datetime.today().strftime('%Y-%m-%d'), 3],
    ["TATAMOTORS", datetime.today().strftime('%Y-%m-%d'), 3],
    ["TECHM", datetime.today().strftime('%Y-%m-%d'), 3],
    ["TITAN", datetime.today().strftime('%Y-%m-%d'), 3],
    ["ULTRACEMCO", datetime.today().strftime('%Y-%m-%d'), 3],
    ["UPL", datetime.today().strftime('%Y-%m-%d'), 3],
    ["WIPRO", datetime.today().strftime('%Y-%m-%d'), 3],
  ],
  title="OAK NSE Forecaster"
)

app.launch()
