from playwright.sync_api import sync_playwright
import psycopg2
from psycopg2.extras import execute_values
import datetime as dt
import tkinter as tk
from tkinter import ttk
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import requests

def crypto_prediction(name, days):
    #CoinGecko API gets crypto info for last 365 days
    crypto_currency = name
    against_currency = "usd"

    url = f"https://api.coingecko.com/api/v3/coins/{crypto_currency}/market_chart"
    params = {"vs_currency": against_currency, "days": 365}

    response = requests.get(url, params=params).json()

    prices = response['prices']
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(df.head())

    #ML
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))

    prediction_days = 60
    future_day = days

    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)-30):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x+future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Neural Network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    #Test
    test_end = dt.datetime.now()
    test_start = test_end - dt.timedelta(days=365)


    from_timestamp = int(test_start.timestamp())
    to_timestamp = int(test_end.timestamp())

    url_test = f"https://api.coingecko.com/api/v3/coins/{crypto_currency}/market_chart/range"
    params_test = {
        "vs_currency": against_currency,
        "from": from_timestamp,
        "to": to_timestamp
    }

    response_test = requests.get(url_test, params=params_test).json()
    test_prices = response_test['prices']

    test_df = pd.DataFrame(test_prices, columns=["timestamp", "price"])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], unit='ms')

    actual_prices = test_df['price'].values

    total_dataset = pd.concat((df['price'], test_df['price']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_df) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for i in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[i-prediction_days:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(prediction_prices, color='green', label='Predicted Prices')
    plt.title(f'{crypto_currency} price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.show()

    #Predict Next Day

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

def show_top_coins(data):
    root = tk.Tk()
    root.title("Top 100 Coins")

    columns = list(data[0].keys())
    tree = ttk.Treeview(root, columns=columns, show='headings')

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    for coin in data[:100]:
        tree.insert("", "end", values=list(coin.values()))

    tree.pack(expand=True, fill='both')
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    input_frame = tk.Frame(root)
    input_frame.pack(side="right", fill="y", padx=10, pady=10)

    # Crypto name entry 
    tk.Label(input_frame, text="Enter Full Crypto Name:").pack(pady=5)
    crypto_entry_widget = tk.Entry(input_frame, width=20)
    crypto_entry_widget.pack(pady=5)

    # Days entry
    tk.Label(input_frame, text="How many days do you want predicted 1-30:").pack(pady=5)
    days_entry_widget = tk.Entry(input_frame, width=20)
    days_entry_widget.pack(pady=5)

    # Submit button
    def get_input():
        crypto = crypto_entry_widget.get()
        days = days_entry_widget.get()
        try:
            days_int = int(days)
            if 1 <= days_int <= 60:
                crypto_prediction(crypto.lower().replace(' ', '-'), days_int)
            else:
                print("Days must be between 1 and 60.")
        except ValueError:
            print("Please enter a valid number for days.")

    submit_button = tk.Button(input_frame, text="Submit", command=get_input)
    submit_button.pack(pady=10)

    root.mainloop()


def main():
    with sync_playwright() as p:

        # scrape

        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto('https://coinmarketcap.com/')

        # scraping down
        for i in range(5):
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(1000)

        trs_list = page.query_selector_all('table tbody tr')

        master_list = []
        
        for tr in trs_list:

            coin_dict = {}

            # Extracting td elements inside each tr
            tds = tr.query_selector_all('td')

            coin_dict['id'] = tds[1].inner_text()
            coin_dict['Name'] = tds[2].inner_text().split('\n')[0]
            symbol_el = tds[2].query_selector('p.coin-item-symbol')
            coin_dict['Symbol'] = symbol_el.inner_text() if symbol_el else 'N/A'
            coin_dict['Price'] = float(tds[3].inner_text().replace('$', '').replace(',', ''))
            coin_dict['Market_cap_usd'] = int(tds[7].inner_text().replace('$', '').replace(',', ''))
            coin_dict['Volume_24h_usd'] = int(tds[8].query_selector('div > a > p').inner_text().replace('$', '').replace(',', ''))
            coin_dict['scrape_date'] = dt.datetime.now()

            master_list.append(coin_dict)

        list_of_tuples = [tuple(dic.values()) for dic in master_list]

        

        # save

        # connect to database
        pgconn = psycopg2.connect(
            host = 'localhost', 
            database = 'postgres',
            user = '', #postgreSQL user
            password = ''
        )

        # create cursor
        pgcursor = pgconn.cursor()

        execute_values(pgcursor,
            """
            INSERT INTO crypto (id, name, symbol, price_usd, market_cap_usd, volume_24h_usd, scrape_date)
            VALUES %s
            ON CONFLICT (id) 
            DO UPDATE SET
                price_usd = EXCLUDED.price_usd,
                market_cap_usd = EXCLUDED.market_cap_usd,
                volume_24h_usd = EXCLUDED.volume_24h_usd,
                scrape_date = EXCLUDED.scrape_date
            """,
    list_of_tuples)

        #commit
        pgconn.commit()

        pgconn.close()

        browser.close()
        
        show_top_coins(master_list)



if __name__ == '__main__':
    main()


