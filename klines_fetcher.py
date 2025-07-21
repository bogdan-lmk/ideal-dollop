import requests
import pandas as pd
from datetime import datetime, timedelta


def get_historical_klines(symbol, interval, limit=100, start_time=None, end_time=None):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&startTime={start_time}&endTime={end_time}'
    try:
        response = requests.get(url)
        data = response.json()
        if not data:
            print(f"Нет данных для символа {symbol} с параметрами {start_time}.")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        #print(f"Получено {len(df)} свечей для {symbol}.")
        return df
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP ошибка: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Ошибка соединения: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Ошибка таймаута: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка запроса: {req_err}")
    except ValueError as val_err:
        print(f"Ошибка преобразования данных: {val_err}")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")

    return pd.DataFrame()  # Возвращает пустой DataFrame в случае ошибки

def fetch_historical_data(symbol, timeframe, start_time, end_time, limit=1000):
    frames = []
    while start_time < end_time:
        df = get_historical_klines(symbol, timeframe, start_time=start_time, end_time=end_time, limit=limit)
        if df.empty:
            break
        frames.append(df)
        # Получаем время закрытия последней свечи и увеличиваем начало следующего запроса
        last_close_time = df['close_time'].iloc[-1]
        next_start_time = int((last_close_time + pd.Timedelta(milliseconds=1)).timestamp() * 1000)

        # Если новых данных нет, выходим из цикла
        if next_start_time == start_time:
            break

        start_time = next_start_time
    return pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()