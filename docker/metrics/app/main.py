import prometheus_client
from prometheus_client import Gauge
from websocket import create_connection
import simplejson as json
import os

if __name__ != "__main__":
    exit(0)

tickers = {
    'AMD': 'AMD',
    'NVDA': 'NVIDIA',
    'STX': 'Seagate',
    'MU': 'Micron Technology',
    'VKTX': 'Viking Therapeutics',
    'PDSB': 'PDS Biotechnology'
}

api_key = os.getenv('TIINGO_API_KEY')
trade_price_gauges = {ticker: Gauge(f'trade_price_{ticker}', company) for ticker, company in tickers.items()}
trade_volume_gauges = {ticker: Gauge(f'trade_volume_{ticker}', company) for ticker, company in tickers.items()}
bid_price_gauges = {ticker: Gauge(f'bid_price_{ticker}', company) for ticker, company in tickers.items()}
ask_price_gauges = {ticker: Gauge(f'ask_price_{ticker}', company) for ticker, company in tickers.items()}
ws = create_connection("wss://api.tiingo.com/iex")

subscribe = {
    'eventName': 'subscribe',
    'authorization': api_key,
    'eventData': {
        'thresholdLevel': 5,
        'tickers': [ticker for ticker in tickers.keys()]
    }
}

ws.send(json.dumps(subscribe))
prometheus_client.start_http_server(5000, '0.0.0.0')
data = json.loads(ws.recv())
subscriptionId = 0

while True:
    data = json.loads(ws.recv())
    if 'data' in data and 'subscriptionId' in data['data']:
        subscriptionId = data['data']['subscriptionId']
        continue
    if 'response' in data:
        continue
    data = data['data']
    ticker = data[3].upper()
    if data[0] == 'T':
        trade_price_gauges[ticker].set(data[9])
        trade_volume_gauges[ticker].set(data[10])
    elif data[0] == 'Q':
        bid_price_gauges[ticker].set(data[5])
        ask_price_gauges[ticker].set(data[7])
    print(data)

subscribe = {
    'eventName': 'unsubscribe',
    'authorization': api_key,
    'eventData': {
        'subscriptionId': subscriptionId,
        'tickers': [ticker for ticker in tickers.keys()]
    }
}

ws.send(json.dumps(subscribe))
print(ws.recv())
