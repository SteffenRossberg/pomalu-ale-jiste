import prometheus_client
from prometheus_client import Gauge
import simplejson as json
import os
import websocket
import ssl
import logging


def on_open(ws):
    global api_key, tickers, _logger, ticker_files
    ticker_files = {ticker: create_file(ticker) for ticker in tickers.keys()}
    prometheus_client.start_http_server(5000, '0.0.0.0')
    subscribe = {
        'eventName': 'subscribe',
        'authorization': api_key,
        'eventData': {
            'thresholdLevel': 0,
            'tickers': [ticker for ticker in tickers.keys()]
        }
    }
    ws.send(json.dumps(subscribe))
    _logger.info('Websocket opened')
    pass


def on_message(ws, message):
    global trade_price_gauges, trade_volume_gauges, bid_price_gauges, ask_price_gauges, _logger, ticker_files
    _logger.info(message)
    data = json.loads(message)
    if 'response' in data or ('data' in data and 'subscriptionId' in data['data']):
        return
    data = data['data']
    if data[0] != 'Q' and data[0] != 'T':
        return
    ticker = data[3].upper()
    last_data = ticker_data[ticker]
    last_data['type'] = record_type = data[0]
    last_data['date'] = data[1]
    last_data['timestamp'] = data[2]
    last_data['bid'] = bid = data[5]
    last_data['ask'] = ask = data[7]
    last_data['trade'] = trade = data[9]
    last_data['volume'] = volume = data[10]
    if record_type == 'T':
        trade_price_gauges[ticker].set(trade)
        trade_volume_gauges[ticker].set(volume)
    elif record_type == 'Q':
        bid_price_gauges[ticker].set(bid)
        ask_price_gauges[ticker].set(ask)
    _logger.info(f'{last_data}')
    file = ticker_files[ticker]
    file.write(f"\n{json.dumps(last_data)}")
    file.flush()
    pass


def on_error(ws, error):
    global _logger
    _logger.info(error)
    pass


def on_close(ws):
    global _logger, ticker_files
    _logger.info('Websocket closed')
    for file in ticker_files.values():
        file.close()
    ticker_files = {}
    pass


def create_file(ticker):
    file_path = f'/data/{ticker}.data'
    mode = 'wt' if not os.path.exists(file_path) else 'at'
    file = open(file_path, mode)
    return file


websocket.enableTrace(True)
_logger = logging.getLogger('websocket')

_logger.info('Initialize tickers')
tickers = {
    'AMD': 'AMD',
    'NVDA': 'NVIDIA',
    'STX': 'Seagate',
    'MU': 'Micron Technology',
    'VKTX': 'Viking Therapeutics',
    'PDSB': 'PDS Biotechnology'
}
_logger.info(f'Tickers initialized: {[ticker for ticker in tickers.keys()]}')

_logger.info('Initialize api key, ticker data and gauges')
api_key = os.getenv('TIINGO_API_KEY')
trade_price_gauges = {ticker: Gauge(f'trade_price_{ticker}', company) for ticker, company in tickers.items()}
trade_volume_gauges = {ticker: Gauge(f'trade_volume_{ticker}', company) for ticker, company in tickers.items()}
bid_price_gauges = {ticker: Gauge(f'bid_price_{ticker}', company) for ticker, company in tickers.items()}
ask_price_gauges = {ticker: Gauge(f'ask_price_{ticker}', company) for ticker, company in tickers.items()}

ticker_data = {ticker: {'bid': 0.0, 'ask': 0.0, 'trade': 0.0, 'volume': 0} for ticker, company in tickers.items()}
ticker_files = {}

_logger.info('Initialize web socket app')
wss = websocket.WebSocketApp(
    "wss://api.tiingo.com/iex",
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close)

_logger.info('Run web socket app')
wss.run_forever()
