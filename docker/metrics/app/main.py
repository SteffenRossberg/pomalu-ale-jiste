import simplejson as json
from datetime import datetime
import os
import websocket
import logging


def on_open(ws):
    global api_key, tickers, _logger, ticker_files
    if _last_date.day != datetime.now().day:
        recreate_files()
    subscribe = {
        'eventName': 'subscribe',
        'authorization': api_key,
        'eventData': {
            'thresholdLevel': 0,
            'tickers': tickers
        }
    }
    ws.send(json.dumps(subscribe))
    _logger.info('Websocket opened')
    pass


def on_message(ws, message):
    global _logger, ticker_files, _counter, _last_date
    _logger.info(message)
    data = json.loads(message)
    if _last_date.day != datetime.now().day:
        recreate_files()
    if 'response' in data or ('data' in data and 'subscriptionId' in data['data']):
        return
    data = data['data']
    if data[0] != 'Q' and data[0] != 'T':
        return
    ticker = data[3].upper()
    record_type = data[0]
    date = data[1]
    timestamp = data[2]
    bid_price = data[5] if data[5] is not None else ''
    bid_vol = data[4] if data[4] is not None else ''
    ask_price = data[7] if data[7] is not None else ''
    ask_vol = data[8] if data[8] is not None else ''
    last_price = data[9] if data[9] is not None else ''
    last_vol = data[10] if data[10] is not None else ''
    line = f'{date};{timestamp};{record_type};{bid_price};{bid_vol};{ask_price};{ask_vol};{last_price};{last_vol}'
    file = ticker_files[ticker]
    file.write(f'{line}\n')
    _counter += 1
    if _counter >= 100:
        file.flush()
        _counter = 0
    pass


def on_error(ws, error):
    global _logger
    _logger.info(error)
    pass


def on_close(ws):
    global _logger, ticker_files
    _logger.info('Websocket closed')
    for file in ticker_files.values():
        file.flush()
        file.close()
    ticker_files = {}
    pass


def recreate_files():
    global ticker_files, _last_date
    date = datetime.now()
    ticker_files = {ticker: recreate_file(ticker, date) for ticker in tickers}
    _last_date = date


def recreate_file(ticker, last_date):
    if ticker in ticker_files:
        file = ticker_files[ticker]
        file.flush()
        file.close()
    directory = f'/data/{last_date:%Y%m%d}'
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    file_path = f'{directory}/{ticker}.csv'
    mode = 'wt' if not os.path.exists(file_path) else 'at'
    file = open(file_path, mode)
    if mode == 'wt':
        file.write('date;timestamp;type;bid_price;bid_volume;ask_price;ask_volume;last_price;last_volume\n')
        file.flush()
    return file


websocket.enableTrace(True)
_logger = logging.getLogger('websocket')
_counter = 0
_last_date = datetime.min
_logger.info('Initialize tickers')
tickers = [
    'MMM', 'ABBV', 'ABMD', 'ATVI', 'ADBE', 'AMC', 'AMD',
    'A', 'AGYS', 'ALB', 'GOOGL', 'AMZN', 'AAL', 'AMGN',
    'AXP', 'AAPL', 'T', 'BIDU', 'BAC', 'GOLD', 'BIIB',
    'BB', 'BA', 'BKNG', 'BXP', 'BMY', 'CAT', 'CVX',
    'CHD', 'CSCO', 'C', 'KO', 'CL', 'DE', 'EBAY',
    'EIX', 'EA', 'EL', 'EXPE', 'XOM', 'FB', 'FDX',
    'FSLR', 'FL', 'FCX', 'GE', 'GME', 'GS', 'GRPN',
    'HD', 'HON', 'IBM', 'ILMN', 'INTC', 'JPM', 'JNJ',
    'KEY', 'KMI', 'KLAC', 'KHC', 'LMT', 'MA', 'MCD',
    'MRK', 'MSFT', 'MS', 'NFLX', 'NKE', 'NVDA', 'ORCL',
    'PGRE', 'PYPL', 'PEP', 'PFE', 'PM', 'PXD', 'PG',
    'QCOM', 'RTN', 'REGN', 'RMD', 'CRM', 'STX', 'SLB',
    'SNE', 'SBUX', 'TGT', 'TSLA', 'TEVA', 'TRV', 'TRIP',
    'TWTR', 'UAA', 'UNH', 'VRSN', 'VZ', 'V', 'WBA',
    'WMT', 'DIS', 'WFC', 'YELP', 'ZNGA', 'VKTX', 'MU',
    'PDSB'
]
_logger.info(f'Tickers initialized: {", ".join(tickers)}')

_logger.info('Initialize api key, ticker data and gauges')
api_key = os.getenv('TIINGO_API_KEY')

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
