import requests
import pandas as pd
import os
from datetime import datetime, timedelta


class DataProvider:

    def __init__(self, api_key, data_directory='data'):
        self.__base_url = 'https://api.tiingo.com'
        self.__eod_url = f'{self.__base_url}/tiingo/daily/'
        self.__intra_day_url = f'{self.__base_url}/iex/'
        self.__tickers = {
                'MMM': '3M Company',
                'ABBV': 'AbbVie Inc',
                'ABMD': 'ABIOMED Inc',
                'ATVI': 'Activision Blizzard',
                'ADBE': 'Adobe Systems Inc.',
                'AMD': 'Advanced Micro Devices Inc',
                'A': 'Agilent Technologies Inc',
                'AGYS': 'Agilysys Inc',
                'ALB': 'Albemarle Corp',
                'GOOGL': 'Alphabet Inc Class A',
                'AMZN': 'Amazon Inc.',
                'AAL': 'American Airlines Group Inc',
                'AMGN': 'Amgen Inc.',
                'AXP': 'American Express Company AMEX',
                'AAPL': 'Apple Inc.',
                'T': 'AT & T',
                'BIDU': 'Baidu Inc.',
                'BAC': 'Bank of America Corp',
                'GOLD': 'Barrick Gold Corporation BC',
                'BIIB': 'Biogen Idec Inc',
                'BB': 'BlackBerry Ltd',
                'BA': 'Boeing Company',
                'BKNG': 'Booking Holdings Inc',
                'BXP': 'Boston Properties Inc',
                'BMY': 'BristolMyers Squibb Company',
                'CAT': 'Caterpillar Inc',
                'CVX': 'Chevron Corp',
                'CHD': 'Church Dwight Company Inc',
                'CSCO': 'Cisco Systems Inc',
                'C': 'Citigroup Inc',
                'KO': 'CocaCola Company',
                'CL': 'ColgatePalmolive Company',
                'DE': 'Deere Company',
                'DOW': 'Dow Inc',
                'EBAY': 'eBay Inc.',
                'EIX': 'Edison International',
                'EA': 'Electronic Arts Inc',
                'EL': 'Estee Lauder Companies Inc',
                'EXPE': 'Expedia Inc',
                'XOM': 'Exxon Mobil Corp',
                'FB': 'Facebook Inc',
                'FDX': 'FedEx Corp',
                'FSLR': 'First Solar Inc',
                'FL': 'Foot Locker Inc',
                'FCX': 'FreeportMcMoRan Copper Gold Inc',
                'GE': 'General Electric Company',
                'GS': 'Goldman Sachs Group Inc',
                'GRPN': 'Groupon Inc',
                'HD': 'Home Depot Inc',
                'HON': 'Honeywell International Inc',
                'IBM': 'IBM International Business Machines Corp',
                'ILMN': 'Illumina Inc',
                'INTC': 'Intel Corp',
                'JPM': 'JPMorgan Chase Company',
                'JNJ': 'Johnson & Johnson',
                'KEY': 'KeyCorp',
                'KMI': 'Kinder Morgan Inc',
                'KLAC': 'KLA-Tencor Corp',
                'KHC': 'Kraft Heinz Company',
                'LMT': 'Lockheed Martin Corp',
                'MA': 'Mastercard Inc',
                'MCD': 'McDonalds Corp',
                'MRK': 'Merck Company Inc',
                'MSFT': 'Microsoft',
                'MS': 'Morgan Stanley',
                'NFLX': 'Netflix Inc',
                'NKE': 'Nike Inc',
                'NVDA': 'NVIDIA Corp',
                'ORCL': 'Oracle Corp',
                'PGRE': 'Paramount Group Inc',
                'PYPL': 'PayPal Holdings Inc',
                'PEP': 'PepsiCo Inc',
                'PFE': 'Pfizer Inc',
                'PM': 'Philip Morris International Inc',
                'PXD': 'Pioneer Natural Resources',
                'PG': 'Procter Gamble Company',
                'QCOM': 'QUALCOMM Inc',
                'RTN': 'Raytheon Company',
                'REGN': 'Regeneron Pharmaceuticals Inc',
                'RMD': 'ResMed Inc',
                'CRM': 'Salesforce.com Inc',
                'STX': 'Seagate Technology PLC',
                'SLB': 'Schlumberger Ltd',
                'WORK': 'Slack Technologies',
                'SNAP': 'Snap Inc',
                'SNE': 'Sony Corporation',
                'S': 'Sprint Corp',
                'SBUX': 'Starbucks Corp',
                'TGT': 'Target Corp',
                'TSLA': 'Tesla Motors Inc',
                'TEVA': 'Teva Pharmaceutical Industries Ltd ADR',
                'TRV': 'Travelers Companies Inc',
                'TRIP': 'Tripadvisor Inc',
                'TWTR': 'Twitter Inc',
                'UAA': 'Under Armour Inc Class A',
                'UNH': 'UnitedHealth Group Inc',
                'VRSN': 'Verisign Inc',
                'VZ': 'Verizon Communications Inc',
                'V': 'Visa Inc',
                'WBA': 'Walgreens Boots Alliance Inc',
                'WMT': 'WalMart Stores Inc',
                'DIS': 'Walt Disney Company',
                'WFC': 'Wells Fargo Company',
                'YELP': 'Yelp Inc',
                'ZNGA': 'Zynga Inc'
            }
        self.__etf_tickers = {
            'URTH': 'MSCI World ETF'
        }
        self.__data_directory = data_directory
        self.__api_key = api_key

    @property
    def tickers(self):
        return self.__tickers

    @property
    def etf_tickers(self):
        return self.__etf_tickers

    @property
    def dow30_tickers(self):
        dow30 = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM',
                 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA',
                 'WMT']
        dow30_tickers = {}
        for ticker in dow30:
            dow30_tickers[ticker] = self.__tickers[ticker]
        return dow30_tickers

    def load(self, ticker, start_date, end_date, enable_cached_data=True):
        relative_file_path = self.__ensure_cache_file_path(ticker, start_date)
        if not os.path.exists(relative_file_path) or not enable_cached_data:
            quotes = self.__load_from_provider(ticker, start_date, end_date)
            if quotes is None:
                return None
            quotes.to_csv(relative_file_path)
        quotes = pd.read_csv(relative_file_path)
        quotes['date'] = pd.to_datetime(quotes['date'], format='%Y-%m-%d')
        quotes.fillna(method='bfill', inplace=True)
        return quotes

    def load_intra_day(self, ticker, start_date, end_date, enable_cached_data=True):
        relative_file_path = self.__ensure_cache_file_path(ticker, start_date, True)
        if not os.path.exists(relative_file_path) or not enable_cached_data:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            final_end = datetime.strptime(end_date, '%Y-%m-%d')
            end = self.__get_last_day_of_next_month(start)
            data = None
            while start < final_end:
                print(f'{ticker} - {start} > {end}')
                quotes = self.__load_intraday_from_provider(ticker, start, end)
                start = (self.__get_first_day_of_next_month(start)
                         if quotes is None or not len(quotes) > 0
                         else quotes['date'].iloc[-1] + timedelta(days=1))
                end = self.__get_last_day_of_next_month(start)
                if quotes is None or not len(quotes) > 0:
                    continue
                quotes = quotes.set_index('date', drop=False)
                data = quotes if data is None else data.append(quotes)
            if data is None:
                return None
            data.to_csv(relative_file_path)
        quotes = pd.read_csv(relative_file_path)
        quotes['date'] = pd.to_datetime(quotes['date'], format='%Y-%m-%d %H:%M:%S')
        quotes.fillna(method='bfill', inplace=True)
        return quotes

    def __ensure_cache_file_path(self, ticker, start_date, intraday=False):
        sub_directory = f'eod/{start_date}'
        if intraday:
            sub_directory = f'intraday/{start_date}'
        relative_directory = f'{self.__data_directory}/{sub_directory}'
        os.makedirs(relative_directory, exist_ok=True)
        relative_file_path = f'{relative_directory}/{ticker}.csv'
        return relative_file_path

    def __load_from_provider(self, ticker, start_date, end_date):
        raw_json = self.__fetch_eod_data(ticker, start_date, end_date)
        if raw_json is None:
            return None
        quotes = DataProvider.__parse_eod_data(raw_json, ticker)
        if quotes is None:
            return None
        quotes = DataProvider.__enforce_data_types(quotes)
        return quotes

    def __load_intraday_from_provider(self, ticker, start_date, end_date):
        raw_json = self.__fetch_intraday_data(ticker, start_date, end_date)
        if raw_json is None:
            return None
        quotes = DataProvider.__parse_data(raw_json, ticker)
        if quotes is None:
            return None
        quotes = DataProvider.__enforce_data_types(quotes)
        return quotes

    def __fetch_eod_data(self, ticker, start_date, end_date):
        url = f'{self.__eod_url}{ticker}/prices'
        payload = {'startDate': start_date,
                   'endDate': end_date,
                   'columns': 'date,adjOpen,adjHigh,adjLow,adjClose',
                   'token': self.__api_key}
        raw_json = DataProvider.__request(url, payload)
        return raw_json

    def __fetch_intraday_data(self, ticker, start_date, end_date):
        url = f'{self.__intra_day_url}{ticker}/prices'
        payload = {'startDate': self.__format_date(start_date),
                   'endDate': self.__format_date(end_date),
                   'columns': 'open,high,low,close',
                   'resampleFreq': '5min',
                   'token': self.__api_key}
        raw_json = DataProvider.__request(url, payload)
        return raw_json

    @staticmethod
    def __request(url, params=None):
        try:
            if params is None:
                params = {}
            params['Content-Type'] = 'application/json'
            response = requests.get(url, params=params)
            return response.json()
        except requests.exceptions.ConnectionError as ex:
            print(f'Failed to connect to {url} (error:  {ex})')
            return None
        except Exception as ex:
            print(f'Unknown error during connect to {url} (error: {ex})')
            return None

    @staticmethod
    def __parse_eod_data(json, ticker):
        quotes = DataProvider.__parse_data(json, ticker)
        if quotes is None:
            return None
        columns = {'adjClose': 'close', 'adjHigh': 'high', 'adjLow': 'low', 'adjOpen': 'open'}
        quotes.rename(columns=columns, inplace=True)
        return quotes

    @staticmethod
    def __parse_data(json, ticker):
        if json is None or len(json) == 0:
            print(f'{ticker:5} - NO DATA')
            return None
        if 'detail' in json and 'Error' in json['detail']:
            print(json['detail'])
            return None
        quotes = pd.DataFrame.from_dict(json)
        return quotes

    @staticmethod
    def __enforce_data_types(quotes):
        if len(quotes) > 0:
            quotes['date'] = pd.to_datetime(quotes['date'], format='%Y-%m-%d')
            quotes = quotes.astype({'open': 'float', 'high': 'float',
                                    'low': 'float', 'close': 'float',
                                    'date': 'datetime64[ns]'})
        return quotes

    @staticmethod
    def __format_date(date):
        if isinstance(date, str):
            formatted_date = date
        else:
            formatted_date = date.strftime('%Y-%m-%d')
        return formatted_date

    @staticmethod
    def __get_first_day_of_next_month(date: datetime):
        day = datetime(date.year + (1 if date.month == 12 else 0),
                       1 if date.month == 12 else date.month + 1,
                       1)
        return day

    @staticmethod
    def __get_last_day_of_next_month(date: datetime):
        day = datetime(date.year + (1 if date.month == 12 else 0),
                       1 if date.month == 12 else date.month + 1,
                       1) + timedelta(days=-1)
        return day
