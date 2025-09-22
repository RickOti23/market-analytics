import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from datetime import date
import os
import pandas_datareader as pdr
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class DataRepo:
    stocks_df: pd.DataFrame
    macro_df: pd.DataFrame
    indices_df: pd.DataFrame

    def __init__(self):
        self.stocks_df = None
        self.macro_df = None
        self.indices_df = None
        self.today = date.today()
        self.start = date(year=self.today.year - 70, month=self.today.month, day=self.today.day)
        self.successful_tickers = []
        
        self.tickers = {
            "USA": [
                "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NVDA", "INTC", "AMD", "CSCO",
                "JPM", "BAC", "GS", "WFC", "MS", "JNJ", "PFE", "MRK", "UNH", "ABT",
                "HD", "NKE", "MCD", "SBUX", "LOW", "PG", "KO", "PEP", "WMT", "COST",
                "GE", "CAT", "BA", "UPS", "DE", "XOM", "CVX", "NEE", "DUK", "SLB",
                "NFLX", "PYPL", "ADBE", "CRM", "TMUS", "ORCL", "IBM", "QCOM", "INTU", "TXN"
            ],
            "CHINA": [
                "BABA", "JD", "PDD", "BIDU", "NTES", "TCEHY", "NIO", "XPEV", "LI", "EDU",
                "TAL", "BILI", "ZTO", "YUMC", "BEKE", "HTHT", "WB", "CAN", "VIOT", "DAO",
                "VIPS", "MOMO","HKD", "CHWY", "IQ", "HUYA", "DOYU", "KWEB"
            ],
            "INDIA": [
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS",
                "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ASIANPAINT.NS", "BAJFINANCE.NS",
                "HCLTECH.NS", "WIPRO.NS", "KOTAKBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
                "TITAN.NS", "TECHM.NS", "POWERGRID.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "TATAMOTORS.NS",
                "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "COALINDIA.NS",
                "BAJAJ-AUTO.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "BPCL.NS",
                "BRITANNIA.NS", "DIVISLAB.NS", "IOC.NS", "HAVELLS.NS", "CIPLA.NS", "TATAELXSI.NS",
                "INDUSINDBK.NS", "DLF.NS", "PIDILITIND.NS", "M&M.NS", "GAIL.NS", "SHREECEM.NS",
                "HINDALCO.NS", "ICICIPRULI.NS"
            ],
            "GERMANY": [
                "ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "CBK.DE", "CON.DE","DBK.DE",
                "DB1.DE","DTE.DE", "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE", "HEN3.DE", "IFX.DE",
                "LIN.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "RWE.DE", "SAP.DE", "SIE.DE", "VOW3.DE", "VNA.DE",
                "1COV.DE", "ZAL.DE", "BVB.DE", "SY1.DE", "FIE.DE", "RRTL.DE", "PSM.DE", "SDF.DE","HFG.DE"
            ]
        }

        self.indices = ["^GDAXI","^GSPC","^DJI","EPI","^VIX","GC=F","CL=F","BZ=F","BTC-USD"]

    def get_stock_data(self):
        stocks_df = pd.DataFrame()
        all_tickers = []
        for country_tickers in self.tickers.values():
            all_tickers.extend(country_tickers)

        for ticker in tqdm(all_tickers, desc="Fetching stock data"):
            try:
                ticker_obj = yf.Ticker(ticker)
                historyPrices = ticker_obj.history(period="max", interval="1d")

                if historyPrices.empty:
                    print(f"Skipped {ticker}: No data available.")
                    continue

                historyPrices.index = pd.to_datetime(historyPrices.index, errors='coerce')
                historyPrices = historyPrices.dropna(subset=["Close"])
                if historyPrices.empty:
                    print(f"Skipped {ticker}: Data had no valid Close prices.")
                    continue

                #Transforming the data
                historyPrices["Ticker"] = ticker
                historyPrices["Year"] = historyPrices.index.year
                historyPrices["Month"] = historyPrices.index.month
                historyPrices["Weekday"] = historyPrices.index.weekday
                historyPrices["Date"] = pd.to_datetime(historyPrices.index.date)

                for window in [1, 3, 7, 30, 90, 365]:
                    historyPrices[f"growth_{window}d"] = historyPrices["Close"] / historyPrices["Close"].shift(window)

                historyPrices["growth_future_30d"] = historyPrices["Close"].shift(-30) / historyPrices["Close"]

                historyPrices["SMA10"] = historyPrices["Close"].rolling(10).mean()
                historyPrices["SMA20"] = historyPrices["Close"].rolling(20).mean()
                historyPrices["growing_moving_average"] = np.where(historyPrices["SMA10"] > historyPrices["SMA20"], 1, 0)
                historyPrices["high_minus_low_relative"] = (historyPrices["High"] - historyPrices["Low"]) / historyPrices["Close"]
                historyPrices["volatility"] = historyPrices["Close"].rolling(30).std() * np.sqrt(252)

                historyPrices["is_positive_growth_30d_future"] = np.where(historyPrices["growth_future_30d"] > 1, 1, 0)

                stocks_df = pd.concat([stocks_df, historyPrices], ignore_index=True, sort=False)
                self.stocks_df = stocks_df
                self.successful_tickers.append(ticker)

                time.sleep(1)

            except Exception as e:
                print(f"Error with ticker {ticker}: {e}")
        stocks_df = stocks_df[stocks_df["Year"]>=2000]
        self.stocks_df = stocks_df

    def fetch_index(self):
        index_df = pd.DataFrame()

        for index in tqdm(self.indices, desc="Fetching indices data"):
            try:
                ticker_obj = yf.Ticker(index)
                historyPrices = ticker_obj.history(period="max", interval="1d")

                if historyPrices.empty:
                    print(f"Skipped {index}: No data available.")
                    continue

                # Ensure the index is datetime
                historyPrices.index = pd.to_datetime(historyPrices.index)
                historyPrices = historyPrices.dropna(subset=["Close"])
                
                # Reset index to move the datetime into a column
                historyPrices.reset_index(inplace=True)  # This creates a 'Date' column
                historyPrices["Ticker"] = index

                # Add growth columns
                for window in [1, 3, 7, 30, 90, 365]:
                    historyPrices[f"growth_{index}_{window}d"] = historyPrices["Close"] / historyPrices["Close"].shift(window)

                # Keep only Date, Ticker, and growth columns
                columns_to_keep = ["Date"] + [col for col in historyPrices.columns if col.startswith("growth")]
                historyPrices = historyPrices[columns_to_keep]

                historyPrices["Date"] = pd.to_datetime(historyPrices["Date"]).dt.tz_localize(None)

                # Group by Date, aggregate each column by first non-null value (or mean, etc.)
                # This assumes that for a given Date, the values are unique per column.
                historyPrices = historyPrices.groupby("Date").first().reset_index()

                index_df = pd.concat([index_df, historyPrices], ignore_index=True)

                time.sleep(1)  # To avoid hitting rate limits

            except Exception as e:
                print(f"Error with ticker {index}: {e}")

        if not index_df.empty:
            index_df["Date"] = pd.to_datetime(index_df["Date"]).dt.tz_localize(None)
            index_df = index_df[index_df["Date"] >= pd.to_datetime("2000-01-01")]
            self.indices_df = index_df
            self.indices_df = self.indices_df.reset_index()
            self.indices_df["Date"] = pd.to_datetime(self.indices_df["Date"]).dt.tz_localize(None)
            self.indices_df = self.indices_df.groupby("Date",as_index=False).first()
            self.indices_df.fillna(0,inplace=True)
        else:
            print("No valid index data fetched.")
            self.indices_df = pd.DataFrame()
            
    def fetch_macro(self):
        '''Fetch Macro data from FRED (using Pandas datareader)'''

        min_date = "2000-01-01"
        

        # Real Potential Gross Domestic Product (GDPPOT), Billions of Chained 2012 Dollars, QUARTERLY
        # https://fred.stlouisfed.org/series/GDPPOT
        gdppot = pdr.DataReader("GDPPOT", "fred", start=min_date)
        gdppot.index = gdppot.index.tz_localize(None)
        gdppot['gdppot_us_yoy'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(4)-1
        gdppot['gdppot_us_qoq'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(1)-1
        gdppot_to_merge = gdppot[['gdppot_us_yoy','gdppot_us_qoq']]
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # # "Core CPI index", MONTHLY
        # https://fred.stlouisfed.org/series/CPILFESL
        cpilfesl = pdr.DataReader("CPILFESL", "fred", start=min_date)
        cpilfesl.index =  cpilfesl.index.tz_localize(None)
        cpilfesl['cpi_core_yoy'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(12)-1
        cpilfesl['cpi_core_mom'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(1)-1
        cpilfesl_to_merge = cpilfesl[['cpi_core_yoy','cpi_core_mom']]   
        time.sleep(1)

        # Fed rate https://fred.stlouisfed.org/series/FEDFUNDS
        fedfunds = pdr.DataReader("FEDFUNDS", "fred", start=min_date)
        fedfunds.index =  fedfunds.index.tz_localize(None)
        time.sleep(1)


        # https://fred.stlouisfed.org/series/DGS1
        dgs1 = pdr.DataReader("DGS1", "fred", start=min_date)
        dgs1.index =  dgs1.index.tz_localize(None)
        time.sleep(1)

        # https://fred.stlouisfed.org/series/DGS5
        dgs5 = pdr.DataReader("DGS5", "fred", start=min_date)
        dgs5.index =  dgs5.index.tz_localize(None)
        time.sleep(1)

        # https://fred.stlouisfed.org/series/DGS10
        dgs10 = pdr.DataReader("DGS10", "fred", start=min_date)
        dgs10.index = dgs10.index.tz_localize(None)
        time.sleep(1)

         # Merging - start from daily stats (dgs1)
        m2 = pd.merge(dgs1,
                    dgs5,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one')
        
        m2.reset_index(inplace=True)
        m2["DATE"] = pd.to_datetime(m2["DATE"]).dt.tz_localize(None) 
        m2['Quarter'] = m2["DATE"].dt.to_period('Q').dt.to_timestamp()

        m3 = pd.merge(m2,gdppot_to_merge,left_on='Quarter',right_index=True,how='left',validate='many_to_one')
        m3['Month'] = m2["DATE"].dt.to_period('M').dt.to_timestamp()
        m4 = pd.merge(m3,
                    fedfunds,
                    left_on='Month',
                    right_index=True,
                    how='left',
                    validate='many_to_one')
        
        m5 = pd.merge(m4,
                    cpilfesl_to_merge,
                    left_on='Month',
                    right_index=True,
                    how='left',
                    validate='many_to_one')
        
        m5.set_index('DATE',inplace=True)
        
        m6 = pd.merge(m5,
                    dgs10,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one')
        
        fields_to_fill = ['cpi_core_yoy','cpi_core_mom','FEDFUNDS','DGS1','DGS5','DGS10']
        # Fill missing values in selected fields with the last defined value
        for field in fields_to_fill:
            m6[field] = m6[field].ffill()

        self.macro_df =  m6
        self.macro_df = self.macro_df.reset_index()
        self.macro_df.rename(columns={"DATE": "Date"}, inplace=True)
        self.macro_df["Date"] = pd.to_datetime(self.macro_df["Date"]).dt.tz_localize(None)

    def fetch(self):
        '''Fetch all data from APIs'''
        print('Fetching Tickers info from YFinance')
        self.get_stock_data()
        print('Fetching Indexes info from YFinance')
        self.fetch_index()
        print('Fetching Macro info from FRED (Pandas_datareader)')
        self.fetch_macro()
    
    def persist(self, data_dir: str):
        '''Save dataframes to files in a local directory 'data_dir' '''
        os.makedirs(data_dir, exist_ok=True)

        # Save stock data
        if hasattr(self, 'stocks_df') and self.stocks_df is not None and not self.stocks_df.empty:
            file_name = 'stocks_df.parquet'
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            self.stocks_df.to_parquet(file_path, compression='brotli')
            print(f"Saved {len(self.stocks_df)} stock records")
        else:
            print("No ticker data to save")

        # Save index data
        if hasattr(self, 'indices_df') and self.indices_df is not None and not self.indices_df.empty:
            file_name = 'indices_df.parquet'
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            self.indices_df.to_parquet(file_path, compression='brotli')
            print(f"Saved {len(self.indices_df)} index records")
        else:
            print("No index data to save")

        # Save macro data
        if self.macro_df is not None and not self.macro_df.empty:
            file_name = 'macro_df.parquet'
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            self.macro_df.to_parquet(file_path, compression='brotli')
            print(f"Saved {len(self.macro_df)} macro records")
        else:
            print("No macro data to save")

    def load(self, data_dir: str):
        """Load dataframes from the local directory"""
        stocks_path = os.path.join(data_dir, 'stocks_df.parquet')
        index_path = os.path.join(data_dir, 'indices_df.parquet')
        macro_path = os.path.join(data_dir, 'macro_df.parquet')

        if os.path.exists(stocks_path):
            self.stocks_df = pd.read_parquet(stocks_path)
            print(f"Loaded {len(self.stocks_df)} ticker records")
        else:
            print("Ticker file not found.")

        if os.path.exists(index_path):
            self.indices_df = pd.read_parquet(index_path)
            print(f"Loaded {len(self.indices_df)} index records")
        else:
            print("Index file not found.")

        if os.path.exists(macro_path):
            self.macro_df = pd.read_parquet(macro_path)
            print(f"Loaded {len(self.macro_df)} macro records")
        else:
            print("Macro file not found.")
