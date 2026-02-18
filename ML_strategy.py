import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
import pandas as pd


# Download price

ticker_1 = "BTC-USD"
ticker_2 = "ETH-USD"

pd.set_option("display.max_columns", None)

data = yf.download("ETH-USD", period="730d", interval="1h")
#plt.plot(data['Close'])
#plt.show()

#Checking summary statistics

data.head()
data.isna().sum() #no n/a
data.describe()
data.count() 





