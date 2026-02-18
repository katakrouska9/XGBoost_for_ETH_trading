import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
import pandas as pd


# Download price

ticker_1 = "BTC-USD"
ticker_2 = "ETH-USD"

pd.reset_option("display.max_rows", None)

data = yf.download("ETH-USD", period="730d", interval="1h")
data.columns = data.columns.get_level_values(0)
#plt.plot(data['Close'])
#plt.show()

#Checking summary statistics

data.head()
data.isna().sum() #no n/a
data.describe()
data.count() 

#Defining the ML dataset

X_prep = data[['Close']].rename(columns = {'Close': 'P_t'})
X_prep['P_t-1'] = X_prep['P_t'].shift(1)
X_prep['Hourly_return_pct'] = ((X_prep['P_t']/X_prep['P_t-1'])-1)*100

X_prep['P_t'].len()

##Defining buy classification:

def create_target(X_prep, horizon=48, tp=0.04, sl=0.02):
    """
    Creates classification 1 or 0: 1 when algorithm should buy, 0 not buy

    1: Price will grow at least tp% in the horizont, before significant fall sl that would trigger stop loss
    
    :param X_prep: Description
    :param horizon: Description
    :param tp: Description
    :param sl: Description
    """
    results = []
    prices = X_prep['P_t'].values 
    
    for i in range(len(prices)):
        entry_price = prices[i-1] #I buy for closing price of previous hour
        future_window = prices[i : i+horizon]
        
        if len(future_window) == 0:
            results.append(0)
            continue

        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)

        
        tp_hits = np.where(future_window >= tp_price)[0]
        sl_hits = np.where(future_window <= sl_price)[0]

        if len(tp_hits) > 0:
            first_tp = tp_hits[0]
            # If SL doesnt happen, assign infinity so that tp dominates
            first_sl = sl_hits[0] if len(sl_hits) > 0 else float('inf')
            
            if first_tp < first_sl:
                results.append(1)
            else:
                results.append(0)
        else:
            results.append(0)
            
    return results

X_prep['target'] = create_target(X_prep)
X_prep['target'].value_counts(normalize = True)

plt.plot(X_prep['target'])
plt.show()
X_prep.head(30)



## BACKTESTING ## Defining position

def create_stable_position(X_prep, tp=0.04, sl=0.02, horizon=48):
    '''
    Creating a position to take based on buy command of previous function

    Algorithm stays long until tp is reached, stop-loss is triggered or max horizon is reached.
    
    '''
    prices = X_prep["P_t"].values
    targets = X_prep["target"].values
    position = np.zeros(len(X_prep))
    in_pos = False
    exit_idx = 0
    for i in range(len(X_prep)):
        if in_pos:
            position[i] = 1
            if i >= exit_idx:
                in_pos = False
            continue
        if targets[i] == 1:
            in_pos = True
            future = prices[i + 1 : i + horizon + 1]
            if len(future) == 0:
                in_pos = False
                continue
            tp_lvl = prices[i] * (1 + tp)
            sl_lvl = prices[i] * (1 - sl)
            tp_hits = np.where(future >= tp_lvl)[0]
            sl_hits = np.where(future <= sl_lvl)[0]
            f_tp = tp_hits[0] if len(tp_hits) > 0 else float("inf")
            f_sl = sl_hits[0] if len(sl_hits) > 0 else float("inf")
            duration = min(f_tp, f_sl, horizon - 1)
            exit_idx = i + duration + 1
            position[i] = 1
    return position

X_prep['stable_position'] = create_stable_position(X_prep, tp=0.04, sl=0.02, horizon=48)
X_prep['stable_position'].value_counts()
X_prep.head(70)


