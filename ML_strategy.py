import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
import pandas as pd



#                   DATASET PREPARATION
##############################################################

# Download price

pd.set_option("display.max.columns", None)

data = yf.download("ETH-USD", period="730d", interval="1h")
data.columns = data.columns.get_level_values(0)

#Ploting 

#plt.plot(data['Close'])
#plt.show()

#Checking summary statistics

data.head()
data.isna().sum() #no n/a
data.describe()
data.count() 

#Defining the ML dataset

X_prep = data[['Close']].rename(columns = {'Close': 'P_t'}) #Working with closing price
X_prep['P_t-1'] = X_prep['P_t'].shift(1)
X_prep['hourly_return_pct'] = ((X_prep['P_t']/X_prep['P_t-1'])-1)

##Defining buy classification:

tp= 0.04
sl = 0.02
horizon = 48

"""
    Creates classification 1 or 0: 1 when algorithm should buy, 0 not buy

    1: Price will grow at least tp% in the horizont, before significant % fall (sl) that would trigger stop loss

    :param X_prep: Original dataset
    :param horizon: Arbitrary variable depending on selected strategy
    :param tp: Arbitrary variable depending on selected strategy
    :param sl: Arbitrary variable depending on selected strategy
    """
def create_target(X_prep, horizon, tp, sl):
    results = []
    prices = X_prep['P_t'].values 
    
    for i in range(len(prices)):
        entry_price = prices[i-1] #I buy for "closing price" of previous hour = opening price of my hour
        future_window = prices[i : i+horizon] 
        
        if len(future_window) == 0:
            results.append(0)
            continue

        #Defining my target price and stop-loss price
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)

        #Defining moments when price >= my target price or <= stop loss price
        tp_hits = np.where(future_window >= tp_price)[0]
        sl_hits = np.where(future_window <= sl_price)[0]

        if len(tp_hits) > 0:
            first_tp = tp_hits[0]
            first_sl = sl_hits[0] if len(sl_hits) > 0 else float('inf') 

            if first_tp < first_sl: 
                results.append(1)
            else:
                results.append(0)
        else:
            results.append(0)

    return results

X_prep['target'] = create_target(X_prep, horizon, tp, sl)
X_prep.iloc[0,3] = 0 # first day must be 0, because doesnt have an opening price
X_prep['target'].value_counts(normalize = True)


plt.plot(X_prep['target'])
plt.show()
X_prep.head(30)


## BACKTESTING ## Defining position

'''
    Creating a position to take based on buy command of previous function

    Algorithm stays long until tp is reached, stop-loss is triggered or max horizon is reached.

    Not in position as default, when algorithm reaches 1 and is NOT in a position, it goes long. Then it ignores all 1s until out of position.
    
    '''

def create_stable_position(X_prep, tp=0.04, sl=0.02, horizon=48):
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
X_prep.head(50)

#Calculating cummulative return before fees
X_prep['perfect_strategy'] = (X_prep['hourly_return_pct']*X_prep['stable_position']).fillna(0)
X_prep['perfect_cum_return'] = (1+X_prep['perfect_strategy']).cumprod()

plt.plot(X_prep['perfect_cum_return'])
plt.show()

#Calculating cummulative return before fees

# Marking time of buy and sell
X_prep['is_buy'] = (X_prep['stable_position'].diff() == 1) | (X_prep['stable_position'].diff() == -1)

fee = 0.001
X_prep['perfect_strategy_w_fees'] = np.where(
    X_prep['is_buy'] == True, 
    X_prep['perfect_strategy'] - fee, 
    X_prep['perfect_strategy']
)
X_prep['perfect_cum_return_w_fees'] = (1+X_prep['perfect_strategy_w_fees']).cumprod()

plt.plot(X_prep['perfect_cum_return_w_fees'])
plt.show()

print(X_prep['perfect_cum_return_w_fees'].max())

#print(X_prep.head(60).sort_values('Datetime')[['stable_position',"is_buy"]])

