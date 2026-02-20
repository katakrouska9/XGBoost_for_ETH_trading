import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report



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

#Defining the ML dataset - Part1

X_prep = data[['Close']].rename(columns = {'Close': 'P_t'}) #Working with closing price
X_prep['P_t-1'] = X_prep['P_t'].shift(1)
X_prep['hourly_return'] = ((X_prep['P_t']/X_prep['P_t-1'])-1)

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

X_prep.head(30)


## BACKTESTING - defining max theoretical return with look-ahead bias ## Defining position

'''
    Creating a position to take based on buy command of previous function

    Algorithm stays long until tp is reached, stop-loss is triggered or max horizon is reached.

    Not in position as default, when algorithm reaches 1 and is NOT in a position, it goes long. Then it ignores all 1s until out of position.
    
    '''

def create_stable_position(df, tp=tp, sl=sl, horizon=horizon):
    prices = df["P_t"].values
    targets = df["target"].values
    position = np.zeros(len(df))
    in_pos = False
    exit_idx = 0
    for i in range(len(df)):
        if in_pos:
            position[i] = 1
            if i >= exit_idx:
                in_pos = False
            continue
        if targets[i] == 1:
            in_pos = True
            future = prices[i: i + horizon]
            if len(future) == 0:
                in_pos = False
                continue
            tp_lvl = prices[i-1] * (1 + tp)
            sl_lvl = prices[i-1] * (1 - sl)
            tp_hits = np.where(future >= tp_lvl)[0]
            sl_hits = np.where(future <= sl_lvl)[0]
            f_tp = tp_hits[0] if len(tp_hits) > 0 else float("inf")
            f_sl = sl_hits[0] if len(sl_hits) > 0 else float("inf")
            duration = min(f_tp, f_sl, horizon - 1)
            exit_idx = i + duration
            position[i] = 1
    return position

X_prep['stable_position'] = create_stable_position(X_prep)
X_prep['stable_position'].value_counts()

#Calculating cummulative return before fees
def cum_return(df):
    df['perfect_strategy'] = (df['hourly_return']* df['stable_position']).fillna(0)
    df['perfect_cum_return'] = (1+df['perfect_strategy']).cumprod()
    return df

X_prep = cum_return(X_prep)

plt.plot(X_prep['perfect_cum_return'])
plt.show()

#Calculating cummulative return after fees

# Marking time of buy and sell

def cum_return_after_fees(df):
    df['is_buy'] = (df['stable_position'].diff() == 1) | (df['stable_position'].diff() == -1)
    fee = 0.001

    df['perfect_strategy_w_fees'] = np.where(
        df['is_buy'] == True, 
        df['perfect_strategy'] - fee, 
        df['perfect_strategy']
    )
    df['perfect_cum_return_w_fees'] = (1+df['perfect_strategy_w_fees']).cumprod()
    return df

X_prep = cum_return_after_fees(X_prep)

plt.plot(X_prep['perfect_cum_return_w_fees'])
plt.show()

maximal_theoretical_return = X_prep['perfect_cum_return_w_fees'][-1]
print(maximal_theoretical_return)
#print(X_prep.head(60).sort_values('Datetime')[['stable_position',"is_buy"]])

#Defining the ML dataset - Part2 - adding additional features

#SMAs
X = pd.DataFrame(X_prep['hourly_return'].shift(1)).rename(columns = {'hourly_return': "hourly_return_t-1"})
X_prep['SMA20_t-1'] = X_prep['P_t-1'].rolling(20).mean()
X['Distance to SMA20'] = (X_prep['P_t-1']/X_prep['SMA20_t-1'])-1
X_prep['SMA50_t-1']= X_prep['P_t-1'].rolling(50).mean()
X['Distance to SMA50'] = (X_prep['P_t-1']/X_prep['SMA50_t-1'])-1
X_prep['SMA200_t-1']= X_prep['P_t-1'].rolling(200).mean()
X['Distance to SMA200'] = (X_prep['P_t-1']/X_prep['SMA200_t-1'])-1

#ATR
def calculate_ATR(df, period = 24):
    hl = abs(df['High']- df['Low'])
    hc = abs(df['High']-df['Close'].shift(1))
    lc = abs(df['Low']-df['Close'].shift(1))
    
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = tr.rolling(period).mean().shift(1)
    atr_rel = atr/ (df['Close'].shift(1))

    return pd.DataFrame({
        'ATR': atr,
        'ATR_rel': atr_rel
    }, index=df.index)

results_atr = calculate_ATR(data)
X[['ATR','ATR_rel']] = results_atr
X['ATR_rel'].mean()
print(X.head(50))

#Time info

X['hour'] = X.index.hour
X['weekday'] = X.index.dayofweek

# Cumulative return - last 12 hours

X['cum_return_12h'] = (X_prep['P_t-1']/X_prep['P_t-1'].shift(12))-1

#Adding Btc as a feature

Btc = yf.download("BTC-USD", period="730d", interval="1h")
Btc.columns = data.columns.get_level_values(0)

Btc.head()
Btc.isna().sum()
Btc.describe()

Btc['hourly_return'] = ((Btc['Close']/Btc['Close'].shift(1))-1)
X['btc_gap'] = X['hourly_return_t-1']-Btc['hourly_return'].shift(1) # ETH follows BTC with a lag
#Btc['24h_average_return']= Btc['hourly_return'].rolling(24).mean()
#X['24h_average_return'] = X['hourly_return_t-1'].rolling(24).mean()
Btc['hourly_return'].shift(1).corr(X['hourly_return_t-1'])

#droping columns not meant for ML
X=X.drop(columns = ['ATR'])
X['target'] = X_prep['target']
X['position'] = X_prep['stable_position'].astype(int)



#Dropping first 50 rows with NaN & last 48 rows with non-reliable target values
X_crop= X.iloc[200:,:]
X_crop = X_crop.iloc[:-48]
X_crop.head(50)
X_crop.isna().sum()

#                   XGBOOST ALGORITHM TRAINING
##############################################################

y_xg = X_crop['target']
X_xg = X_crop.drop(columns = ['position', 'target'])

X_xg_train, X_xg_test, y_xg_train, y_xg_test = train_test_split(X_xg,y_xg, test_size=0.3, shuffle= False)

model = XGBClassifier(
    n_estimators=100,     # Dostatek pokusů na učení
    learning_rate=0.02,   # Pomalé a precizní učení
    max_depth=4,          # Jednoduchá, robustní pravidla
    subsample=0.7,        # Trénuj jen na části dat pro každý strom
    colsample_bytree=0.7,
    scale_pos_weight=2.96, # Náhodně vybírej indikátory
    random_state=42
)
model.fit(X_xg_train,y_xg_train)
y_pred = model.predict(X_xg_test)
print(f"Accuracy: {accuracy_score(y_xg_test, y_pred):.2f}")
print(classification_report(y_xg_test, y_pred))

# Stricter buying threshold
y_probs = model.predict_proba(X_xg_test)[:, 1]
threshold = 0.60
y_pred_strict = (y_probs >= threshold).astype(int)

print(f"Accuracy: {accuracy_score(y_xg_test, y_pred_strict):.2f}")
print(classification_report(y_xg_test, y_pred_strict))

comparison = pd.DataFrame({
    'Skutečnost (Reality)': y_xg_test.values,
    'Predikce (Model)': y_pred_strict
}, index=y_xg_test.index)

print(comparison.head(50))

#XG_return = X_xg_test['hourly_return_t-1'].shift(-1)* y_pred
#cummulative_XG = (1+XG_return).cumprod()

#Posuvne okno

#Rolling 6month training, one month test
train_size = 24*30*6
test_size = 24*30
step = test_size

results=[]
y_pred_rolling = []

for start in range(0, len(X_xg) - train_size - test_size, step):

    train_end = start + train_size
    test_end = train_end + test_size

    X_train, X_test = X_xg.iloc[start:train_end], X_xg.iloc[train_end:test_end]
    y_train, y_test = y_xg.iloc[start:train_end], y_xg.iloc[train_end:test_end]

    spw = (y_train == 0).sum()/(y_train == 1).sum()

    model = XGBClassifier(n_estimators = 100, max_depth = 4, scale_pos_weight = spw, eval_metric = "logloss" )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    y_pred = (probs > 0.60).astype(int)
    y_pred_series = pd.Series(y_pred, index=X_test.index)

    report = classification_report(y_test, y_pred, output_dict= True, zero_division= 0)
    prec1 = report['1']['precision']
    rec1 = report['1']['recall']
    f1_score = report['1']['f1-score']
    n_trades = y_pred.sum().item()

    results.append({'start_date': X_test.index[0], 'end_date': X_test.index[-1],'precision_1': prec1, 'recall_1': rec1, 'f1_score': f1_score, 'n_trades': n_trades})
        
    y_pred_rolling.append(y_pred_series)
    
results_call = pd.DataFrame(results)
print(results_call)
results_call['precision_1'].mean()
results_call['recall_1'].mean()

all_predictions = pd.concat(y_pred_rolling)
all_predictions

#                   BACKTEST
##############################################################

#Creating position -- ONLY IF NOT PREDICTING POSITION ALREADY
backtest_df = (pd.DataFrame(all_predictions.copy())).rename(columns = {0: 'target'})
backtest_df['P_t'] = X_prep['P_t']

backtest_df['stable_position']= create_stable_position(backtest_df)


#Adding return
#backtest_df = (pd.DataFrame(all_predictions.copy())).rename(columns = {0: 'stable_position'})
backtest_df['hourly_return']= X_prep['hourly_return'] #merged based on index


# Cum return before fees
backtest_df = cum_return(backtest_df)

backtest_df = cum_return_after_fees(backtest_df)
print(backtest_df['perfect_cum_return'][-1],backtest_df['perfect_cum_return_w_fees'][-1])
print(backtest_df.tail(50))


plt.plot(backtest_df['perfect_cum_return_w_fees'])
plt.plot(backtest_df['perfect_cum_return'])
plt.show()