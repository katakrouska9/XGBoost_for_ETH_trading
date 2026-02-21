import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import plot_importance



#                   DATASET LOADING 
##############################################################

# Download price

pd.reset_option("display.max.rows", None)

data = yf.download("ETH-USD", period="730d", interval="1h")
data.columns = data.columns.get_level_values(0)

#Checking summary statistics

data.head()
data.isna().sum() #no n/a
data.describe()
data.count() 

#Dataset preparation

X_prep = data[['Close']].rename(columns = {'Close': 'P_t'}) #Working with closing price
X_prep['P_t-1'] = X_prep['P_t'].shift(1)
X_prep['hourly_return'] = ((X_prep['P_t']/X_prep['P_t-1'])-1)


#                   TARGET CALCULATION 
##############################################################

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

## Defining position

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

##############. BACKTESTING  ##################### defining max theoretical return with look-ahead bias

#Calculating cummulative return before fees
"""
Calculates cumulative return using hourly historical return and long/no position
"""
def cum_return(df):
    df['perfect_strategy'] = (df['hourly_return']* df['stable_position']).fillna(0)
    df['perfect_cum_return'] = (1+df['perfect_strategy']).cumprod()
    return df

X_prep = cum_return(X_prep)

#plt.plot(X_prep['perfect_cum_return'])
#plt.show()

#Calculating cummulative return after fees
"""
Calculates cumulative return after substracting a fixed fee in a time of purchase/sale - fee approximates also for slippage
"""
def cum_return_after_fees(df):
    df['is_buy'] = (df['stable_position'].diff() == 1) | (df['stable_position'].diff() == -1)
    fee = 0.0015 #including both fee and average slippage

    df['perfect_strategy_w_fees'] = np.where(
        df['is_buy'] == True, 
        df['perfect_strategy'] - fee, 
        df['perfect_strategy']
    )
    df['perfect_cum_return_w_fees'] = (1+df['perfect_strategy_w_fees']).cumprod()
    return df

X_prep = cum_return_after_fees(X_prep)

#plt.plot(X_prep['perfect_cum_return_w_fees'])
#plt.show()

maximal_theoretical_return = X_prep['perfect_cum_return_w_fees'][-1]
print(maximal_theoretical_return)


#                   FEATURE ENGINEERING - creating X 
##############################################################

#SMAs = moving averages
X = pd.DataFrame(X_prep['hourly_return'].shift(1)).rename(columns = {'hourly_return': "hourly_return_t-1"})
X_prep['SMA20_t-1'] = X_prep['P_t-1'].rolling(20).mean()
X['Distance to SMA20'] = (X_prep['P_t-1']/X_prep['SMA20_t-1'])-1
X_prep['SMA50_t-1']= X_prep['P_t-1'].rolling(50).mean()
X['Distance to SMA50'] = (X_prep['P_t-1']/X_prep['SMA50_t-1'])-1
X_prep['SMA200_t-1']= X_prep['P_t-1'].rolling(200).mean()
X['Distance to SMA200'] = (X_prep['P_t-1']/X_prep['SMA200_t-1'])-1

#ATR = volatility estimate
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

#Time info

X['hour'] = X.index.hour
X['weekday'] = X.index.dayofweek

# Cumulative return - last 12 hours

X['cum_return_12h'] = (X_prep['P_t-1']/X_prep['P_t-1'].shift(12))-1

#RSI (Relative Strength Index)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

X['RSI'] = calculate_rsi(X_prep['P_t'], 14)
X['RSI_t-1'] = X['RSI'].shift(1)

#BTC

Btc = yf.download("BTC-USD", period="730d", interval="1h")
Btc.columns = data.columns.get_level_values(0)

Btc.head()
Btc.isna().sum()
Btc.describe()

Btc['hourly_return'] = ((Btc['Close']/Btc['Close'].shift(1))-1)
X['btc_gap'] = X['hourly_return_t-1']-(Btc['hourly_return'].shift(1)) # Difference between ETH and BTC return


#droping columns not meant for ML
X=X.drop(columns = ['ATR', 'RSI'])
X['target'] = X_prep['target']
X['position'] = X_prep['stable_position'].astype(int)


#Dropping first 50 rows with NaN & last 48 rows with non-reliable target values
X_crop= X.iloc[200:,:]
X_crop = X_crop.iloc[:-48]

#                   XGBOOST ALGORITHM TRAINING
##############################################################

y_xg = X_crop['target']
X_xg = X_crop.drop(columns = ['position', 'target'])


###### 1) One test & train window ####
X_xg_train, X_xg_test, y_xg_train, y_xg_test = train_test_split(X_xg,y_xg, test_size=0.3, shuffle= False)

model_total = XGBClassifier(
    n_estimators=100,     # Dostatek pokusů na učení
    learning_rate=0.02,   # Pomalé a precizní učení
    max_depth=4,          # Jednoduchá, robustní pravidla
    subsample=0.7,        # Trénuj jen na části dat pro každý strom
    colsample_bytree=0.7,
    scale_pos_weight=2.96, # Náhodně vybírej indikátory
    random_state=42
)
model_total.fit(X_xg_train,y_xg_train)
y_pred = model_total.predict(X_xg_test)
print(f"Accuracy: {accuracy_score(y_xg_test, y_pred):.2f}")
print(classification_report(y_xg_test, y_pred))

# Stricter buying threshold
y_probs = model_total.predict_proba(X_xg_test)[:, 1]
threshold = 0.60
y_pred_strict = (y_probs >= threshold).astype(int)

print(f"Accuracy: {accuracy_score(y_xg_test, y_pred_strict):.2f}")
print(classification_report(y_xg_test, y_pred_strict))

#### 2) Widening train window #### 6 months training, 1 month test, then move by one month and repeat

train_start = 0
train_size = 24*30*6
test_size = 24*30
step = test_size

results=[]
y_pred_rolling = []
models_list = []

for train_end in range(train_size, len(X_xg) - test_size, step):

    
    test_end = train_end + test_size

    X_train, X_test = X_xg.iloc[0:train_end], X_xg.iloc[train_end:test_end]
    y_train, y_test = y_xg.iloc[0:train_end], y_xg.iloc[train_end:test_end]

    spw = (y_train == 0).sum()/(y_train == 1).sum()

    model = XGBClassifier(n_estimators = 100, max_depth = 4, scale_pos_weight = spw, eval_metric = "logloss" )
    model.fit(X_train, y_train)
    models_list.append(model)

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

backtest_df['stable_position']= create_stable_position(backtest_df, tp=0.06, sl=sl, horizon=horizon)


#Adding return
#backtest_df = (pd.DataFrame(all_predictions.copy())).rename(columns = {0: 'stable_position'})
backtest_df['hourly_return']= X_prep['hourly_return'] #merged based on index


# Cum return before fees
backtest_df = cum_return(backtest_df)

backtest_df = cum_return_after_fees(backtest_df)
print(backtest_df['perfect_cum_return'][-1],backtest_df['perfect_cum_return_w_fees'][-1])
print(backtest_df.tail(50))

### random
backtest_df['cum_return_market']= (1+backtest_df['hourly_return']).cumprod()

plt.plot(backtest_df['perfect_cum_return_w_fees'], label = 'Cumulative return after fees')
plt.plot(backtest_df['perfect_cum_return'], label = 'Cumulative return before fees')
plt.plot(backtest_df['cum_return_market'], label = 'Market return (no fees)')
plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
plt.title('Cumulative return before and after fees')
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.legend()
#plt.savefig('cum_return.png', dpi = 300)
plt.show()

plot_importance(model)
plt.title('Feature importance')
#plt.savefig('feature_importance.png', dpi = 300)
plt.show()


#                   TESTING ON NEWEST DATA - unseen
##############################################################

predict_january = X_xg.loc['2026-01-26 20:00:00+00:00':].copy()
probs_01 = model.predict_proba(predict_january)[:,1]
pred_01 = (probs_01>0.65).astype(int)
probs_01.mean()
pd.DataFrame(pred_01).value_counts(normalize = True)

#Calculation of cumulative return
y_pred01_series = pd.DataFrame(pd.Series(pred_01, index=predict_january.index)).rename(columns = {0: 'target'})
y_pred01_series['hourly_return'] = X_prep['hourly_return']
y_pred01_series['P_t'] = X_prep['P_t']

y_pred01_series['stable_position']= create_stable_position(y_pred01_series, tp=0.04, sl=sl, horizon=24)
y_pred01_series = cum_return(y_pred01_series)
y_pred01_series = cum_return_after_fees(y_pred01_series)
print(y_pred01_series['perfect_cum_return'][-1], y_pred01_series['perfect_cum_return_w_fees'][-1])

plt.plot(y_pred01_series['perfect_cum_return_w_fees'], label = "Cumulative return after fees" )
plt.plot(y_pred01_series['perfect_cum_return'], label = "Cumulative return before fees" )
plt.plot((1+y_pred01_series['hourly_return']).cumprod(), label = "Market return")
plt.title('Cumulative return - February 2026 fall')
#plt.savefig('cum_return - 02.2026.png', dpi = 300)
plt.show()
