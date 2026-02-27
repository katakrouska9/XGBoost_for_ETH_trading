# Ethereum Algorithmic Trading: XGBoost Strategy with Transaction Costs

This project implements a short-term prediction model for Ethereum (ETH-USD) using the **XGBoost** classification algorithm. The main goal was to build a robust trading algorithm capable of performing in real market conditions, accounting for transaction fees, slippage, and regime changes.

## Target & strategy definition

The model is trained on an hourly dataset of the last 730 days and predicts whether it is an optimal time to enter a long position.

* **Labeling:** A binary target variable was engineered using a forward-looking return calculation. It equals 1 if the future return (in a 48h time window) reaches a specific threshold (e.g., >=4%) and 0 otherwise.

* **Methodology:** Model was defined using max_depth of 3 to avoid overfitting to noise in hourly ETH data. The model was evaluated using two training regimes: a static Full Training Set and an Expanding Window (Walk-forward).

* **Execution:** Once a signal is triggered, the strategy manages the position using a triple-barrier exit logic: Take-Profit, Stop-Loss, or Time-Horizon expiration—whichever occurs first.

## 📈 Performance & Risk mitigation

* **Training regimes selection:** Since cryptocurrency return prediction demands substantial data to produce even directionally reliable signals, the expanding window model was found to be less suitable. Model with the simple train/test split (70/30) performed more consistently and therefore was selected for backtesting. 

* **Train vs. Test :**  Model achieves f1-score of ±0.52 on the training set period, reaching ±650% cumulative return (before fees), largely supported by strict stop-loss limits. Test set f1-score is ±0.36. Both statistics suggest the presence of weak but non-random signal in the data.

* **Gross vs. Net Cumulative returns on Test set:** Before fees cumulative return on the test period reaches 86% compared to market return of 52%. By implementing a realistic **0.15% fee per trade** (accounting for slippage), I calculate the cum. return after fees. The strategy performs better than the market, when it achieves a net return of approximately **-30%** after all costs (July 2025 – Jan 2026) vs. **-50%** of the market's buy&hold.

![Cumulative Returns](cum_return.png)

* **Regime dependence:** The model delivered high performance untill November 2025, when market price was growing or declining gradually. However, model underperforms in the period of steep price decline.


## 🛠️ Feature Engineering & Logic
The model's predictive power is derived from a mix of momentum, volatility, and cross-asset correlation features.

![Feature Importance](feature_importance.png)

* **Distance to SMA200/50:** Identified as the most significant features, indicating a reliance on mean-reversion signals.
* **ATR_rel (Volatility):** Helps the model adjust risk thresholds based on current market noise.
* **BTC Gap:** While Bitcoin is often a leading indicator, its relative impact on hourly Ethereum data was found not consequential, likely because arbitrage between the two assets occurs on even lower timeframes, while fundamental shifts of capital happen on longer scale (daily).

## 🚀 Future work to be done
* **Optimilizing training data:** Model should be trained on data that contain periods representing both bull and bear markets to be better protected against periods of extreme price movements. 
* **Flexible probability thresholding:** Instead of a fixed P > 0.50, a dynamic threshold could be implemented. In a strong uptrend (e.g., price > SMA200), keep the threshold to catch the momentum. In a downtrend, increase it to 0.60 or higher to be more defensive.
* **Addition of new explanatory variables:** Add more variables to better identify bull and bear runs, such as volume profiles or ETH funding rates.

## Caution
While the backtest includes a 0.15% fee, real-world execution involves variable slippage and potentially network gas fees that can impact net profitability on lower timeframes.