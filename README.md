# Aave V2 Wallet Credit Scoring

## 📌 Problem Statement


Methodology

Data Collection:
Loads transaction data from user_transactions.json.
Filters transactions to include only wallets listed in wallet_addresses.csv.
Converts transaction amounts to USD values, handling token decimals.

Feature Extraction:
Computes wallet-level features: number of transactions, active days, average transaction USD value, total deposits, borrows, repays, liquidation count, borrow-to-deposit ratio, repay ratio, and unique assets used.

Risk Scoring:
Calculates a raw score using a weighted combination of features (repay ratio: 35%, borrow-to-deposit ratio: 25%, liquidation count: 20%, transaction frequency: 15%, asset diversification: 5%).
Scales raw scores to a 0-1000 range using MinMaxScaler.
