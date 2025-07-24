# Credit Scoring Analysis — Aave V2 Wallets

## Objective

The objective of this project is to assign a credit score (0–1000) to each wallet address based on historical DeFi behavior, particularly on the Aave V2 protocol. This score quantifies the trustworthiness of a wallet based on its interactions with the protocol — distinguishing responsible users from exploitative actors or bots.

---

## Methodology

### 1. **Data Ingestion**
We ingest a raw JSON file consisting of transaction records per wallet, where each transaction can be a `deposit`, `borrow`, `repay`, `redeemunderlying`, or `liquidationcall`.

### 2. **Feature Engineering**
Key features were engineered from the transaction data:
- **Transaction Count**: Total number of transactions.
- **Action Distribution**: Proportion of each action type (`deposit`, `borrow`, etc.).
- **Recency & Frequency**: Time between first and last transaction, and overall transaction frequency.
- **Borrow/Repay Ratio**: Indicates responsible borrowing.
- **Liquidation Frequency**: High values penalized.
- **Bot-like Behavior**: Very high or uniform frequency penalized.
- **Loan Cycles**: Pattern of borrowing and repaying fully suggests reliability.
- **Collateral Usage**: Wallets that deposit before borrowing are favored.
- **Volatility Score**: Measures rapid shifts in behavior, indicating instability.

### 3. **Scoring Mechanism**
Each feature contributes to a weighted formula that outputs a credit score between 0 and 1 (later scaled to 0–1000):
