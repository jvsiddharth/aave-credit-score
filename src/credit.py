import json
import pandas as pd
from decimal import Decimal
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import argparse

def load_data(json_path):
    print(f"Loading JSON data from: {json_path}")
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    records = []
    for tx in tqdm(raw_data):
        wallet = tx.get("userWallet")
        ts = pd.to_datetime(tx.get("timestamp"), unit="s")
        action = tx.get("action").lower()
        action_data = tx.get("actionData", {})
        asset = action_data.get("assetSymbol")

        try:
            amount = Decimal(action_data.get("amount", "0"))
            price = Decimal(action_data.get("assetPriceUSD", "0"))
            usd_value = float(amount * price) / 1e18 if amount > 1e12 else float(amount * price)
        except:
            usd_value = 0.0

        records.append({
            "wallet": wallet,
            "timestamp": ts,
            "action": action,
            "usd_value": usd_value,
            "asset": asset
        })

    df = pd.DataFrame(records)
    df.sort_values(by=["wallet", "timestamp"], inplace=True)
    return df

def extract_features(df):
    print("Generating wallet-level features...")
    wallet_groups = df.groupby("wallet")
    features = []

    for wallet, group in tqdm(wallet_groups):
        f = {"wallet": wallet}
        f["num_tx"] = len(group)
        f["active_days"] = (group["timestamp"].max() - group["timestamp"].min()).days + 1
        f["avg_tx_usd"] = group["usd_value"].mean()

        deposits = group[group["action"] == "deposit"]
        borrows = group[group["action"] == "borrow"]
        repays = group[group["action"] == "repay"]
        liquidations = group[group["action"] == "liquidationcall"]

        f["deposit_total"] = deposits["usd_value"].sum()
        f["borrow_total"] = borrows["usd_value"].sum()
        f["repay_total"] = repays["usd_value"].sum()
        f["liquidation_count"] = len(liquidations)

        f["borrow_deposit_ratio"] = f["borrow_total"] / f["deposit_total"] if f["deposit_total"] > 0 else 0
        f["repay_ratio"] = f["repay_total"] / f["borrow_total"] if f["borrow_total"] > 0 else 0

        features.append(f)

    features_df = pd.DataFrame(features)
    features_df.fillna(0, inplace=True)
    return features_df

def score_wallets(features_df):
    print("Scoring wallets...")

    # Basic heuristic scoring logic
    features_df["score_raw"] = (
        features_df["repay_ratio"] * 0.4 +
        (1 - features_df["borrow_deposit_ratio"]) * 0.2 +
        (1 / (1 + features_df["liquidation_count"])) * 0.2 +
        (features_df["num_tx"] / features_df["active_days"].replace(0, 1)) * 0.2
    )

    scaler = MinMaxScaler(feature_range=(0, 1000))
    features_df["credit_score"] = scaler.fit_transform(features_df[["score_raw"]])

    return features_df[["wallet", "credit_score"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to user_transactions.json")
    parser.add_argument("--output", default="wallet_scores.csv", help="Path to save wallet scores")
    args = parser.parse_args()

    df = load_data(args.input)
    features_df = extract_features(df)
    scores_df = score_wallets(features_df)

    print(f"Saving scores to {args.output}")
    scores_df.to_csv(args.output, index=False)

    print("✅ Scoring complete!")

if __name__ == "__main__":
    main()
