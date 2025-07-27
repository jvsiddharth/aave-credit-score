import json
import pandas as pd
from decimal import Decimal
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
def load_data(json_path, wallet_csv_path):
    print(f"Loading JSON data from: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(json_path, "r") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")

    if not raw_data:
        raise ValueError(f"JSON file {json_path} is empty")

    print(f"Loading wallet addresses from: {wallet_csv_path}")
    if not os.path.exists(wallet_csv_path):
        raise FileNotFoundError(f"CSV file not found: {wallet_csv_path}")

    try:
        wallet_df = pd.read_csv(wallet_csv_path)
        if "wallet_id" not in wallet_df.columns:
            raise ValueError(f"CSV file {wallet_csv_path} must contain a 'wallet_id' column")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file {wallet_csv_path} is empty")

    valid_wallets = set(wallet_df["wallet_id"].str.lower())
    print(f"Loaded {len(valid_wallets)} wallet addresses from CSV")
    print(f"Sample wallet addresses: {list(valid_wallets)[:5]}")

    records = []
    json_wallets = set()
    for tx in tqdm(raw_data, desc="Processing transactions"):
        wallet = tx.get("userWallet", "").lower()
        json_wallets.add(wallet)
        if not wallet or wallet not in valid_wallets:
            continue

        if not all(key in tx for key in ["timestamp", "action", "actionData"]):
            print(f"Warning: Skipping transaction with missing fields: {tx}")
            continue

        ts = pd.to_datetime(tx.get("timestamp"), unit="s", errors="coerce")
        if pd.isna(ts):
            print(f"Warning: Invalid timestamp in transaction: {tx}")
            continue

        action = tx.get("action", "").lower()  # Expect Compound V2/V3 actions
        action_data = tx.get("actionData", {})
        asset = action_data.get("assetSymbol", "")

        try:
            amount = Decimal(action_data.get("amount", "0"))
            price = Decimal(action_data.get("assetPriceUSD", "0"))
            usd_value = float(amount * price) / 1e18 if amount > 1e12 else float(amount * price)
        except (ValueError, TypeError):
            print(f"Warning: Invalid amount or price in transaction: {tx}")
            usd_value = 0.0

        records.append({
            "wallet": wallet,
            "timestamp": ts,
            "action": action,
            "usd_value": usd_value,
            "asset": asset
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("Warning: No valid transactions found for provided wallets.")
        print(f"Wallets in JSON: {len(json_wallets)}")
        print(f"Sample wallets in JSON: {list(json_wallets)[:5]}")
        print(f"Wallets in CSV not found in JSON: {valid_wallets - json_wallets}")
    else:
        print(f"Processed {len(df)} transactions for {df['wallet'].nunique()} wallets")
        df.sort_values(by=["wallet", "timestamp"], inplace=True)

    return df, valid_wallets

def extract_features(df):
    print("Generating wallet-level features...")
    if df.empty:
        return pd.DataFrame()

    wallet_groups = df.groupby("wallet")
    features = []

    for wallet, group in tqdm(wallet_groups, desc="Extracting features"):
        f = {"wallet": wallet}
        f["num_tx"] = len(group)
        f["active_days"] = (group["timestamp"].max() - group["timestamp"].min()).days + 1
        f["avg_tx_usd"] = group["usd_value"].mean()

        deposits = group[group["action"].isin(["supply", "mint"])]
        borrows = group[group["action"] == "borrow"]
        repays = group[group["action"] == "repay"]
        liquidations = group[group["action"].isin(["liquidationcall", "liquidateborrow"])]

        f["deposit_total"] = deposits["usd_value"].sum()
        f["borrow_total"] = borrows["usd_value"].sum()
        f["repay_total"] = repays["usd_value"].sum()
        f["liquidation_count"] = len(liquidations)

        f["borrow_deposit_ratio"] = f["borrow_total"] / f["deposit_total"] if f["deposit_total"] > 0 else 0
        f["repay_ratio"] = f["repay_total"] / f["borrow_total"] if f["borrow_total"] > 0 else 0
        f["unique_assets"] = group["asset"].nunique()

        features.append(f)

    features_df = pd.DataFrame(features)
    features_df.fillna(0, inplace=True)
    return features_df

def score_wallets(features_df, valid_wallets):
    print("Scoring wallets...")
    if features_df.empty:
        print("No features extracted. Generating zero scores for all wallets.")
        return pd.DataFrame({"wallet_id": list(valid_wallets), "score": [0.0] * len(valid_wallets)})

    features_df["score_raw"] = (
        features_df["repay_ratio"] * 0.35 +
        (1 - features_df["borrow_deposit_ratio"]) * 0.25 +
        (1 / (1 + features_df["liquidation_count"])) * 0.2 +
        (features_df["num_tx"] / features_df["active_days"].replace(0, 1)) * 0.15 +
        (features_df["unique_assets"] / features_df["unique_assets"].max()) * 0.05
    )

    scaler = MinMaxScaler(feature_range=(0, 1000))
    features_df["credit_score"] = scaler.fit_transform(features_df[["score_raw"]]).round(2)

    all_wallets = pd.DataFrame({"wallet_id": list(valid_wallets)})
    scores_df = all_wallets.merge(
        features_df[["wallet", "credit_score"]].rename(columns={"wallet": "wallet_id", "credit_score": "score"}),
        on="wallet_id",
        how="left"
    )
    scores_df["score"].fillna(0.0, inplace=True)

    return scores_df[["wallet_id", "score"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to user-wallet-transactions.json")
    parser.add_argument("--wallets", required=True, help="Path to wallets.csv")
    parser.add_argument("--output", default="wallet_scores.csv", help="Path to save wallet scores")
    args = parser.parse_args()

    try:
        df, valid_wallets = load_data(args.input, args.wallets)
        features_df = extract_features(df)
        scores_df = score_wallets(features_df, valid_wallets)

        print(f"Saving scores to {args.output}")
        scores_df.to_csv(args.output, index=False)

        print("✅ Scoring complete!")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
