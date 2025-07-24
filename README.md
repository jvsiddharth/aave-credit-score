# Aave V2 DeFi Credit Scorer

## 📌 Problem

Build a machine learning system to generate credit scores (0–1000) for wallets on the Aave V2 protocol using historical transaction behavior only.

## 🔍 Dataset

- Format: JSON (~87 MB unzipped)
- Records: ~100,000+ transactions
- Link: [user-transactions.json](https://drive.google.com/file/d/1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS/view?usp=sharing)

## ⚙️ Project Structure

```bash
project/
│
├── score_generator.ipynb     # Jupyter notebook containing all code and logic
├── wallet_scores.csv         # Output CSV with wallet address and score
├── analysis.md               # Summary of methods and insights
└── README.md                 # This file
