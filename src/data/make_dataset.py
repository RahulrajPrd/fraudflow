import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("data/raw/creditcard.csv")
    
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Class"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["Class"])
    
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    main()
