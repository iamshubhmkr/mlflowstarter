import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_versions():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    # Clean
    df = df.drop(["Cabin", "Name", "Ticket"], axis=1)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    os.makedirs("data", exist_ok=True)

    # Version 1.1: Only numerical + Sex
    v1_1 = df[["Pclass", "Age", "SibSp", "Fare", "Sex_male", "Survived"]]
    v1_1.to_csv("data/v1_1_train.csv", index=False)

    # Version 1.2: All except PassengerId
    v1_2 = df.drop("PassengerId", axis=1)
    v1_2.to_csv("data/v1_2_train.csv", index=False)

    # Version 1.3: Use 70% random subset of v1_2
    v1_3 = v1_2.sample(frac=0.7, random_state=42)
    v1_3.to_csv("data/v1_3_train.csv", index=False)

if __name__ == "__main__":
    prepare_versions()
    print("✅ Titanic data versions prepared.")
