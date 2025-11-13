import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_and_preprocess(file_path):
    """
    TSVファイルを読み込み、テキストとラベルに分割する関数
    """
    # データの読み込み
    df = pd.read_csv(file_path, sep="\t")
    # テキストデータ(X)とラベル(y)の取得
    texts = df["sentence"].values
    labels = df["label"].values
    
    return texts, labels

def main():
    # 1. データ準備
    file_path = "train.tsv"
    print(f"Loading data from {file_path}...")

    try:
        texts, labels = load_and_preprocess(file_path)
    except FileNotFoundError:
        print("Error: train.tsv not found. Please place the dataset in the same directory.")
        return

    # 2. データ分割 (学習用:テスト用 = 8:2)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 3. ベクトル化 (Bag of Words)
    vectorizer = CountVectorizer()
    
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 4. モデル学習 (ロジスティック回帰)
    model = LogisticRegression(max_iter=1000)
    print("Training model...")
    model.fit(x_train_vec, y_train)

    # 5. 評価
    y_pred = model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()