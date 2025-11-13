# Text Classification with Scikit-learn
(ロジスティック回帰によるテキスト分類)

## 📖 概要
Bag-of-Words (BoW) アプローチとロジスティック回帰を用いて、テキストデータを2値分類するコードを実装しました。
自然言語処理（NLP）における基本的なデータ前処理、ベクトル化、モデル学習、評価の一連の流れを構築しました。

## 🛠 使用技術
* Python 3.10.9
* Pandas (データ前処理)
* Scikit-learn (CountVectorizer, LogisticRegression)

## 🤖 開発スタイル (Development with AI)
本プロジェクトは、**生成AI (Google Gemini)** を技術的な相談役（ペアプログラマー）として活用し、対話的に実装を進めました。

* **活用プロセス:**
    * 実装方針の策定と、ライブラリ（Scikit-learn, Pandas）の最適な使用方法の選定。
    * コードの可読性を高めるためのリファクタリングと、型定義の確認。
    * エラー発生時の原因切り分けとデバッグ。

## 🚀 実行方法
## 📊 データセットと出典 (Dataset & Credits)
本プロジェクトでは、以下のデータセットを利用して学習を行いました。

### 1. データセットの出典とリンク
* **データセット名:** Stanford Sentiment Treebank (SST-2)
* **起源サイト:** [Stanford NLP Group](https://nlp.stanford.edu/sentiment/)
* **ベンチマーク定義元:** [GLUE Benchmark](https://gluebenchmark.com/)

### 2. ダウンロードリンクと倫理的配慮
著作権を尊重し、データセット本体はGitHubに含めておりません。再現を行う場合は、以下の公式リンクからデータを取得してください。

* **直接ダウンロードURL (FAIR Server):**
    [SST-2.zip](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)

### 3. 参考文献
* **R. Socher et al.,** *"Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank"*, Proceedings of the 2013 Conference on Empirical Methods for Natural Language Processing (**EMNLP 2013**).
* **A. Wang et al.,** *"GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding"*, **Published as a conference paper at ICLR 2019**.


2. **ライブラリのインストール:**
   ```bash
   pip install -r requirements.txt
   ```

3. **実行:**
   ```bash
   python main.py

   ```
