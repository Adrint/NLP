from datasets import load_dataset
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

def load_polish_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        polish_stopwords = set(file.read().splitlines())
    return polish_stopwords

def load_data():
    # Loading dataset
    ds = load_dataset("laugustyniak/abusive-clauses-pl")
    print(f"Dataset: \n{ds}\n")

    # Dividing dataset into the 3 dataframes
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas()
    df_validation = ds['validation'].to_pandas()

    # Checking for any null values in dataset
    print("Checking for any null values in dataset:")
    print(df_train["text"].isnull().any() or df_train["label"].isnull().any() or
          df_test["text"].isnull().any() or df_test["label"].isnull().any() or
          df_validation["text"].isnull().any() or df_validation["label"].isnull().any())

    # Calculating number of legal and illegal statuses in the dataset
    count_legal = (df_train["label"] == 1).sum()
    count_illegal = (df_train["label"] == 0).sum()
    print("Number of legal and illegal statuses:\n")
    print(f"Legal values: {count_legal}\nIllegal values: {count_illegal}\n")

    # Calculating percent of legal and illegal statuses in the dataset
    legal_percent = round(count_legal / len(df_train) * 100)
    illegal_percent = round(count_illegal / len(df_train) * 100)
    print("Percent of legal and illegal statuses:\n")
    print(f"Legal values: {legal_percent}%\nIllegal values: {illegal_percent}%\n")

    return df_train, df_test, df_validation

def cleaning_data(df_train, df_test, df_validation):
    polish_stopwords = load_polish_stopwords(stopwords_path)
    print(f"Original text:\n{df_train['text'].iloc[1]}")

    # Normalizing text in train section
    df_train['text'] = df_train['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    df_train['text'] = df_train['text'].str.lower()
    df_train['text'] = df_train['text'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))

    # Normalizing text in test section
    df_test['text'] = df_test['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    df_test['text'] = df_test['text'].str.lower()
    df_test['text'] = df_test['text'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))

    # Normalizing text in validation section
    df_validation['text'] = df_validation['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    df_validation['text'] = df_validation['text'].str.lower()
    df_validation['text'] = df_validation['text'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))
    #df_validation['text'] = df_validation['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in polish_stopwords))

    print(f"Modified text:\n{df_train['text'].iloc[1]}\n\nMachine learning:")

    return df_train,df_test,df_validation

def machine_learning(df_train,df_test,df_validation):
    x_df_train = df_train['text']
    y_df_train = df_train['label']

def main():
    df_train, df_test, df_validation = load_data()
    stopwords_path = "polish_stopwords.txt"  # Ścieżka do pliku z polskimi stopwordami
    cleaning_data(df_train, df_test, df_validation, stopwords_path)



if __name__ == "__main__":
    main()
