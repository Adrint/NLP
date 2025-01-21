from datasets import load_dataset
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

'''This program processes a dataset to classify whether a given statement is legal or illegal.
It includes training and test datasets. The program cleans the data and trains models
using Logistic Regression, Naive Bayes, Random Forest, and Decision Tree algorithms.'''

def load_polish_stopwords(filepath):
    '''Loads Polish stopwords from a text file.'''
    with open(filepath, "r", encoding="utf-8") as file:
        polish_stopwords = file.read().splitlines()
    return polish_stopwords

def load_data():
    '''Loads the dataset and splits it into training and test sets.'''
    ds = load_dataset("laugustyniak/abusive-clauses-pl")
    print(f"Dataset: \n{ds}\n")

    # Split into training and test data
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas()

    # Check for missing values
    print("Are there any missing values in the data:")
    print(df_train["text"].isnull().any() or df_train["label"].isnull().any() or
          df_test["text"].isnull().any() or df_test["label"].isnull().any())

    # Calculate the number of legal and illegal examples
    count_legal = (df_train["label"] == 1).sum()
    count_illegal = (df_train["label"] == 0).sum()
    print("Number of legal and illegal examples:\n")
    print(f"Legal: {count_legal}\nIllegal: {count_illegal}\n")

    # Calculate the percentage of legal and illegal examples
    legal_percent = round(count_legal / len(df_train) * 100)
    illegal_percent = round(count_illegal / len(df_train) * 100)
    print("Percentage of legal and illegal examples:\n")
    print(f"Legal: {legal_percent}%\nIllegal: {illegal_percent}%\n")

    return df_train, df_test

def cleaning_data(df_train, df_test):
    '''Cleans text data by removing punctuation, converting to lowercase, and removing numbers.'''
    print(f"Sample original text:\n{df_train['text'].iloc[1]}")

    # Normalize text in the training set
    df_train['text'] = df_train['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    df_train['text'] = df_train['text'].str.lower()
    df_train['text'] = df_train['text'].apply(lambda x: re.sub('\\w*\\d\\w*', ' ', x))

    # Normalize text in the test set
    df_test['text'] = df_test['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    df_test['text'] = df_test['text'].str.lower()
    df_test['text'] = df_test['text'].apply(lambda x: re.sub('\\w*\\d\\w*', ' ', x))

    print(f"Sample cleaned text:\n{df_train['text'].iloc[1]}\n\n")

    return df_train, df_test

def prepare_text_data(df_train, df_test, polish_stopwords):
    '''Processes text data into a word occurrence matrix.'''
    x_df_train = df_train['text']
    x_df_test = df_test['text']

    # Convert text into numerical representation using CountVectorizer
    cv = CountVectorizer(stop_words=list(polish_stopwords))
    x_df_train_cv = cv.fit_transform(x_df_train)
    x_df_test_cv = cv.transform(x_df_test)

    return x_df_train_cv, x_df_test_cv

def train_logistic_regression(x_train, y_train, x_test):
    '''Trains a logistic regression model and makes predictions on the test set.'''
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred

def train_naive_bayes(x_train, y_train, x_test):
    '''Trains a Naive Bayes model and makes predictions on the test set.'''
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    return y_pred

def train_random_forest(x_train, y_train, x_test):
    '''Trains a Random Forest model and makes predictions on the test set.'''
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return y_pred

def train_decision_tree(x_train, y_train, x_test):
    '''Trains a Decision Tree model and makes predictions on the test set.'''
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    return y_pred

def evaluate_model(y_true, y_pred):
    '''Evaluates a model based on metrics such as accuracy, precision, recall, and F1-score.'''
    cm = confusion_matrix(y_true, y_pred)
    true_neg, false_pos = cm[0, 0], cm[0, 1]
    false_neg, true_pos = cm[1, 0], cm[1, 1]

    accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
    precision = round(true_pos / (true_pos + false_pos), 3)
    recall = round(true_pos / (true_pos + false_neg), 3)
    f1 = round(2 * (precision * recall) / (precision + recall), 3)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Illegal', 'Predicted Legal'], yticklabels=['Actual Illegal', 'Actual Legal'])
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def main():
    '''Main function of the program.'''
    stopwords_path = "polish_stopwords.txt"
    polish_stopwords = load_polish_stopwords(stopwords_path)

    df_train, df_test = load_data()
    df_train, df_test = cleaning_data(df_train, df_test)

    y_train = df_train['label']
    y_test = df_test['label']

    x_train_cv, x_test_cv = prepare_text_data(df_train, df_test, polish_stopwords)

    # Train and predict using different models
    print("Logistic Regression:")
    y_pred_lr = train_logistic_regression(x_train_cv, y_train, x_test_cv)
    evaluate_model(y_test, y_pred_lr)

    print("\nNaive Bayes:")
    y_pred_nb = train_naive_bayes(x_train_cv, y_train, x_test_cv)
    evaluate_model(y_test, y_pred_nb)

    print("\nRandom Forest:")
    y_pred_rf = train_random_forest(x_train_cv, y_train, x_test_cv)
    evaluate_model(y_test, y_pred_rf)

    print("\nDecision Tree:")
    y_pred_dt = train_decision_tree(x_train_cv, y_train, x_test_cv)
    evaluate_model(y_test, y_pred_dt)

if __name__ == "__main__":
    main()
