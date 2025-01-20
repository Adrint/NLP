from datasets import load_dataset
import pandas as pd


def load_data():
    # Loading dataset
    ds = load_dataset("laugustyniak/abusive-clauses-pl")
    print(f"Dataset: \n{ds}\n")

    # Dividing dataset into the 3 dataframes
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas()
    df_validation = ds['validation'].to_pandas()

    # Displaying sample row from dataset
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    #print(df_train.head(1))

    # Checking for any null values in dataset
    print("Checking for any null values in dataset:")
    print(df_train["text"].isnull().any() or df_train["label"].isnull().any() or df_test["text"].isnull().any() or df_test["label"].isnull().any() or df_validation["text"].isnull().any() or df_validation["label"].isnull().any())

    # Calculating number of legal and illegal statuses in the dataset
    count_legal = (df_train["label"] == 1).sum()
    count_illegal = (df_train["label"] == 0).sum()
    print("Number of legal and illegal statuses:\n")
    print(f"Legal values: {count_legal}\nIllegal values: {count_illegal}\n")

    # Calculating percent of legal and illegal statuses in the dataset
    legal_percent = round(count_legal/len(df_train)*100)
    illegal_percent = round(count_illegal/len(df_train)*100)
    print("Percent of legal and illegal statuses:\n")
    print(f"Legal values: {legal_percent}%\nIllegal values: {illegal_percent}%\n")

def data_processing():
    

def main():
    load_data()
    data_processing()

if __name__ == "__main__":
    main()

