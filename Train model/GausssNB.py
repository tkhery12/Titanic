import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB

def train_model_GaussNB(train_data_file: str, output_dir: str):

    print("Start processing.")

    df = pd.read_csv(train_data_file)

    _preprocess_dataframe(df, output_dir)

    _train_log_randomfores(df)
    print("Model trained.")

def _preprocess_dataframe(df: pd.DataFrame, output_dir: str):
    print("Preprocessing DataFrame...")

    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    _preprocess_categorical_data(df, output_dir)
    _impute_data(df)

def _preprocess_categorical_data(df: pd.DataFrame, output_dir: str):
    # pick column conatain text : Sex and Embarked
    for obj_col in df.select_dtypes('object'):
        df[obj_col] = df[obj_col].astype('category')

    # mode lọc các chế độ thường xuất hiện nhất
    for cat_col in df.select_dtypes('category'):
        df[cat_col].fillna(df[cat_col].mode().iloc[0], inplace=True)
        # print(df[cat_col])

    label_encoder_dict = defaultdict(LabelEncoder)

    for cat_col in df.select_dtypes('category'):
        # text to number
        df[cat_col] = label_encoder_dict[cat_col].fit_transform(df[cat_col])
        # print(df[cat_col])

    _save_label_encoders(label_encoder_dict, output_dir)


def _save_label_encoders(label_encoder_dict: defaultdict, output_dir: str):
    # Save encoders in output_dir/encoders folder
    for label, encoder in label_encoder_dict.items():
        with open(os.path.join(output_dir, 'encoders', "f{label}.json"), 'w') as f:
            json.dump(list(encoder.classes_), f)


# fill  by mean of value
def _impute_data(df: pd.DataFrame):
    for float_col in df.select_dtypes('float64'):
        df[float_col].fillna(df[float_col].mean(), inplace=True)
        # print(df[float_col])

    for col in df.columns:
        df[col].fillna(df[col].mode().iloc[0], inplace=True)
        #print(df[col])

def preprocessingtest(dftest:pd.DataFrame):

    dftest.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    for obj_col in dftest.select_dtypes('object'):
        dftest[obj_col] = dftest[obj_col].astype('category')

        # mode lọc các chế độ thường xuất hiện nhất
    for cat_col in dftest.select_dtypes('category'):
        dftest[cat_col].fillna(dftest[cat_col].mode().iloc[0], inplace=True)
        # print(df[cat_col])

    label_encoder_dict = defaultdict(LabelEncoder)

    for cat_col in dftest.select_dtypes('category'):
        # text to number
        dftest[cat_col] = label_encoder_dict[cat_col].fit_transform(dftest[cat_col])
        # print(df[cat_col])

    for float_col in dftest.select_dtypes('float64'):
        dftest[float_col].fillna(dftest[float_col].mean(), inplace=True)
        # print(df[float_col])

    for col in dftest.columns:
        dftest[col].fillna(dftest[col].mode().iloc[0], inplace=True)
        #print(df[col])

def _train_log_randomfores(df: pd.DataFrame):

    X = df.drop(columns=['Survived'])
    y = df['Survived']

    clf = GaussianNB()

    dftest = pd.read_csv(test_pat)
    preprocessingtest(dftest)
    clf.fit(X,y)
    #print('\n classification report:\n', classification_report(X['Survived'], y))
    data = pd.read_csv("../data/test.csv")
    data.drop(columns=['Name', 'Ticket', 'Cabin','Pclass','Age','Sex','Fare','Embarked','SibSp','Parch','Ticket'], inplace=True)
    data['Survived'] = list(clf.predict(dftest))
    print(data)
    data.to_csv(r'..\Data\test\export_Gauss.csv', index=False,header=True)
    #print(list(search.predict(dftest)))

if __name__ == '__main__':
    train_path = "../data/train.csv"
    output_path = "../data/model"
    test_pat = "../data/test.csv"
    outtest_path = "..data/test/export_data.csv"
    train_model_GaussNB(train_path, output_path)