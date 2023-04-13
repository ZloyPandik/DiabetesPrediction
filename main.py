import numpy as np
import pandas as pd
import missingno as msno
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('diabetes.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option("display.float_format",lambda x: "%.5f" % x)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

df = pd.read_csv("diabetes.csv")
df.head()

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    df[col].replace(0, np.NaN, inplace=True)



msno.bar(df);



msno.heatmap(df);

for col in df.columns:
    df.loc[(df["Outcome"] == 0) & (df[col].isnull()), col] = df[df["Outcome"] == 0][col].median()
    df.loc[(df["Outcome"] == 1) & (df[col].isnull()), col] = df[df["Outcome"] == 1][col].median()


df.hist(figsize = (15,7));
def outlier_thresholds(dataframe, col_name, th1=0.05, th3=0.95):
    quartile1 = dataframe[col_name].quantile(th1)
    quartile3 = dataframe[col_name].quantile(th3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name, th1=0.05, th3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, th1, th3)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
num_cols = [col for col in df.columns if df[col].dtypes in [int, float]
            and df[col].nunique() > 10]
for col in df.columns:
    print(check_outlier(df, col))
for col in df.columns:
    replace_with_thresholds(df, col)
for col in df.columns:
    print(check_outlier(df, col))

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
df['NEW_BMI_CAT'] = pd.cut(x=df['BMI'], bins=[0, 18.4, 25.0, 30.0, 70.0],
                           labels=['weakness', 'normal', 'slightly_fat', 'obese']).astype('O')

# New categorical Glucose
df['NEW_GLUCOSE_CAT'] = pd.cut(x=df['Glucose'], bins=[0, 139, 200],
                               labels=['Normal', 'Prediabetes']).astype('O')

#  New categorical BloodPressure
df['NEW_BLOOD_CAT'] = pd.cut(x=df['BloodPressure'], bins=[0, 79, 90, 123],
                             labels=['Normal', 'Hypertension_S1', 'Hypertension_S2']).astype('O')

# New categorical SkinThickness
df['NEW_SKINTHICKNESS_CAT'] = df['SkinThickness'].apply(lambda x: 1 if x <= 18.0 else 0)

# New categorical Insulin
df['NEW_INSULIN_CAT'] = df['Insulin'].apply(lambda x: 'Normal' if 16.0 <= x <=166   else 'Abnormal')
label_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() <= 2]
for col in label_cols:
    label_encoder(df, col)
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.columns = [col.upper() for col in df.columns]

y = df[['OUTCOME']]
X = df.drop('OUTCOME', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

rf = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc_random_forest = round(rf.score(X_test, y_pred) * 100, 2)
acc_random_forest
