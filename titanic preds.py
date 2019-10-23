# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:20:17 2019

@author: pgood
"""

from kaggle.api.kaggle_api_extended import KaggleApi
from nose.tools import assert_equal
import os
import zipfile
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
import numpy as np

"""
while True:
    
    username = input('Enter your username: ')
    key = input('Enter your key: ')
    
    kaggle_key = {'username': username, 'key': key}
    
    api = KaggleApi(kaggle_key)
    api.authenticate()
    break
"""


api = KaggleApi()
api.authenticate()
api.competition_download_files("titanic", 'titanic data')


print('Beginning file download with wget module')

def shape_check(table, shape, threshold, data_name):
    import pandas as pd
    
    cells = shape[0]*shape[1] 
    if type(table) == pd.core.frame.DataFrame:
        
        assert_equal(table.shape, shape)
        assert(sum(table.count())/cells > threshold)
        print("{} loaded correctly".format(data_name))
        
    elif type(table) == np.ndarray:
        
        assert_equal(table.shape, shape)
        assert(np.count_nonzero(np.isnan(table))/cells < (1 - threshold))
        print("{} loaded correctly".format(data_name))

        
with zipfile.ZipFile('titanic data/titanic.zip', 'r') as zip_ref:
    zip_ref.extractall('titanic data/titanic files')

root = 'titanic data/titanic files'
dfs = []
for root, dirs, files in os.walk(root):
    for file in files:
        if '.csv' in file:
            dfs.append(pd.read_csv(os.path.join(root, file)))

train_raw = dfs[2]
#shape_check(train_raw, (891,12), .8, 'Train')
holdout_raw = dfs[1]
#shape_check(holdout_raw, (418,11), .8, 'Test')

class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        # what other output you want
        self.X = X
        print(X)
        pd.DataFrame(X).to_csv('data.csv')
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class ColumnDropper(TransformerMixin):
    
    def __init__ (self, cols, convert = False):
        self.cols = cols
        self.convert = convert
        
    def transform(self, df):
        if self.convert == False:
            return df.drop(columns = self.cols)
        else:
            return df.drop(columns = self.cols).to_numpy()
    
    def fit(self, *_):
        return self


def count_cabins(col):
    
    try:
        return len(col.split())
    except:
        return 0
    
def cabin_preprocess(df):
    
    df['cabin_count'] = df.Cabin.map(count_cabins)
    df.Cabin = df.Cabin.fillna('u')
    df['cabin_type'] = df.Cabin.map(lambda x: x[0])
    return df.drop(columns = 'Cabin')

def name_flags(col):
    
    try:
        if 'Miss.' in col:
            return 'miss'
        elif 'Mrs.' in col:
            return 'mrs'
        else:
            return 'un'
        
    except:
        return 'un'

def name_preprocess(df):
    
    df['Name'] = df.Name.map(name_flags)
    return df

def ticket_preprocess(df):
    
    df['Ticket'] = train_raw.Ticket.str.contains('[A-Z a-z]')
    return df
    
def fare_preprocess(df):
    
    
    df['family_size'] = df['Parch'] + df['SibSp'] + 1
    df['avg_ticket'] = df['Fare'] / df['family_size']
    return df
    
class ColumnProcess(TransformerMixin):
    
    def __init__(self, func):
        self.func = func
        
    def transform(self, df):
        return self.func(df)

    def fit(self, *_):
        return self

    
categorical_transformer = Pipeline([
    ('cat_imputer', SimpleImputer(strategy ='constant', fill_value ='missing', add_indicator = True)),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))
])
    
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
    
numeric_features = ['Age', 'Fare', 'cabin_count', 'Parch', 'Sibsp']
categorical_features = ['Pclass', 'Embarked', 'cabin_type', 'Sex', 'Name']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
        #,('num', numeric_transformer, numeric_features)
])

cd = ColumnDropper(['PassengerId'])
ccs = ColumnProcess(cabin_preprocess)
name = ColumnProcess(name_preprocess)
ticket = ColumnProcess(ticket_preprocess)
fare = ColumnProcess(fare_preprocess)
cd2 = ColumnDropper(categorical_features, convert = True)

impute_rf = ExtraTreesRegressor(n_estimators=100)
num_imputer = IterativeImputer(estimator = impute_rf, initial_strategy ='median')
#num_imputer = SimpleImputer(strategy ='median')
 
y = train_raw['Survived']
train_raw = train_raw.drop(columns = 'Survived')

x_train, x_test, y_train, y_test = train_test_split(train_raw, y, test_size = .3, shuffle = True)


rf = RandomForestClassifier()
ada = AdaBoostClassifier()
xgb = xgb.XGBClassifier(silent=1, nthread = -1)


#my_data = model_training.fit_transform(train_raw, y)
"""
rf_grid = {
 #  'num_imputer__n_nearest_features': [None,2,5],
    'clf__min_samples_leaf': [2,5],
    'clf__min_samples_split': [5],
    'clf__max_depth' : [10, 20,],
    'clf__n_estimators': [500]
}

xgb_grid = {
    'num_imputer__n_nearest_features': [5],
    'num_imputer__add_indicator': [False, True],
    'xgb__learning_rate': [.02, .05],
    'xgb__max_depth': list(range(4,7,1)),
    'xgb__n_estimators': [100,200,300],
    'xgb__colsample_bytree' : [.6],
    'clf__reg_lambda': [.1],
    'clf__subsample': [.6]
}

ada_grid = {
    'num_imputer__n_nearest_features': [5],
    'num_imputer__add_indicator': [False, True],
    'clf__learning_rate': [.1, .5, 1],
    'clf__n_estimators': [100,200,300],
}
"""

composite_grid = {
 #  'num_imputer__n_nearest_features': [None,2,5],
    #'num_imputer__n_nearest_features': [5],
    #'num_imputer__add_indicator': [False, True],
    'voter__rf__min_samples_leaf': [2,5],
    'voter__rf__min_samples_split': [5],
    'voter__rf__max_depth' : [10, 20,],
    'voter__rf__n_estimators': [500],

    'voter__xgb__learning_rate': [.05],
    'voter__xgb__max_depth': list(range(4,5,1)),
    'voter__xgb__n_estimators': [300],
    'voter__xgb__colsample_bytree' : [.6],
    'voter__xgb__reg_lambda': [.1],
    'voter__xgb__subsample': [.6],

    'voter__ada__learning_rate': [1],
    'voter__ada__n_estimators': [300]
}

models = [('xgb', xgb), ('rf', rf), ('ada', ada)]
voter = VotingClassifier(estimators = models, n_jobs = -1, voting = 'soft')

model_training = Pipeline([
        ('cder', cd),
        ('ccs', ccs),
        ('name', name),
        ('ticket', ticket),
        ('fare', fare),
        ('union', FeatureUnion([('preprocess', preprocessor), ('cd2', cd2)])),
        #('cat_imputer', SimpleImputer(strategy ='constant', fill_value ='missing', add_indicator = True)),
        #('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False)),
        ('num_imputer', num_imputer),
        #('debug', Debug()),
        ('voter', voter)
])

cv = GridSearchCV(model_training, composite_grid, cv = 10, n_jobs = -1)
cv.fit(x_train, y_train)

print(cv.best_score_)

preds_test = cv.predict(x_test)
print(sum(preds_test == y_test)/ len(y_test))

holdout_raw = dfs[1]
preds = cv.predict(holdout_raw)
