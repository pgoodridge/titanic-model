# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:53:55 2019

@author: pgood
"""

from HelperFuncs import *
import pickle as pkl
import xgboost as xgb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


#load the training file into memeory as a pandas DF and create all the splits
#we need for fitting/tuning.
def load_data(test_size, shape = (891,12)):
    
    dfs = load_files('titanic data/titanic files')
    
    train_raw = dfs[2]
    shape_check(train_raw, shape, .8, 'Train')
    
    y = train_raw['Survived']
    train_raw = train_raw.drop(columns = 'Survived')
    
     
    x_train, x_test, y_train, y_test = train_test_split(train_raw, y, 
        test_size = test_size, shuffle = True)
    
    return x_train, x_test, y_train, y_test, train_raw, y

#Cabins associated with a ticket are separated by a space on the Cabin line.  
#There might be correlation between the number of Cabins in a group (large families)
#and survival
def count_cabins(col):
    
    try:
        return len(col.split())
    except:
        return 0
    
#The Cabin also has a letter associated with it, likely indicating the area
#of the ship it is in.  We definitely want that data.

def cabin_preprocess(df):
    
    df['cabin_count'] = df.Cabin.map(count_cabins)
    df.Cabin = df.Cabin.fillna('u')
    df['cabin_type'] = df.Cabin.map(lambda x: 'u' in x)
    
    return df.drop(columns = 'Cabin')

#We can determine whether a woman is married or not with their salutation.
#Boys could also be identified with the "master" salutation, but that is just
#a combination of age and gender, two existing variables.

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

#Some ticket numbers contain letters.  We flag these in case there is any
#correlation with survival. (not used based on EDA)

def ticket_preprocess(df):
    
    df['Ticket'] = df.Ticket.str.contains('[A-Z a-z]')
    
    return df
    
#We want to determine the full party size.  Since 1 is added, it isn't just
#a linear combination of Parch and SibSp.
#The fare variable is the combined fare for everyone in the party.  We
#create a variable for fare per person.  This should be better at identifying
#wealthy passengers
    
def fare_preprocess(df):
    
    df['family_size'] = df['Parch'] + df['SibSp'] + 1
    df['Fare'] = df['Fare'] / df['family_size']
    
    return df

#With numerics, we can use the new IterativeImputer, similar to the MICE
#package in R

def num_pipeline(n_estimators):
    
    impute_rf = ExtraTreesRegressor(n_estimators=n_estimators)
    num_imputer = IterativeImputer(estimator = impute_rf, 
        initial_strategy ='median')
    
    return num_imputer

#With categoricals, we need to impute and one-hot encode them.  The iterative
#imputer didn't seem to work with categoricals so we used the simpleimputer.

def catogoricals_pipeline(categorical_features):
    
    categorical_transformer = Pipeline([
        ('cat_imputer', SimpleImputer(strategy ='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))
    ])
        

    categorical_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
    ])
    
    return categorical_preprocessor


#We are going to use the voting classifier ensemble method, so we need to
#instantiate a list of models.  change the models and names to try a different
#ensemble
    
def model_ensemble(model_types):
    
    rf = RandomForestClassifier(n_estimators = 10)
    ada = AdaBoostClassifier()
    xg = xgb.XGBClassifier(silent=1, nthread = -1)
    models = [xg, rf, ada]
    
    
    return list(zip(model_types, models))


#A shape check step is added at key points to make sure the data shape
#is what one would expect

def model_pipeline(grid, categorical_features, imp_estimators, model_types):
    
    raw_pipeline = Pipeline([
            ('cder', ColumnDropper(['PassengerId', 'Ticket'])),
            ('cdebug', Debug()),
            ('ccs', ColumnProcess(cabin_preprocess)),
            ('name', ColumnProcess(name_preprocess)),
            #('ticket', ColumnProcess(ticket_preprocess)),
            ('fare', ColumnProcess(fare_preprocess)),
            ('union', 
                 FeatureUnion([
                    ('cat_preprocess', catogoricals_pipeline(categorical_features)), 
                    ('cd2', ColumnDropper(categorical_features, convert = True))
                ])),
            ('udebug', Debug()),
            ('num_imputer', num_pipeline(imp_estimators)),
            ('idebug', Debug()),
            ('voter', VotingClassifier(estimators = model_ensemble(model_types), 
                n_jobs = -1, voting = 'soft'))
    ])
    
   
 
    cv = GridSearchCV(raw_pipeline, grid, cv = 10, n_jobs = -1)
    return cv, raw_pipeline

#The next functions serve to save the objects we need in the 
#score_model file

def pickle_files(obj, fname):
    
    try:
        with open('{}.pickle'.format(fname), 'wb') as f:
            pkl.dump(obj, f)
    except:
        print('Problem pickling file {}'.format(fname))
        
    return


def save_objs(objs):
    
    fnames =  ['final_model', 'training_model', 'x_test', 'y_test']
    flist = list(zip(objs, fnames))
    
    for data, name in flist:
        pickle_files(data, name)


def fit_model(grid_params, categorical_features, model_types):
    
    x_train, x_test, y_train, y_test, train_raw, y = load_data(test_size = .3)
    
    #Fit a training model for hyperparameter optimization and scoring on the
    #training set (including a test set derived from the training set).
    print("Running cross validation...")
    cv, raw_pipeline = model_pipeline(composite_grid, 
        categorical_features, 100, model_types)
    cv.fit(x_train, y_train)
    
    print('Cross Validation Accuracy: {:.3f}'.format(cv.best_score_))
    
    #Fit the final model using the full training dataset and the optimized
    #hyperparameters.
    
    print("Fitting final model...")
    best_params = cv.best_params_
    raw_pipeline.set_params(**best_params)
    raw_pipeline.fit(train_raw, y)
    
    #To fullfill the requirement of scoring on the score_data file,
    #we need to save both models and the randomly chosen x_test and 
    #y_test splits.  I would prefer scoring the training data here and the
    #testing data on the score_model script.
    objs = [raw_pipeline, cv, x_test, y_test]
    save_objs(objs)

    return
        


categorical_features = ['Pclass', 'Embarked', 'cabin_type', 'Sex', 'Name']
model_types = ['xg', 'rf', 'ada']

#Small grid so that running the file is quick.  In practice, this grid would
#include hundreds of hyperparameter combinations.
composite_grid = {
    'union__cat_preprocess__cat__cat_imputer__add_indicator': [False, True],
    'num_imputer__n_nearest_features': [5],
    'num_imputer__add_indicator': [False],
    'voter__rf__min_samples_leaf': [2,5],
    'voter__rf__min_samples_split': [5],
    'voter__rf__max_depth' : [20],
    'voter__rf__n_estimators': [500],

    'voter__xg__learning_rate': [.05],
    'voter__xg__max_depth': list(range(4,5,1)),
    'voter__xg__n_estimators': [300],
    'voter__xg__colsample_bytree' : [.6],
    'voter__xg__reg_lambda': [.1],
    'voter__xg__subsample': [.6],

    'voter__ada__learning_rate': [1],
    'voter__ada__n_estimators': [300]
}

fit_model(composite_grid, categorical_features, model_types)