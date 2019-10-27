# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:53:55 2019

@author: pgood
"""

import HelperFuncs as hf
import pickle as pkl
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier,\
    GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


#load the training file into memeory as a pandas DF and create all the splits
#we need for fitting/tuning.
def my_load_data(test_size, shape = (891,12)):
    
    dfs = hf.load_files('titanic data/titanic files')
    
    train_raw = dfs[2]
    hf.shape_check(train_raw, shape, .8, 'Train')
    
    y = train_raw['Survived']
    train_raw = train_raw.drop(columns = 'Survived')
    
     
    x_train, x_test, y_train, y_test = train_test_split(train_raw, y, 
        test_size = test_size,train_size = .7)
    
    return x_train, x_test, y_train, y_test, train_raw, y


#With numerics, we can use the new IterativeImputer, similar to the MICE
#package in R

def num_pipeline(n_estimators):
    
    num_imputer = SimpleImputer(strategy ='median')
    
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
    models = [rf]
    
    
    return list(zip(model_types, models))


#A shape check step is added at key points to make sure the data shape
#is what one would expect

def model_pipeline(grid, categorical_features, imp_estimators, model_types):
    
    raw_pipeline = Pipeline([
            ('cder', hf.ColumnDropper(['PassengerId', 'Ticket'])),
            ('ccs', hf.ColumnProcess(hf.cabin_preprocess)),
            ('name', hf.ColumnProcess(hf.name_preprocess)),
            #('ticket', ColumnProcess(ticket_preprocess)),
            ('fare', hf.ColumnProcess(hf.fare_preprocess)),
            ('union', 
                 FeatureUnion([
                    ('cat_preprocess', catogoricals_pipeline(categorical_features)), 
                    ('cd2', hf.ColumnDropper(categorical_features, convert = True))
                ])),
            ('num_imputer', num_pipeline(imp_estimators)),
            ('rf', RandomForestClassifier())
    ])
    
   
 
    cv = GridSearchCV(raw_pipeline, grid, cv = 10)
    return cv, raw_pipeline

#The next functions serve to save the objects we need in the 
#score_model file

def pickle_files(obj, fname):
    from sklearn.externals import joblib
    
    try:
        joblib.dump(obj, '{}.joblib'.format(fname))
    except:
        print('Problem pickling file {}'.format(fname))
        
    return


def save_objs(objs):
    
    fnames =  ['final_model', 'training_model', 'x_test', 'y_test']
    flist = list(zip(objs, fnames))
    
    for data, name in flist:
        pickle_files(data, name)


def fit_model(grid_params, categorical_features, model_types):
    
    x_train, x_test, y_train, y_test, train_raw, y = my_load_data(test_size = .3)
    
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
    objs = [raw_pipeline]
    save_objs(objs)

    return
        


categorical_features = ['Pclass', 'Embarked', 'cabin_type', 'Sex', 'Name']
model_types = ['xg', 'rf', 'ada']

#Small grid so that running the file is quick.  In practice, this grid would
#include hundreds of hyperparameter combinations.
composite_grid = {
    'rf__min_samples_leaf': [2,5],
    'rf__min_samples_split': [5],
    'rf__max_depth' : [20],
    'rf__n_estimators': [500],
}

fit_model(composite_grid, categorical_features, model_types)