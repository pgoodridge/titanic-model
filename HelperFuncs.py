# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:19:43 2019

@author: pgood
"""
from sklearn.base import TransformerMixin, BaseEstimator

###########Anything needed in more than one file is included here#############


#Sum missing values in a given 2D dataset.  Works with both pandas DFs
#and numpy arrays
def na_count(table, cells):
    
    import pandas as pd 
    import numpy as np

    if type(table) == pd.core.frame.DataFrame:
        return sum(table.count())/cells
        
    elif type(table) == np.ndarray:
        return 1 - np.count_nonzero(np.isnan(table))/cells 
        

#This is our main datachecking device.  It takes in an expected shape and 
#threshold for the ratio of NA values and throws an error if
#either of these assumptions is violated.
def shape_check(table, shape, threshold, data_name):
    
    from nose.tools import assert_equal
    
    cells = shape[0]*shape[1] 
    na_ratio = na_count(table, cells)
        
    assert_equal(table.shape, shape)
    assert(na_ratio > threshold)#I couldn't find an inequality assertion in nose.tools
    
    print("{} loaded correctly".format(data_name))
    
    return

#loads all csv files in a director into a list of pandas DFs.  If we were 
#working with bigger data, it would be benificial to allow the user to 
#pick particular files        
def load_files(path):
    import pandas as pd
    import os
    
    dfs = []
    for path, dirs, files in os.walk(path):
        for file in sorted(files):
            if '.csv' in file:
                dfs.append(pd.read_csv(os.path.join(path, file)))
    return dfs

#############Helper functions repeated from train_model.py######################
#these are here so that score_model has access to them.  For descriptions
#see the train_model.py file
def count_cabins(col):
    
    try:
        return len(col.split())
    except:
        return 0
    
def cabin_preprocess(df):

    import numpy as np
    import pandas as pd
    
    if type(df) == np.ndarray:
        df = pd.DataFrame(df)
    
    df['cabin_count'] = df.Cabin.map(count_cabins)
    df.Cabin = df.Cabin.fillna('u')
    df['cabin_type'] = df.Cabin.map(lambda x: 'u' in x)
    
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
    
    
    df['Ticket'] = df.Ticket.str.contains('[A-Z a-z]')
    
    return df
    
def fare_preprocess(df):
    
    df['family_size'] = df['Parch'] + df['SibSp'] + 1
    df['Fare'] = df['Fare'] / df['family_size']
    
    return df

#####################Generic pipeline helper Classes#########################
    

#Exactly as the name describes, but only works with pandas DFs
class ColumnDropper(TransformerMixin):
    

    
    def __init__ (self, cols, convert = False):
        self.cols = cols
        self.convert = convert
        
    def transform(self, df):
        
        import numpy as np
        import pandas as pd
    
        if type(df) == np.ndarray:
            return pd.DataFrame(df).drop(columns = self.cols)
        else:
            return df.drop(columns = self.cols)
    
    def fit(self, *_):
        return self

#Takes a function and applies it to a pandas DF.  Very useful for keeping
#track of our preprocessing steps.
class ColumnProcess(TransformerMixin):
    
    def __init__(self, func):
        self.func = func
        
    def transform(self, df):
        return self.func(df)

    def fit(self, *_):
        return self