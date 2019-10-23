# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:41:11 2019

@author: pgood
"""

import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve
from HelperFuncs import *

    
#The next two functions take in the files/data we need, check it, and
#return it.  

def pickle_load(fname):
    
    with open('{}.pickle'.format(fname), 'rb') as f:
        return pkl.load(f)

def load_data():
    
    #Data from .CSV files
    dfs = load_files('titanic data/titanic files')
    holdout_raw = dfs[1]
    shape_check(holdout_raw, (418,11), .8, 'Holdout')
    
    #Pickled objects from the train_model.py file
    x_test = pickle_load('x_test')
    y_test = pickle_load('y_test')
    cv = pickle_load('training_model')
    final_model = pickle_load('final_model')
    
    return holdout_raw, x_test, y_test, cv, final_model

#Plots and saves a ROC curve from our test set
def plot_roc(labels, probs):
    
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.show()
    plt.savefig('ROC curve.png')
    
    return

#Combines the ROC curve with classification report
def full_classification_report(labels, probs, pred_labels):
    
    plot_roc(labels, probs)
    report = classification_report(y_test, preds_test, output_dict = True, 
                               target_names = ['Perished', 'Survived'])
    
    print('Classification Report:\n\n')
    print(report)
    
    df_report = pd.DataFrame(report)
    df_report.to_csv('Classification Report.csv')
    
    return df_report
    

#load our data/models
holdout_raw, x_test, y_test, cv, final_model = load_data()

#Predict with the test set we created
preds_test = cv.predict(x_test)
preds_probs = cv.predict_proba(x_test)[:,1]

#Get a ROC curve and classfication report from our test set
df_report = full_classification_report(y_test, preds_probs, preds_test) 

#Predict on the Kaggle testing data and save it
preds = pd.DataFrame(final_model.predict(holdout_raw))
preds.to_csv('Predictions.csv')