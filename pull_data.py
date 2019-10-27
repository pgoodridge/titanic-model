# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:20:56 2019

@author: pgood
"""
import os
import zipfile
from HelperFuncs import shape_check, load_files
import json

#This allows the user to change the dataset name if Kaggle has moved the location
#of the Titanic contest
def choice_logic(choice):
    
    if choice.lower() in ['Yes', 'y']:
        
        return 'titanic', 'titanic data'
    
    else:
        
        contest = input('Please enter contest name: ')
        dataset = input('Please enter dataset name: ')
        
        return contest, dataset

#Use the environment variables instead of worrying about the kaggle.json
#file.  Based on line 142 from the authenticate method from the 
#kaggle_extended_api file, the module will first try to authenticate
#with environment variables.  Only if they're not located does it use
#kaggle.json

def get_user_info():
    
    username = input('Enter your username: ')
    key = input('Enter your Kaggle API key: ')
    choice = input('Is the Titanic dataset still in the "titanic" contest \nand "titanic data" dataset? (y/n)')
            
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    return choice
    
#When run from the command line, all of these will actions will throw an error 
#if the user isn't logged in.
#Add parameters to allow for use with other datasets
def kaggle_actions(contest = "titanic", dataset = 'titanic data'):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(contest, dataset)
        return 
    
#Loop until the user succesfully downloads the files
while True:
    try:
        choice = get_user_info()
        contest, dataset = choice_logic(choice)
        kaggle_actions(contest, dataset)
        break
    
    except IOError:
        print('Download failed.  Check your /n permissions/disk space.')
    except OSError:
        print('Download failed.  Check your /n permissions/disk space.')
    except PermissionError:
        print('Download failed.  Check your /n permissions/disk space.')
        
    except:#APIException from the Kaggle package causes an error if imported
        #before the user logs.  The best we can do is guess on any APIException errors
        print('Download failed.\n Please ensure you have the correct Kaggle key and contest information.')
        pass

#unzip
with zipfile.ZipFile('titanic data/titanic.zip', 'r') as zip_ref:
    zip_ref.extractall('titanic data/titanic files')

#load data from the local filesystem
dfs = load_files('titanic data/titanic files')
data = dfs[1].to_json(orient = 'records')


with open('test.json', 'w') as f:
    json.dump(data, f)

#confirm the data loaded correctly (see HelperFuns.py for more details)
shape_check(dfs[2], (891,12), .8, 'Train')
shape_check(dfs[1], (418,11), .8, 'Test')
