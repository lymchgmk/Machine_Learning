# userID jobID tag일치도 applied


import numpy as np
import tensorflow as tf
import pandas as pd
import csv


train = pd.read_csv('data/train_job/train.csv')
user_tags = pd.read_csv('data/train_job/user_tags.csv')
job_tags = pd.read_csv('data/train_job/job_tags.csv')

for index, row in train.iterrows():
    userID, jobID, applied = row['userID'], row['jobID'], row['applied']
    key_user_tag = user_tags['userID'] == userID
    key_job_tag = job_tags['jobID'] == jobID
    user_tag = user_tags[key_user_tag]['tagID']
    job_tag = job_tags[key_job_tag]['tagID']
    
    for ut in user_tag:
        if ut in job_tag:
            print(ut)
   
   
