import numpy as np
import tensorflow as tf
import pandas as pd
import csv

train = pd.read_csv('data/train_job/train.csv')
user_tags = pd.read_csv('data/train_job/user_tags.csv')
job_tags = pd.read_csv('data/train_job/job_tags.csv')

# find = user_tags.loc[user_tags['userID'] == 'f32196295a6f0c13d5bedc880d3d66a2']
# print(find)

for index, row in train.iterrows():
    print(user_tags.loc['userID'] == row['userID'])
    

#
# train, test = train_test_split(reader, test_size=0.2)
# print(train)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense()
# ])