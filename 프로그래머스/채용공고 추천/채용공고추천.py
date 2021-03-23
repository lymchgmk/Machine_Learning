import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import csv


with open('./data/test_job.csv') as job_tags_csv:
    reader = csv.reader(job_tags_csv, delimiter=',')
    print(next(reader))
    print(reader)
    # for row in reader:
    #     print(row)
        
        
train, test = train_test_split(reader, test_size=0.2)
print(train)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense()
])