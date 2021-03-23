import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import csv


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy,
        metrics=['accuracy']
    )
    
    return model


model = get_compiled_model()
model.fit(train_dataset, epochs=100)