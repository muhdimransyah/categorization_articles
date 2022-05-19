# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:25:24 2022

@author: imran
"""

import os,datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from assessment3_module import DetailAnalysis,ModelImplementation,ModelEvaluation

# Link need to be checked again from github
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKEN_SAVE_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(), 'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

df = pd.read_csv(URL)
rev = df['category']
sent = df['text']

dna = DetailAnalysis()
review = dna.remove_tags(rev)
review = dna.lower_split(rev)

rev = dna.sent_tokenizer(rev, TOKEN_SAVE_PATH)
rev = dna.sent_pad_sequence(rev)

one_hot_encoder = OneHotEncoder(sparse=False)
sent = one_hot_encoder.fit_transform(np.expand_dims(sent,axis=-1))

nb_categories = len(np.unique(sent))

X_train, X_test, y_train, y_test = train_test_split(rev, 
                                                    sent, 
                                                    test_size=0.3, 
                                                    random_state=123)

X_train = np.expand_dims(X_train,axis=1)
x_test = np.expand_dims(X_test,axis=1)

print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0],axis=0)))

mc = ModelImplementation()
words = 10000
model = mc.lstm_layer(words, nb_categories)

log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

#%% Model Fit

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='acc')
model.fit(X_train,
          y_train,
          epochs=3,
          validation_data=(X_test,y_test),
          callbacks=tensorboard_callback)

predicted_advanced = np.empty([len(X_test), 2])
for index, i in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(i,axis=0))
    
y_pred = np.argmax(predicted_advanced,axis=1)
y_true = np.argmax(y_test,axis=1)

me = ModelEvaluation()
me.report_metrics(y_true,y_pred)

model.save(MODEL_SAVE_PATH)