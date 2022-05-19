# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:28:54 2022

@author: imran
"""

import re,json,os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#%% Classes

class DetailAnalysis():
    
    def remove_tags(self,data):
        for index, text in enumerate(data):
            data[index] = re.sub('<.*?>', '', text)
        return data
    
    def lower_split(self,data):
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]',' ',text).lower().split()
            
        return data
            
    def sent_tokenizer(self,data,token_save_path,
                            num_words=1000,
                            oov_token='<oov>', 
                            prt=False):
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        # To save the tokenizer for deployment purpose
        token_json = tokenizer.to_json()
        
        with open(token_save_path,'w') as json_file:
            json.dump(token_json,json_file)
        
        # To observe number of words
        word_index = tokenizer.word_index
        
        if prt == True:
            #print(word_index)
            print(dict(list(word_index.items())[0:10]))
        
        # To vectorize the sequences of the text
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def sent_pad_sequence(self,data):
        return pad_sequences(data,maxlen=200,padding='post',truncating='post')
    
class ModelImplementation():
    
   # def __init__(self,train):
   #     self.train = train
    
    def lstm_layer(self, num_words,category, embedding_output=64, nodes=32, dropout=0.2):
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(nodes))
        model.add(Dropout(dropout))
        model.add(Dense(category,activation='softmax'))
        model.summary()
        
        return model
    
    def simple_lstm_layer(self, num_words,category, embedding_output=64, nodes=32, dropout=0.2):
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(category,activation='softmax'))
        model.summary()
        
        return model

class ModelEvaluation():
    
        def report_metrics(self,y_true,y_pred):
            print(classification_report(y_true,y_pred))
            print(confusion_matrix(y_true,y_pred))
            print(accuracy_score(y_true,y_pred))
            
#%%  

if __name__ == '__main__':
    
    URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
    LOG_PATH = os.path.join(os.getcwd(), 'log')
    MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
    TOKENIZER_JSON_PATH = os.path.join(os.getcwd(),'tokenizer.data.json')
    
    df = pd.read_csv(URL)
    rev = df['category']
    sent  = df['text']
    
#%%
    eda = DetailAnalysis()
    test = eda.remove_tags(rev)
    test = eda.lower_split(test)
    
    test = eda.sent_tokenizer(test,token_save_path=TOKENIZER_JSON_PATH)
    test = eda.sent_pad_sequence(test)
    
#%%
    category = len(sent.unique())
    implement = ModelImplementation()
    model = implement.simple_lstm_layer(10000, category)