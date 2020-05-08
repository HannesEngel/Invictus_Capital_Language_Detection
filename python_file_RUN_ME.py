# install the relevant libraries
# import numpy as np
import pandas as pd
import re
# import pickle as pk
import json

from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json

"""
    1. import and convert the data on which the model will be tested
"""
# import the vocabulary from the text file 'vocab.txt'
with open('vocab.txt', 'r') as file:
    vocab_to_int = json.load(file)

# creating another dictionary to convert our languages (strings) to integers
languages_to_int = {'English': 0, 'Afrikaans': 1, 'Nederlands': 2}

# read the CSV file to be tested
test_df = pd.read_csv('lang_data_test.csv')
print("Data has been read.")

# function that cleans the data. This function creates an X_test_pad array. 
# This array is used to create predictions.
# This function also creates a y_test_encoded array against which the predicted values will be compared.
def clean_and_engineer_data(df):

    # additional helper functions
    def process_sentence(sentence): # removes all special characters, strip white space and make lowercase
        return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|-]', '', sentence.lower().strip())

    def convert_to_int(data, data_int):
        """
            converts all our text to integers
            :param data: The text to be converted
            :return: All sentences in ints
        """
        all_items = []
        for sentence in data: 
            all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])
        
        return all_items

    X_test = df['text'].apply(process_sentence)
    y_test = df['language']

    '''
        creating the encoded arrays, X_test_encoded and y_test_encoded
    '''
    
    # X data is encoded
    X_test_encoded = convert_to_int(X_test, vocab_to_int)

    # y data is one hot encoded
    y_data = convert_to_int(y_test, languages_to_int)
    enc = OneHotEncoder()
    enc.fit(y_data)
    y_test_encoded = enc.fit_transform(convert_to_int(y_test, languages_to_int)).toarray()

    '''
        creating the padded array X_test_pad
    '''
    # hyperparameters
    max_sentence_length = 200 # some room for having longer pieces of text
    embedding_vector_length = 300
    dropout = 0.5
    X_test_pad = sequence.pad_sequences(X_test_encoded, maxlen=max_sentence_length)
    print('Cleaning the data.')
    return X_test_pad, y_test_encoded

# calling the above cleaning function to clean and engineer the data
test_data_to_model = clean_and_engineer_data(test_df)
[X_test_pad, y_test_encoded] = [test_data_to_model[0], test_data_to_model[1]]

"""
    2. load the model
"""
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk.")

"""
    3. evaluate the model on the data provided
"""
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
scores = loaded_model.evaluate(X_test_pad, y_test_encoded, verbose=0)

acc = "{:.2f}".format(scores[1]*100)
print(f"The loaded LSTM Neural Network achieves an overall accuracy of {acc}% on the provided text records.")